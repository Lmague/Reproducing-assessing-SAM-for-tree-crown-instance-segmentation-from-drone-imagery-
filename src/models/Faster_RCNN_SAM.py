"""
Faster R-CNN + SAM - Pipeline Hybride

Selon l'Appendix C du papier:
1. Modèle 1 (Détecteur): Faster R-CNN (ResNet-50)
   - Entraînement: Adam (lr=0.0001 pour finetuning, 0.0005 si scratch)
   - Betas=(0.9, 0.999), weight_decay=0.0005
   - Scheduler exponentiel (update tous les 10 epochs)
   - Batch size 32 (RGB) ou 16 (RGB+DSM)
   
2. Inférence: Passer les bounding boxes prédites par Faster R-CNN comme prompts à SAM

3. Score final: Moyenne du score de sortie de Faster R-CNN et du score IoU prédit par SAM
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON,
    TEST_RGB as TEST_IMG_DIR, SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_faster_rcnn")
FRCNN_CHECKPOINT = os.path.join(OUTPUT_DIR, "model_final.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 1. FASTER R-CNN TRAINING
# ============================================================================

def register_datasets():
    """Enregistre les datasets au format Detectron2."""
    try:
        register_coco_instances("trees_train", {}, TRAIN_JSON, DATA_ROOT)
        register_coco_instances("trees_val", {}, VAL_JSON, DATA_ROOT)
        register_coco_instances("trees_test", {}, TEST_JSON, DATA_ROOT)
    except:
        pass  # Déjà enregistré


def build_faster_rcnn_cfg(num_classes: int, use_dsm: bool = False) -> "CfgNode":
    """
    Configuration Faster R-CNN selon l'Appendix C.
    
    Args:
        num_classes: Nombre de classes (hors background)
        use_dsm: Si True, utilise RGB+DSM (4 canaux), sinon RGB seul (3 canaux)
    """
    cfg = get_cfg()
    
    # Config de base Faster R-CNN ResNet50 FPN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Datasets
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST = ("trees_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Poids pré-entraînés
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # --- HYPERPARAMÈTRES DU PAPIER (Appendix C) ---
    
    # "Batch size 32 (RGB) ou 16 (RGB+DSM)" - Adaptés pour A40 (48GB)
    cfg.SOLVER.IMS_PER_BATCH = 2 if use_dsm else 4  # Papier: 16/32, A40: 4/8, RTX3090: 2/4
    
    # "Adam optimizer (lr=0.0001 for finetuning)"
    # Règle linéaire pour batch réduit
    base_batch = 16 if use_dsm else 32
    actual_batch = cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = 0.0001 * (actual_batch / base_batch)
    cfg.SOLVER.MOMENTUM = 0.9  # Sera ignoré si on utilise Adam
    
    # "weight_decay=0.0005"
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    
    # "linear warmup"
    cfg.SOLVER.WARMUP_FACTOR = 0.01  # warmup start = 10^-6 / base_lr = 0.01
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    # "trained for a maximum of 100 epochs"
    # Taille réelle du dataset train (18746 images)
    TRAIN_SIZE = 18746
    iters_per_epoch = TRAIN_SIZE // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER = 100 * iters_per_epoch
    
    # "Scheduler exponentiel (update tous les 10 epochs)"
    # Avec Detectron2, on utilise step decay comme approximation
    cfg.SOLVER.STEPS = tuple([10 * i * iters_per_epoch for i in range(1, 10)])
    cfg.SOLVER.GAMMA = 0.9  # Decay factor
    
    # Inférence
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Évaluation périodique
    cfg.TEST.EVAL_PERIOD = 500
    
    cfg.MODEL.DEVICE = DEVICE
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


class FasterRCNNTrainer(DefaultTrainer):
    """Trainer Faster R-CNN avec Adam optimizer et augmentations."""
    
    @classmethod
    def build_train_loader(cls, cfg):
        """DataLoader avec RandomFlip augmentation."""
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(),
            ]
        )
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Utilise Adam au lieu de SGD (selon Appendix C).
        "Adam (lr=0.0001 for finetuning), Betas=(0.9, 0.999), weight_decay=0.0005"
        """
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            
            # Réduire le LR pour le backbone (finetuning)
            if "backbone" in key:
                lr = lr * 0.1
            
            params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
        
        optimizer = Adam(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        return optimizer
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    def build_hooks(self):
        ret = super().build_hooks()
        if self.cfg.TEST.EVAL_PERIOD > 0:
            ret.insert(-1, hooks.EvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                lambda: inference_on_dataset(
                    self.model,
                    build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0]),
                    self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
                )
            ))
        return ret


def train_faster_rcnn(num_classes: int):
    """Entraîne Faster R-CNN."""
    print("=" * 60)
    print("🚀 Entraînement Faster R-CNN")
    print("=" * 60)
    
    register_datasets()
    cfg = build_faster_rcnn_cfg(num_classes)
    
    print(f"Configuration:")
    print(f"  - Output: {cfg.OUTPUT_DIR}")
    print(f"  - Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  - Learning rate: {cfg.SOLVER.BASE_LR}")
    print("=" * 60)
    
    trainer = FasterRCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg


# ============================================================================
# 2. FASTER R-CNN + SAM INFERENCE
# ============================================================================

def run_faster_rcnn_sam_inference(faster_rcnn_checkpoint: str, num_classes: int):
    """
    Pipeline hybride: Faster R-CNN détecte les boîtes, SAM segmente.
    
    Selon le papier:
    - Passer les bounding boxes prédites comme prompts à SAM
    - Score final = moyenne(score_faster_rcnn, score_iou_sam)
    """
    print("=" * 60)
    print("🔄 Inférence Faster R-CNN + SAM")
    print("=" * 60)
    
    output_file = os.path.join(BASE_DIR, "faster_rcnn_sam_predictions.json")
    
    # 1. Charger Faster R-CNN
    print("Chargement de Faster R-CNN...")
    register_datasets()
    cfg = build_faster_rcnn_cfg(num_classes)
    cfg.MODEL.WEIGHTS = faster_rcnn_checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Seuil plus bas pour avoir plus de candidats
    
    faster_rcnn = build_model(cfg)
    faster_rcnn.eval()
    DetectionCheckpointer(faster_rcnn).load(faster_rcnn_checkpoint)
    
    # 2. Charger SAM
    print(f"Chargement de SAM ViT-H sur {DEVICE}...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    # 3. Charger les images de test
    with open(TEST_JSON, 'r') as f:
        coco_gt = json.load(f)
    
    images_info = coco_gt['images']
    coco_results = []
    
    print(f"Traitement de {len(images_info)} images...")
    
    for img_info in tqdm(images_info):
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        img_path = os.path.join(TEST_IMG_DIR, os.path.basename(file_name))
        if not os.path.exists(img_path):
            continue
        
        # Lecture image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 4. Détection avec Faster R-CNN
        with torch.no_grad():
            # Préparer l'input pour Detectron2
            height, width = image_bgr.shape[:2]
            image_tensor = torch.as_tensor(image_bgr.astype("float32").transpose(2, 0, 1))
            
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = faster_rcnn(inputs)[0]
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores_frcnn = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        if len(boxes) == 0:
            continue
        
        # 5. Segmentation avec SAM
        predictor.set_image(image_rgb)
        
        for i, box in enumerate(boxes):
            # Box format: [x1, y1, x2, y2]
            box_prompt = box.reshape(1, 4)
            
            masks, iou_preds, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                multimask_output=False
            )
            
            mask = masks[0]  # (H, W)
            iou_score = float(iou_preds[0])
            
            # 6. Score final (Appendix C):
            # "Average of Faster R-CNN output score and SAM IoU predicted score"
            final_score = (scores_frcnn[i] + iou_score) / 2.0
            
            # Encoder le masque en RLE
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Box XYWH
            x1, y1, x2, y2 = box
            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            res = {
                "image_id": image_id,
                "category_id": int(pred_classes[i]) + 1,  # Detectron2: 0-indexed → COCO: 1-indexed
                "segmentation": rle,
                "score": float(final_score),
                "bbox": bbox_xywh,
                "bbox_mode": 1,  # XYWH
            }
            coco_results.append(res)
    
    # Sauvegarde
    with open(output_file, 'w') as f:
        json.dump(coco_results, f)
    
    print(f"Prédictions sauvegardées dans {output_file}")
    print(f"Total: {len(coco_results)} instances détectées")
    
    return coco_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    NUM_CLASSES = 7
    train_faster_rcnn(NUM_CLASSES)
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERREUR: SAM checkpoint not found: {SAM_CHECKPOINT}")
    else:
        run_faster_rcnn_sam_inference(FRCNN_CHECKPOINT, NUM_CLASSES)
