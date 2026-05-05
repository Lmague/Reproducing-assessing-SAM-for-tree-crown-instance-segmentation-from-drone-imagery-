"""
Mask R-CNN (RGB) - Entraînement pour segmentation d'instances d'arbres

Selon l'Appendix C du papier:
- Backbone: ResNet-50 avec FPN
- Entrée: 3 canaux (RGB)
- Initialisation: Poids pré-entraînés ImageNet (via COCO)
- Hyperparamètres:
    - Optimizer: SGD (lr=0.0001, momentum=0.9, weight_decay=0.0005)
    - Scheduler: Linear warmup (start 10^-6)
    - Batch size: 32
    - Epochs: Max 100
- Augmentations: RandomFlip
- Critère de sélection: Meilleur mAP de segmentation sur validation
"""

import os
import json
import torch
from detectron2.engine import DefaultTrainer, DefaultPredictor, hooks, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T

# Shared utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.coco_helpers import fix_coco_json_if_needed
from utils.training_hooks import EarlyStoppingHook
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON, TEST_RGB as TEST_IMG_DIR,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "output_mask_rcnn_RGB")

def register_datasets():
    """
    Enregistre les 3 splits au format Detectron2.
    Corrige aussi les fichiers JSON COCO si nécessaire.
    """
    # Fix les fichiers JSON si 'info' manque (évite KeyError de pycocotools)
    for json_path in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if os.path.exists(json_path):
            fix_coco_json_if_needed(json_path)
    
    try:
        register_coco_instances("trees_train", {}, TRAIN_JSON, DATA_ROOT)
        register_coco_instances("trees_val", {}, VAL_JSON, DATA_ROOT)
        register_coco_instances("trees_test", {}, TEST_JSON, DATA_ROOT)
    except AssertionError:
        # Déjà enregistré
        pass


def build_cfg(num_classes: int) -> "CfgNode":
    """
    Crée la config Detectron2 pour Mask R-CNN RGB standard.
    Paramètres alignés sur l'Appendix C de l'article.
    """
    cfg = get_cfg()
    # Config de base Mask R-CNN ResNet50 FPN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Datasets
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST  = ("trees_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Poids pré-entraînés (COCO/ImageNet) - Appendix C: "ImageNet pretrained weights"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # --- HYPERPARAMÈTRES DU PAPIER (Appendix C) - Adaptés pour A40 (48GB) ---
    
    # 1. Batch Size: "Batch size is 32 for Mask R-CNN"
    # A40: On utilise 8 pour être safe. Pour compenser, on utilise gradient accumulation
    # ou on ajuste le LR avec la règle linéaire: LR_new = LR_paper * (batch_new / batch_paper)
    cfg.SOLVER.IMS_PER_BATCH = 4  # Papier: 32, A40: 8, RTX3090: 4
    
    # 2. Optimizer: "SGD optimizer with learning rate 0.0001, momentum 0.9"
    # Règle linéaire: 0.0001 * (8/32) = 0.000025
    cfg.SOLVER.BASE_LR = 0.0000125  # Ajusté pour batch 4 (RTX3090)
    cfg.SOLVER.MOMENTUM = 0.9
    
    # 3. Weight Decay: "weight decay 0.0005"
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    
    # 4. Warmup: "linear warmup starting at 10^-6"
    # warmup_factor = start_lr / base_lr = 1e-6 / 1e-4 = 0.01
    cfg.SOLVER.WARMUP_FACTOR = 0.01  # Start at LR * 0.01 = 1e-6
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    # 5. Durée entraînement: "trained for a maximum of 100 epochs"
    # Calcul pour 18000 images: 18000 / batch 8 = 2250 iters/epoch
    # 100 epochs = 225000 iters (MAIS avec early stopping on peut s'arrêter avant)
    # On met max 50 epochs = 112500 iters comme limite haute
    cfg.SOLVER.MAX_ITER = 225000
    # LR decay à 60% et 80% du training
    cfg.SOLVER.STEPS = (135000, 180000)
    cfg.SOLVER.GAMMA = 0.1

    # Inférence
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Évaluation périodique - "best model based on segmentation mAP on validation"
    # Tous les 2 epochs environ = 4500 iters
    cfg.TEST.EVAL_PERIOD = 9000

    # Hardware
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

class RGBTrainer(DefaultTrainer):
    """Trainer avec évaluation périodique, augmentations du papier et EARLY STOPPING."""
    
    @classmethod
    def build_train_loader(cls, cfg):
        """DataLoader avec augmentations du papier (RandomFlip)."""
        mapper = DatasetMapper(
            cfg, 
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(),  # Appendix C: "RandomFlip augmentation"
            ]
        )
        return build_detection_train_loader(cfg, mapper=mapper)
    
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
            # EARLY STOPPING: patience de 10 évaluations (~20 epochs)
            ret.append(EarlyStoppingHook(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                patience=10,  # 10 * 2 epochs = 20 epochs sans amélioration -> stop
                metric_name="segm/AP",  # Métrique COCO pour segmentation
                mode="max"
            ))
        return ret


def train(num_classes: int):
    """
    Entraîne Mask R-CNN sur trees_train, évalue sur trees_val.
    """
    print("=" * 60)
    print("🚀 Entraînement Mask R-CNN RGB")
    print("=" * 60)

    register_datasets()
    cfg = build_cfg(num_classes)
    
    print(f"Configuration:")
    print(f"  - Output: {cfg.OUTPUT_DIR}")
    print(f"  - Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  - Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  - Eval period: {cfg.TEST.EVAL_PERIOD} iters (~2 epochs)")
    print(f"  - Early stopping patience: 10 évaluations (~20 epochs)")
    
    # Estimation du temps
    # A40: ~0.3-0.5 sec/iter pour Mask R-CNN batch 8
    iters_per_epoch = 18000 // cfg.SOLVER.IMS_PER_BATCH
    time_per_iter = 0.4  # secondes (estimation conservative)
    estimated_time_50_epochs = (50 * iters_per_epoch * time_per_iter) / 3600
    estimated_cost_50_epochs = estimated_time_50_epochs * 0.40  # $0.40/h
    print(f"\n💰 Estimation coût (50 epochs max):")
    print(f"  - Temps: ~{estimated_time_50_epochs:.1f} heures")
    print(f"  - Coût: ~${estimated_cost_50_epochs:.2f}")
    print(f"  - Avec early stopping (~20-30 epochs): ~${estimated_cost_50_epochs * 0.5:.2f}")
    print("=" * 60)

    trainer = RGBTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Évaluation finale sur validation
    print("\n" + "=" * 60)
    print("📊 Évaluation finale sur validation")
    print("=" * 60)
    evaluator = COCOEvaluator("trees_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "trees_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    
    # Évaluation sur test
    print("\n" + "=" * 60)
    print("📊 Évaluation finale sur test")
    print("=" * 60)
    evaluator_test = COCOEvaluator("trees_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "trees_test")
    print(inference_on_dataset(trainer.model, test_loader, evaluator_test))


def test_inference(num_classes: int, checkpoint_path: str):
    """
    Exemple d’inférence sur le set test avec le modèle appris.
    """

    register_datasets()
    cfg = build_cfg(num_classes)
    cfg.MODEL.WEIGHTS = checkpoint_path  # model_final.pth entraîné
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import cv2

    meta = MetadataCatalog.get("trees_test")

    # charge une image de test
    img_path = os.path.join(TEST_IMG_DIR, os.listdir(TEST_IMG_DIR)[0])
    img = cv2.imread(img_path)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("test_prediction.png", out.get_image()[:, :, ::-1])
    print("Image prédite sauvegardée dans test_prediction.png")

if __name__ == "__main__":
    # nombre de classes de ton COCO (hors background)
    NUM_CLASSES = 7  # ex: single-class "tree_crown" ; sinon mets len(liste_espèces)

    # 1) entraînement + évaluation
    train(NUM_CLASSES)

    # 2) inference exemple (utilise le best checkpoint généré par Detectron2)
    #    adapte le chemin si besoin
    # ckpt = os.path.join(OUTPUT_DIR, "model_final.pth")
    # test_inference(NUM_CLASSES, ckpt)