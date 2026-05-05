"""
Mask R-CNN + SAM - Pipeline Hybride

Selon l'Appendix C du papier:
1. Modèle 1 (Segmentateur): Mask R-CNN (voir Mask_RCNN_RGB.py)

2. Inférence: Passer les BOÎTES ET les MASQUES prédits par Mask R-CNN comme prompts à SAM

3. Score final: Moyenne du score Mask R-CNN et score IoU SAM
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR, SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_RCNN_CHECKPOINT = os.path.abspath(os.path.join(BASE_DIR, "output_mask_rcnn_RGB/model_final.pth"))

OUTPUT_FILE = os.path.join(BASE_DIR, "mask_rcnn_sam_predictions.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def register_datasets():
    """Enregistre les datasets au format Detectron2."""
    try:
        register_coco_instances("trees_test", {}, TEST_JSON, DATA_ROOT)
    except:
        pass


def build_mask_rcnn_cfg(num_classes: int) -> "CfgNode":
    """Configuration Mask R-CNN pour l'inférence."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.WEIGHTS = MASK_RCNN_CHECKPOINT
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Seuil plus bas pour avoir plus de candidats
    cfg.MODEL.DEVICE = DEVICE
    
    return cfg


def mask_to_sam_prompt(mask_binary: np.ndarray) -> np.ndarray:
    """
    Convertit un masque binaire en prompt de masque pour SAM.
    SAM attend un masque de taille (256, 256) comme prompt dense.
    
    Args:
        mask_binary: Masque binaire (H, W)
    
    Returns:
        Masque redimensionné pour SAM (256, 256)
    """
    # Redimensionner à 256x256 pour SAM
    mask_resized = cv2.resize(
        mask_binary.astype(np.float32), 
        (256, 256), 
        interpolation=cv2.INTER_LINEAR
    )
    return mask_resized


def run_mask_rcnn_sam_inference(num_classes: int):
    """
    Pipeline hybride: Mask R-CNN détecte et segmente, SAM affine.
    
    Selon le papier:
    - Passer les bounding boxes ET les masques comme prompts à SAM
    - Score final = moyenne(score_mask_rcnn, score_iou_sam)
    """
    print("=" * 60)
    print("🔄 Inférence Mask R-CNN + SAM")
    print("=" * 60)
    
    # Vérifications
    if not os.path.exists(MASK_RCNN_CHECKPOINT):
        print(f"ERREUR: Checkpoint Mask R-CNN non trouvé: {MASK_RCNN_CHECKPOINT}")
        print("Entraînez d'abord Mask R-CNN avec Mask_RCNN_RGB.py")
        return
    
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERREUR: Checkpoint SAM non trouvé: {SAM_CHECKPOINT}")
        return
    
    # 1. Charger Mask R-CNN
    print("Chargement de Mask R-CNN...")
    register_datasets()
    cfg = build_mask_rcnn_cfg(num_classes)
    
    mask_rcnn = build_model(cfg)
    mask_rcnn.eval()
    DetectionCheckpointer(mask_rcnn).load(MASK_RCNN_CHECKPOINT)
    
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
        height, width = image_bgr.shape[:2]
        
        # 4. Détection et Segmentation avec Mask R-CNN
        with torch.no_grad():
            image_tensor = torch.as_tensor(image_bgr.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = mask_rcnn(inputs)[0]
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        masks_mrcnn = instances.pred_masks.numpy()  # (N, H, W) bool
        scores_mrcnn = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        if len(boxes) == 0:
            continue
        
        # 5. Affinage avec SAM
        predictor.set_image(image_rgb)
        
        for i in range(len(boxes)):
            box = boxes[i]  # [x1, y1, x2, y2]
            mask_mrcnn = masks_mrcnn[i]  # (H, W) bool
            score_mrcnn = scores_mrcnn[i]
            
            # Préparer les prompts pour SAM
            box_prompt = box.reshape(1, 4)
            
            # Convertir le masque Mask R-CNN en prompt pour SAM
            # Le masque doit être en logits (valeurs continues) 
            mask_prompt = mask_to_sam_prompt(mask_mrcnn.astype(np.uint8))
            # Convertir en logits-like: masque binaire -> valeurs continues
            mask_prompt = (mask_prompt * 2.0 - 1.0) * 10.0  # Scaling pour logits
            mask_prompt = mask_prompt[None, :, :]  # (1, 256, 256)
            
            # Prédiction SAM avec box + mask prompts
            masks_sam, iou_preds, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                mask_input=mask_prompt,  # Prompt masque!
                multimask_output=False
            )
            
            mask_sam = masks_sam[0]  # (H, W) bool
            iou_score = float(iou_preds[0])
            
            # 6. Score final (Appendix C):
            # "Average of Mask R-CNN output score and SAM IoU predicted score"
            final_score = (score_mrcnn + iou_score) / 2.0
            
            # Encoder le masque en RLE
            rle = mask_util.encode(np.asfortranarray(mask_sam.astype(np.uint8)))
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
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco_results, f)
    
    print(f"Prédictions sauvegardées dans {OUTPUT_FILE}")
    print(f"Total: {len(coco_results)} instances détectées")
    
    return coco_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    NUM_CLASSES = 7  # Nombre de classes d'arbres
    
    run_mask_rcnn_sam_inference(NUM_CLASSES)
