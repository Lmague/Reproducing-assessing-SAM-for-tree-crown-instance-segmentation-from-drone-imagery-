"""
SAM Automatic Mask Generation - Out-of-the-Box Inference

Paper (Appendix C) setup:
- Backbone: ViT-H
- Mode: automatic mask generation
- Points per side: tested values below
- Post-processing: NMS with score=0.5 and IoU=0.5
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms
import pycocotools.mask as mask_util

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR, SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POINTS_PER_SIDE_OPTIONS = [10, 5, 3]

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_nms_to_masks(masks, score_threshold=0.5, iou_threshold=0.5):
    """
    NMS over SAM masks.

    Args:
        masks: list of dicts with 'segmentation', 'predicted_iou', 'bbox', etc.
        score_threshold: minimum predicted IoU to keep
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered list of masks after NMS.
    """
    if len(masks) == 0:
        return []
    
    # Score filter
    masks = [m for m in masks if m['predicted_iou'] >= score_threshold]
    
    if len(masks) == 0:
        return []
    
    # Convert SAM bbox [x, y, w, h] -> [x1, y1, x2, y2] for NMS
    boxes = []
    scores = []
    for m in masks:
        x, y, w, h = m['bbox']
        boxes.append([x, y, x + w, y + h])
        scores.append(m['predicted_iou'])
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=DEVICE)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE)
    
    # Run NMS
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    
    # Keep selected masks
    return [masks[i] for i in keep_indices.cpu().numpy()]


def run_inference(points_per_side: int):
    """
    Run SAM automatic mask generation for a given grid size.

    Args:
        points_per_side: number of prompt points per side

    Returns:
        COCO-format predictions list.
    """
    output_file = os.path.join(BASE_DIR, f"sam_otb_pps{points_per_side}_predictions.json")
    
    print(f"\n{'='*60}")
    print(f"SAM Automatic - Points per Side: {points_per_side}")
    print(f"{'='*60}")
    
    # Load SAM (ViT-H)
    print(f"Chargement de SAM ViT-Huge sur {DEVICE}...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    
    # Automatic mask generator config (Appendix C)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,  # 100 (défaut) ou 10
        points_per_batch=64,
        pred_iou_thresh=0.0,  # On filtre après avec NMS
        stability_score_thresh=0.0,  # On filtre après avec NMS
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
    )

    # Load test metadata
    with open(TEST_JSON, "r") as f:
        coco_gt = json.load(f)

    images_info = coco_gt["images"]
    print(f"Images à traiter : {len(images_info)}")

    coco_results = []

    # Inference
    print("Début de l'inférence...")

    with torch.inference_mode():
        for img_info in tqdm(images_info):
            file_name = img_info["file_name"]
            image_id = img_info["id"]

            # Robust path handling
            img_path = os.path.join(TEST_IMG_DIR, os.path.basename(file_name))
            if not os.path.exists(img_path):
                alt_path = os.path.join(DATA_ROOT, file_name)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    continue

            # Read image
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Generate masks
            masks = mask_generator.generate(image_rgb)
            
            # NMS post-process (Appendix C)
            masks = apply_nms_to_masks(masks, SCORE_THRESHOLD, IOU_THRESHOLD)

            # Convert to COCO results format
            for m in masks:
                # Encode mask as RLE
                mask_binary = m["segmentation"].astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask_binary))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # SAM bbox is [x, y, w, h]
                bbox = m["bbox"]
                
                res = {
                    "image_id": image_id,
                    "category_id": 1,  # Single class "Tree"
                    "segmentation": rle,
                    "score": float(m["predicted_iou"]),
                    "bbox": [float(b) for b in bbox],
                    "bbox_mode": 1,  # XYWH_ABS
                }
                coco_results.append(res)

    # Save predictions
    with open(output_file, "w") as f:
        json.dump(coco_results, f)

    print(f"Terminé. Prédictions sauvegardées dans {output_file}")
    print(f"Total: {len(coco_results)} instances détectées")
    
    return coco_results


if __name__ == "__main__":
    print("=" * 60)
    print("🌳 SAM Automatic Mask Generation - Out-of-the-Box")
    print("   Papier: Tree Crown Instance Segmentation")
    print("=" * 60)
    
    # Checkpoint check
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERREUR: Checkpoint SAM introuvable: {SAM_CHECKPOINT}")
        print("Téléchargez-le depuis: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        exit(1)
    
    # Run for the configured points_per_side values
    for pps in POINTS_PER_SIDE_OPTIONS:
        run_inference(points_per_side=pps)
    
    print("\n✅ Tous les tests terminés!")
