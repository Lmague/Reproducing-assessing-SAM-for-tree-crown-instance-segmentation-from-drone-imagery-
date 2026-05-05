"""
SAM2 Automatic Mask Generation - Out-of-the-Box Inference

Même approche que SAM_OTB.py mais avec SAM2 (sam2.1_hiera_large).
SAM2 améliore la qualité des masques et le score IoU prédit vs SAM v1.
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms
import pycocotools.mask as mask_util

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR, SAM2_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POINTS_PER_SIDE_OPTIONS = [10, 5, 3]
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD   = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_nms_to_masks(masks, score_threshold=0.5, iou_threshold=0.5):
    if not masks:
        return []
    masks = [m for m in masks if m["predicted_iou"] >= score_threshold]
    if not masks:
        return []
    boxes  = [[m["bbox"][0], m["bbox"][1], m["bbox"][0]+m["bbox"][2], m["bbox"][1]+m["bbox"][3]] for m in masks]
    scores = [m["predicted_iou"] for m in masks]
    keep   = nms(torch.tensor(boxes, dtype=torch.float32),
                 torch.tensor(scores, dtype=torch.float32),
                 iou_threshold).numpy()
    return [masks[i] for i in keep]


def run_inference(points_per_side: int):
    output_file = os.path.join(BASE_DIR, f"sam2_otb_pps{points_per_side}_predictions.json")
    print(f"\n{'='*60}\nSAM2 Automatic - Points per Side: {points_per_side}\n{'='*60}")

    print(f"Chargement SAM2 sur {DEVICE}...")
    # Essaie d'abord la config interne du package SAM2, puis le fichier local
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", SAM2_CHECKPOINT, device=DEVICE)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        points_per_batch=64,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        crop_n_layers=0,
        min_mask_region_area=0,
    )

    with open(TEST_JSON) as f:
        images_info = json.load(f)["images"]
    print(f"Images à traiter : {len(images_info)}")

    coco_results = []
    with torch.inference_mode():
        for img_info in tqdm(images_info):
            image_id = img_info["id"]
            img_path = os.path.join(TEST_IMG_DIR, os.path.basename(img_info["file_name"]))
            if not os.path.exists(img_path):
                alt = os.path.join(DATA_ROOT, img_info["file_name"])
                img_path = alt if os.path.exists(alt) else None
            if not img_path:
                continue

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            masks = mask_generator.generate(image_rgb)
            masks = apply_nms_to_masks(masks, SCORE_THRESHOLD, IOU_THRESHOLD)

            for m in masks:
                rle = mask_util.encode(np.asfortranarray(m["segmentation"].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                coco_results.append({
                    "image_id":    image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score":  float(m["predicted_iou"]),
                    "bbox":   [float(b) for b in m["bbox"]],
                    "bbox_mode": 1,
                })

    with open(output_file, "w") as f:
        json.dump(coco_results, f)
    print(f"Terminé. {len(coco_results)} instances → {output_file}")
    return coco_results


if __name__ == "__main__":
    print("=" * 60)
    print("SAM2 Automatic Mask Generation - OTB")
    print("=" * 60)
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"ERREUR: checkpoint SAM2 introuvable: {SAM2_CHECKPOINT}")
        exit(1)
    for pps in POINTS_PER_SIDE_OPTIONS:
        run_inference(points_per_side=pps)
    print("\nTous les tests SAM2 terminés!")
