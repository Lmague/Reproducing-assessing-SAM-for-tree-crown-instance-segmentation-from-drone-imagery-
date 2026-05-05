"""
Mask R-CNN + SAM — Pipeline Hybride (depuis JSON existant)

Utilise les prédictions Mask R-CNN RGB déjà sauvegardées (masques RLE + boîtes)
comme prompts pour SAM, sans réentraîner Mask R-CNN.

Score final = moyenne(score_mask_rcnn, iou_sam)
"""

import os, cv2, json, torch
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util
from segment_anything import sam_model_registry, SamPredictor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR, SAM_CHECKPOINT,
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MRCNN_PREDS   = os.path.join(BASE_DIR, "mask_rcnn_rgb_predictions.json")
OUTPUT_FILE   = os.path.join(BASE_DIR, "mask_rcnn_sam_predictions.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mask_to_sam_prompt(mask_binary: np.ndarray) -> np.ndarray:
    import cv2
    m = cv2.resize(mask_binary.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
    return (m * 2.0 - 1.0) * 10.0  # logits-like scaling


def run():
    print("=" * 60)
    print("Mask R-CNN + SAM (from JSON)")
    print("=" * 60)

    # Charger prédictions Mask R-CNN groupées par image
    with open(MRCNN_PREDS) as f:
        mrcnn_preds = json.load(f)

    by_image = {}
    for p in mrcnn_preds:
        by_image.setdefault(p["image_id"], []).append(p)

    # Charger SAM
    print(f"Chargement SAM ViT-H sur {DEVICE}...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    with open(TEST_JSON) as f:
        images_info = json.load(f)["images"]

    results = []
    for img_info in tqdm(images_info):
        img_id = img_info["id"]
        preds  = by_image.get(img_id, [])
        if not preds:
            continue

        img_path = os.path.join(TEST_IMG_DIR, os.path.basename(img_info["file_name"]))
        if not os.path.exists(img_path):
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        for p in preds:
            # Décoder masque RLE → binaire
            rle = p["segmentation"]
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            mask_bin = mask_util.decode(rle).astype(bool)   # (H, W)

            # Box [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = p["bbox"]
            box = np.array([[x, y, x + w, y + h]])

            mask_prompt = mask_to_sam_prompt(mask_bin.astype(np.uint8))[None]  # (1,256,256)

            masks_sam, iou_preds, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box, mask_input=mask_prompt,
                multimask_output=False,
            )
            mask_sam  = masks_sam[0]
            iou_score = float(iou_preds[0])
            final_score = (p["score"] + iou_score) / 2.0

            out_rle = mask_util.encode(np.asfortranarray(mask_sam.astype(np.uint8)))
            out_rle["counts"] = out_rle["counts"].decode("utf-8")

            results.append({
                "image_id":    img_id,
                "category_id": p["category_id"],
                "segmentation": out_rle,
                "score":        float(final_score),
                "bbox":         [float(x), float(y), float(w), float(h)],
                "bbox_mode":    1,
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"Sauvegardé: {OUTPUT_FILE} ({len(results)} instances)")


if __name__ == "__main__":
    run()
