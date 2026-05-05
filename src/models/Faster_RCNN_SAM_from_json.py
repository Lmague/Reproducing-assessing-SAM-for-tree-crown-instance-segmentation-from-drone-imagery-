"""
Faster R-CNN + SAM — Pipeline Hybride (depuis JSON existant)

Utilise les boîtes Faster R-CNN déjà sauvegardées comme prompts pour SAM,
sans réentraîner Faster R-CNN.

Score final = moyenne(score_faster_rcnn, iou_sam)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRCNN_PREDS   = os.path.join(BASE_DIR, "faster_rcnn_predictions.json")
OUTPUT_FILE   = os.path.join(BASE_DIR, "faster_rcnn_sam_predictions.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    print("=" * 60)
    print("Faster R-CNN + SAM (from JSON)")
    print("=" * 60)

    with open(FRCNN_PREDS) as f:
        frcnn_preds = json.load(f)

    by_image = {}
    for p in frcnn_preds:
        by_image.setdefault(p["image_id"], []).append(p)

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
            x, y, w, h = p["bbox"]
            box = np.array([[x, y, x + w, y + h]])

            masks_sam, iou_preds, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box, multimask_output=False,
            )
            mask_sam  = masks_sam[0]
            iou_score = float(iou_preds[0])
            final_score = (p["score"] + iou_score) / 2.0

            out_rle = mask_util.encode(np.asfortranarray(mask_sam.astype(np.uint8)))
            out_rle["counts"] = out_rle["counts"].decode("utf-8")

            rows = np.any(mask_sam, axis=1)
            cols = np.any(mask_sam, axis=0)
            if not rows.any():
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            results.append({
                "image_id":    img_id,
                "category_id": p["category_id"],
                "segmentation": out_rle,
                "score":        float(final_score),
                "bbox":         [float(cmin), float(rmin),
                                 float(cmax - cmin + 1), float(rmax - rmin + 1)],
                "bbox_mode":    1,
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"Sauvegardé: {OUTPUT_FILE} ({len(results)} instances)")


if __name__ == "__main__":
    run()
