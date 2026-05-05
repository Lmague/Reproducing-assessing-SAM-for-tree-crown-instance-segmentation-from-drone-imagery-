#!/usr/bin/env python3
"""
Sanity check for metric functions in evaluate_models.py.

Tests inputs with known expected outputs to validate each metric.
Run with:  python src/misc/sanity_check_metrics.py
"""
import sys
import math
import numpy as np

# --------------------------------------------------------------------------
# Helpers to build minimal COCO dicts and RLE masks
# --------------------------------------------------------------------------

def _make_mask_rle(h, w, row_start, row_end, col_start, col_end):
    """Create a binary mask RLE for a rectangle region."""
    from pycocotools import mask as mask_utils
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[row_start:row_end, col_start:col_end] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle, mask


def _make_gt_dict(annotations, images=None):
    if images is None:
        images = [{"id": 1, "height": 100, "width": 100}]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tree", "supercategory": "tree"}],
        "info": {},
        "licenses": [],
    }


# --------------------------------------------------------------------------
# Import functions to test
# --------------------------------------------------------------------------
sys.path.insert(0, ".")
from src.misc.evaluate_models import (
    compute_best_match_miou,
    compute_pixel_metrics,
    evaluate_map,
    build_coco_from_dict,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []


def check(name, got, expected, tol=1e-3):
    ok = abs(got - expected) < tol if not math.isnan(expected) else math.isnan(got)
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}: got={got:.4f}, expected={expected:.4f}")
    if not ok:
        errors.append(name)


# ==========================================================================
# TEST 1 — mIoU: perfect match → IoU = 1.0
# ==========================================================================
print("\n== TEST 1: mIoU perfect match (IoU should be 1.0) ==")

rle1, _ = _make_mask_rle(100, 100, 10, 40, 10, 40)
gt_dict = _make_gt_dict([
    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": rle1, "area": 900, "bbox": [10,10,30,30], "iscrowd": 0}
])
preds = [
    {"image_id": 1, "category_id": 1, "segmentation": rle1, "score": 1.0, "bbox": [10,10,30,30]}
]
miou = compute_best_match_miou(gt_dict, preds)
check("mIoU perfect", miou, 100.0)

# ==========================================================================
# TEST 2 — mIoU: no predictions → IoU = 0.0
# ==========================================================================
print("\n== TEST 2: mIoU no predictions (IoU should be 0.0) ==")

miou_empty = compute_best_match_miou(gt_dict, [])
check("mIoU no preds", miou_empty, 0.0)

# ==========================================================================
# TEST 3 — mIoU: half-overlap → IoU ≈ 0.333
# ==========================================================================
print("\n== TEST 3: mIoU 50% overlap rectangle (IoU = 1/3) ==")

# GT: rows 10-40, cols 10-40 (30×30=900px)
# Pred: rows 10-40, cols 25-55 (30×30=900px)
# Intersection: rows 10-40, cols 25-40 (30×15=450px)
# Union: 900+900-450 = 1350
# IoU = 450/1350 = 1/3
rle_pred_half, _ = _make_mask_rle(100, 100, 10, 40, 25, 55)
preds_half = [
    {"image_id": 1, "category_id": 1, "segmentation": rle_pred_half, "score": 1.0, "bbox": [25,10,30,30]}
]
miou_half = compute_best_match_miou(gt_dict, preds_half)
check("mIoU half overlap", miou_half, (1/3)*100, tol=0.5)

# ==========================================================================
# TEST 4 — Pixel metrics: perfect prediction
# ==========================================================================
print("\n== TEST 4: Pixel metrics perfect prediction ==")

pm_perfect = compute_pixel_metrics(gt_dict, preds)
check("pixel_precision perfect", pm_perfect["pixel_precision"], 100.0)
check("pixel_recall perfect",    pm_perfect["pixel_recall"],    100.0)
check("pixel_f1 perfect",        pm_perfect["pixel_f1"],        100.0)
check("pixel_iou perfect",       pm_perfect["pixel_iou"],       100.0)
check("pixel_mse perfect",       pm_perfect["pixel_mse"],       0.0)

# ==========================================================================
# TEST 5 — Pixel metrics: no predictions → precision=0, recall=0, IoU=0
# ==========================================================================
print("\n== TEST 5: Pixel metrics no predictions ==")

pm_empty = compute_pixel_metrics(gt_dict, [])
check("pixel_precision empty", pm_empty["pixel_precision"], 0.0)
check("pixel_recall empty",    pm_empty["pixel_recall"],    0.0)
check("pixel_f1 empty",        pm_empty["pixel_f1"],        0.0)
check("pixel_iou empty",       pm_empty["pixel_iou"],       0.0)

# ==========================================================================
# TEST 6 — COCO mAP: perfect single prediction → AP should be 1.0 (100%)
# Note: needs summarize() to be called internally
# ==========================================================================
print("\n== TEST 6: COCO mAP perfect prediction (should be ~100%) ==")

from copy import deepcopy

gt_coco = build_coco_from_dict(deepcopy(gt_dict))
map_result = evaluate_map(gt_coco, preds, {1: "tree"})
check("mAP perfect (single class)", map_result["mAP"], 100.0, tol=1.0)

# ==========================================================================
# TEST 7 — COCO mAP: no predictions → AP = 0
# ==========================================================================
print("\n== TEST 7: COCO mAP no predictions (should be 0%) ==")

gt_coco2 = build_coco_from_dict(deepcopy(gt_dict))
map_empty = evaluate_map(gt_coco2, [], {1: "tree"})
# Returns nan when no preds (by design in evaluate_map)
print(f"  [INFO] mAP with no preds = {map_empty['mAP']!r} (expected nan)")

# ==========================================================================
# TEST 8 — summarize() was called: stats must be non-empty
# ==========================================================================
print("\n== TEST 8: coco_eval.stats populated (summarize() was called) ==")

from pycocotools.cocoeval import COCOeval
gt_coco3 = build_coco_from_dict(deepcopy(gt_dict))
dt_coco3 = gt_coco3.loadRes(preds)
coco_eval = COCOeval(gt_coco3, dt_coco3, iouType="segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
stats_ok = len(coco_eval.stats) >= 9
status = PASS if stats_ok else FAIL
print(f"  [{status}] stats length={len(coco_eval.stats)}, expected>=9")
if not stats_ok:
    errors.append("summarize() check")
ap50_val = coco_eval.stats[1] * 100
check("AP50 from stats (perfect pred)", ap50_val, 100.0, tol=1.0)

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "="*50)
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("All tests PASSED.")
    sys.exit(0)
