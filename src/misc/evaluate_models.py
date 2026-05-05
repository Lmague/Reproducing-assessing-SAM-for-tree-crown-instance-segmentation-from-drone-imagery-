#!/usr/bin/env python3
"""
Evaluate multiple instance-segmentation models against a COCO GT file.

Metrics:
- mAP (COCO AP@[.5:.95], multi-class)
- wmAP (weighted by GT instance counts)
- per-class AP
- single-class AP (all categories collapsed to "tree")
- mIoU (best-match per GT instance; ignores false positives)
- pixel metrics on union masks (MSE, RMSE, IoU, precision, recall, F1, accuracy)

Notes:
- This script expects COCO-format predictions with segmentation in RLE or polygons.
- If your GT split is ambiguous, pass --gt explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


# --------------------------
# Utilities
# --------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_coco_from_dict(dataset_dict: dict) -> COCO:
    coco = COCO()
    coco.dataset = dict(dataset_dict)
    coco.dataset.setdefault("info", {})
    coco.dataset.setdefault("licenses", [])
    coco.createIndex()
    return coco


def to_rle(segmentation, height: int, width: int) -> dict:
    """Convert a segmentation (polygon or RLE) to compressed RLE."""
    if isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rle)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rle)
        else:
            rle = dict(segmentation)
    else:
        raise ValueError("Unsupported segmentation format")
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def guess_model_name(path: Path) -> str:
    p = str(path)
    name = path.stem
    # Heuristics for nicer names
    if "mask_rcnn_sam_predictions" in p:
        return "Mask R-CNN + SAM"
    if "faster_rcnn_sam_predictions" in p:
        return "Faster R-CNN + SAM"
    if "sam2_otb_pps" in p:
        suffix = path.stem.replace("sam2_otb_", "").replace("_predictions", "")
        return f"SAM2 OTB {suffix}"
    if "sam3_otb_descriptive" in p:
        return "SAM3 OTB (descriptive prompts)"
    if "sam3_otb_generic" in p:
        return "SAM3 OTB (generic prompts)"
    if "sam3_otb" in p:
        return "SAM3 OTB"
    if "sam_otb_pps" in p:
        suffix = path.stem.replace("sam_otb_", "").replace("_predictions", "")
        return f"SAM OTB {suffix}"
    if "sam_dsm_prompts" in p:
        return "SAM + DSM prompts"
    if "Output_mask_rcnn_RGB_DSM" in p or "output_mask_rcnn_RGB_DSM" in p:
        return "Mask R-CNN RGB+DSM"
    if "output_mask_rcnn_RGB" in p or "Output_mask_rcnn_RGB" in p:
        return "Mask R-CNN RGB"
    if "output_faster_rcnn" in p or "Output_faster_rcnn" in p:
        return "Faster R-CNN"
    if "output_dinov3_vitdet_rgb" in p or "Output_DINOv3_vitdet_rgb" in p:
        return "DINOv3 ViT-Det RGB"
    if "DINOv3_MaskRCNN_Trees" in p:
        return "DINOv3 Mask R-CNN (trees)"
    # Fallback for standalone prediction files from src/models/
    if "mask_rcnn_rgb_predictions" in p:
        return "Mask R-CNN RGB"
    if "faster_rcnn_predictions" in p:
        return "Faster R-CNN"
    return name


# --------------------------
# COCO mAP / wmAP
# --------------------------

def extract_ap_per_class(coco_eval: COCOeval, cat_ids: List[int]) -> Dict[int, float]:
    precisions = coco_eval.eval["precision"]  # (IoU, recall, category, area, max_det)
    ap_per_class = {}
    for idx, cat_id in enumerate(cat_ids):
        precision = precisions[:, :, idx, 0, -1]
        valid = precision[precision > -1]
        ap = float(np.mean(valid)) if valid.size else 0.0
        ap_per_class[cat_id] = ap
    return ap_per_class


def evaluate_map(gt_coco: COCO, preds: List[dict], cat_id_to_name: Dict[int, str]) -> dict:
    if not preds:
        return {
            "mAP": float("nan"),
            "wmAP": float("nan"),
            "per_class_ap": {cat_id_to_name[cid]: float("nan") for cid in cat_id_to_name},
            "coco_eval": None,
        }

    coco_dt = gt_coco.loadRes(preds)
    coco_eval = COCOeval(gt_coco, coco_dt, iouType="segm")
    coco_eval.params.catIds = sorted(cat_id_to_name.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap_per_class_raw = extract_ap_per_class(coco_eval, coco_eval.params.catIds)
    counts = {cid: len(gt_coco.getAnnIds(catIds=[cid])) for cid in cat_id_to_name}
    total = sum(counts.values()) or 1.0

    mAP = float(np.mean(list(ap_per_class_raw.values()))) if ap_per_class_raw else float("nan")
    wmAP = float(
        sum(ap_per_class_raw[cid] * counts[cid] for cid in ap_per_class_raw) / total
    )

    per_class_named = {cat_id_to_name[cid]: ap_per_class_raw[cid] * 100 for cid in ap_per_class_raw}

    return {
        "mAP": mAP * 100,
        "wmAP": wmAP * 100,
        "per_class_ap": per_class_named,
        "coco_eval": coco_eval,
    }


def collapse_to_single_class(gt_dict: dict, preds: List[dict]) -> Tuple[dict, List[dict]]:
    gt_single = deepcopy(gt_dict)
    gt_single["categories"] = [{"id": 1, "name": "tree", "supercategory": "tree"}]
    for ann in gt_single["annotations"]:
        ann["category_id"] = 1
    preds_single = []
    for p in preds:
        q = dict(p)
        q["category_id"] = 1
        preds_single.append(q)
    return gt_single, preds_single


# --------------------------
# mIoU best match
# --------------------------

def compute_best_match_miou(
    gt_dict: dict,
    preds: List[dict],
    chunk_size: int = 300,
) -> float:
    img_sizes = {img["id"]: (img["height"], img["width"]) for img in gt_dict["images"]}

    gts_by_image = defaultdict(list)
    for ann in gt_dict["annotations"]:
        gts_by_image[ann["image_id"]].append(ann["segmentation"])

    preds_by_image = defaultdict(list)
    for pred in preds:
        preds_by_image[pred["image_id"]].append(pred["segmentation"])

    best_ious = []
    for image_id, gt_segs in gts_by_image.items():
        height, width = img_sizes[image_id]
        gt_rles = [to_rle(seg, height, width) for seg in gt_segs]
        pred_segs = preds_by_image.get(image_id, [])
        if not gt_rles:
            continue
        if pred_segs:
            pred_rles = [to_rle(seg, height, width) for seg in pred_segs]
            best = np.zeros(len(gt_rles), dtype=np.float32)
            for chunk in chunked(pred_rles, chunk_size):
                ious = mask_utils.iou(chunk, gt_rles, [0] * len(gt_rles))
                if ious.size:
                    best = np.maximum(best, ious.max(axis=0))
        else:
            best = np.zeros(len(gt_rles), dtype=np.float32)
        best_ious.extend(best.tolist())
    return float(np.mean(best_ious)) * 100 if best_ious else float("nan")


# --------------------------
# Pixel-level metrics on union masks
# --------------------------

def union_rle(seg_list: List, height: int, width: int) -> Optional[dict]:
    if not seg_list:
        return None
    rles = [to_rle(seg, height, width) for seg in seg_list]
    if len(rles) == 1:
        return rles[0]
    return mask_utils.merge(rles)


def compute_pixel_metrics(
    gt_dict: dict,
    preds: List[dict],
    ignore_empty: bool = False,
) -> dict:
    img_sizes = {img["id"]: (img["height"], img["width"]) for img in gt_dict["images"]}

    gts_by_image = defaultdict(list)
    for ann in gt_dict["annotations"]:
        gts_by_image[ann["image_id"]].append(ann["segmentation"])

    preds_by_image = defaultdict(list)
    for pred in preds:
        preds_by_image[pred["image_id"]].append(pred["segmentation"])

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_pixels = 0.0

    for image_id, (height, width) in img_sizes.items():
        gt_segs = gts_by_image.get(image_id, [])
        pred_segs = preds_by_image.get(image_id, [])

        gt_union = union_rle(gt_segs, height, width)
        pred_union = union_rle(pred_segs, height, width)

        area_gt = float(mask_utils.area(gt_union)) if gt_union is not None else 0.0
        area_pred = float(mask_utils.area(pred_union)) if pred_union is not None else 0.0

        if ignore_empty and area_gt == 0.0 and area_pred == 0.0:
            continue

        if area_gt == 0.0 and area_pred == 0.0:
            iou = 1.0
            intersection = 0.0
        elif area_gt == 0.0 or area_pred == 0.0:
            iou = 0.0
            intersection = 0.0
        else:
            iou = float(mask_utils.iou([pred_union], [gt_union], [0])[0][0])
            intersection = (iou * (area_pred + area_gt)) / (1.0 + iou) if iou > 0 else 0.0

        tp = intersection
        fp = area_pred - intersection
        fn = area_gt - intersection

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pixels += float(height * width)

    denom_prec = total_tp + total_fp
    denom_rec = total_tp + total_fn
    denom_iou = total_tp + total_fp + total_fn

    precision = total_tp / denom_prec if denom_prec > 0 else 0.0
    recall = total_tp / denom_rec if denom_rec > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    pixel_iou = total_tp / denom_iou if denom_iou > 0 else 0.0
    mse = (total_fp + total_fn) / total_pixels if total_pixels > 0 else float("nan")
    rmse = math.sqrt(mse) if mse == mse else float("nan")
    accuracy = 1.0 - mse if mse == mse else float("nan")

    return {
        "pixel_precision": precision * 100,
        "pixel_recall": recall * 100,
        "pixel_f1": f1 * 100,
        "pixel_iou": pixel_iou * 100,
        "pixel_mse": mse,
        "pixel_rmse": rmse,
        "pixel_acc": accuracy * 100 if accuracy == accuracy else float("nan"),
    }


# --------------------------
# IO helpers
# --------------------------

def find_prediction_files(root: Path) -> List[Path]:
    preds = []
    for p in root.rglob("*.json"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        name = p.name
        if name == "coco_instances_results.json" or "predictions" in name:
            preds.append(p)
    return sorted(set(preds))


def resolve_pred_paths(pred_args: Optional[List[str]]) -> List[Path]:
    if not pred_args:
        return find_prediction_files(Path("."))

    out: List[Path] = []
    for entry in pred_args:
        p = Path(entry)
        if p.is_dir():
            out.extend(find_prediction_files(p))
        elif p.is_file():
            out.append(p)
    return sorted(set(out))


def detect_default_gt() -> Optional[Path]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.paths import DATA_ROOT
        p = Path(DATA_ROOT) / "coco" / "val.json"
        if p.exists():
            return p
    except Exception:
        pass
    candidates = [
        Path("./data"),
        Path("../data"),
        Path("../../data"),
        Path("./Output"),
        Path("../Output"),
        Path("../../Output"),
    ]
    for root in candidates:
        p = root / "coco" / "val.json"
        if p.exists():
            return p
    return None


# --------------------------
# Analysis report
# --------------------------

def best_by(results: List[dict], key: str, higher: bool = True) -> Optional[dict]:
    items = [r for r in results if r.get(key) is not None and not math.isnan(r[key])]
    if not items:
        return None
    return max(items, key=lambda r: r[key]) if higher else min(items, key=lambda r: r[key])


def render_analysis(results: List[dict]) -> str:
    if not results:
        return "No results to analyze."

    lines = []
    lines.append("# Metrics analysis")

    best_map = best_by(results, "mAP_pct", higher=True)
    best_wmap = best_by(results, "wmAP_pct", higher=True)
    best_tree = best_by(results, "mAP_tree_pct", higher=True)
    best_miou = best_by(results, "mIoU_bestmatch_pct", higher=True)
    best_rmse = best_by(results, "pixel_rmse", higher=False)

    if best_map:
        lines.append(f"- Best mAP (multi-class): {best_map['model_name']} ({best_map['mAP_pct']:.2f}%)")
    if best_wmap:
        lines.append(f"- Best wmAP: {best_wmap['model_name']} ({best_wmap['wmAP_pct']:.2f}%)")
    if best_tree:
        lines.append(f"- Best single-class AP (tree): {best_tree['model_name']} ({best_tree['mAP_tree_pct']:.2f}%)")
    if best_miou:
        lines.append(f"- Best mIoU (best match): {best_miou['model_name']} ({best_miou['mIoU_bestmatch_pct']:.2f}%)")
    if best_rmse:
        lines.append(f"- Lowest pixel RMSE: {best_rmse['model_name']} ({best_rmse['pixel_rmse']:.4f})")

    # Split warning
    splits = {r.get("gt_path") for r in results}
    if len(splits) > 1:
        lines.append("- Warning: results use different GT files; avoid comparing across splits.")

    # Simple diagnostics per model
    lines.append("\n## Notes")
    for r in results:
        note_parts = []
        if r.get("wmAP_pct") is not None and r.get("mAP_pct") is not None:
            if not math.isnan(r["wmAP_pct"]) and not math.isnan(r["mAP_pct"]):
                if r["wmAP_pct"] - r["mAP_pct"] > 5.0:
                    note_parts.append("wmAP >> mAP (likely class imbalance; better on frequent classes)")
        if r.get("mIoU_bestmatch_pct") is not None and r.get("mAP_tree_pct") is not None:
            if not math.isnan(r["mIoU_bestmatch_pct"]) and not math.isnan(r["mAP_tree_pct"]):
                if r["mIoU_bestmatch_pct"] > r["mAP_tree_pct"] + 20.0:
                    note_parts.append("mIoU >> AP (good overlap but weak instance confidence/separation)")
        if r.get("pixel_mse") is not None and r.get("mAP_tree_pct") is not None:
            if not math.isnan(r["pixel_mse"]) and not math.isnan(r["mAP_tree_pct"]):
                if r["pixel_mse"] < 0.05 and r["mAP_tree_pct"] < 5.0:
                    note_parts.append("low pixel error but low AP (possible over-merged instances)")
        if note_parts:
            lines.append(f"- {r['model_name']}: " + "; ".join(note_parts))

    return "\n".join(lines)


# --------------------------
# Main
# --------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, default=None, help="Path to COCO GT json")
    ap.add_argument("--gt-root", type=str, default=None, help="Dataset root containing coco/")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default=None)
    ap.add_argument("--pred", action="append", help="Prediction json file or directory (repeatable)")
    ap.add_argument("--score-thr", type=float, default=None, help="Filter predictions by score")
    ap.add_argument("--skip-pixel", action="store_true", help="Skip pixel metrics (MSE/RMSE/etc)")
    ap.add_argument("--ignore-empty", action="store_true", help="Ignore images with no GT and no preds for pixel metrics")
    ap.add_argument("--chunk", type=int, default=300, help="Chunk size for mIoU computation")
    ap.add_argument("--out-json", type=str, default="metrics_summary.json")
    ap.add_argument("--out-csv", type=str, default="metrics_summary.csv")
    ap.add_argument("--analysis-out", type=str, default="metrics_analysis.md")

    args = ap.parse_args()

    # Resolve GT
    if args.gt:
        gt_path = Path(args.gt)
    elif args.gt_root and args.split:
        gt_path = Path(args.gt_root) / "coco" / f"{args.split}.json"
    else:
        gt_path = detect_default_gt()
        if gt_path is None:
            print("ERROR: GT not found. Please pass --gt or --gt-root + --split.")
            return 1
        print(f"[WARN] Using default GT: {gt_path}")

    if not gt_path.exists():
        print(f"ERROR: GT file not found: {gt_path}")
        return 1

    gt_dict = load_json(gt_path)
    gt_dict.setdefault("info", {})
    gt_dict.setdefault("licenses", [])

    cat_id_to_name = {c["id"]: c["name"] for c in gt_dict.get("categories", [])}
    valid_cat_ids = set(cat_id_to_name.keys())

    pred_paths = resolve_pred_paths(args.pred)
    if not pred_paths:
        print("ERROR: No prediction files found.")
        return 1

    results = []

    for pred_path in pred_paths:
        preds = load_json(pred_path)

        # Optional score filter
        if args.score_thr is not None:
            preds = [p for p in preds if p.get("score", 1.0) >= args.score_thr]

        # Filter to known categories
        preds = [p for p in preds if p.get("category_id") in valid_cat_ids]

        # Filter to GT image ids (avoid pycocotools assertion)
        valid_img_ids = {img["id"] for img in gt_dict.get("images", [])}
        preds = [p for p in preds if p.get("image_id") in valid_img_ids]

        num_pred = len(preds)
        num_pred_images = len({p.get("image_id") for p in preds})
        num_gt_images = len(gt_dict.get("images", []))
        num_gt_instances = len(gt_dict.get("annotations", []))

        # COCO eval
        gt_coco = build_coco_from_dict(deepcopy(gt_dict))
        multi = evaluate_map(gt_coco, preds, cat_id_to_name)

        # single class
        single_gt, single_preds = collapse_to_single_class(gt_dict, preds)
        single_coco = build_coco_from_dict(single_gt)
        single = evaluate_map(single_coco, single_preds, {1: "tree"})
        miou = compute_best_match_miou(single_gt, single_preds, chunk_size=args.chunk)

        # pixel metrics
        pixel_metrics = {}
        if not args.skip_pixel:
            pixel_metrics = compute_pixel_metrics(single_gt, single_preds, ignore_empty=args.ignore_empty)

        # Standard COCO stats
        stats = None
        if multi["coco_eval"] is not None and hasattr(multi["coco_eval"], "stats"):
            stats = multi["coco_eval"].stats

        result = {
            "model_name": guess_model_name(pred_path),
            "pred_path": str(pred_path),
            "gt_path": str(gt_path),
            "num_pred": num_pred,
            "num_pred_images": num_pred_images,
            "num_gt_images": num_gt_images,
            "num_gt_instances": num_gt_instances,
            "mAP_pct": multi["mAP"],
            "wmAP_pct": multi["wmAP"],
            "mAP_tree_pct": single["mAP"],
            "mIoU_bestmatch_pct": miou,
            "per_class_ap_pct": multi["per_class_ap"],
            "pixel_precision_pct": pixel_metrics.get("pixel_precision"),
            "pixel_recall_pct": pixel_metrics.get("pixel_recall"),
            "pixel_f1_pct": pixel_metrics.get("pixel_f1"),
            "pixel_iou_pct": pixel_metrics.get("pixel_iou"),
            "pixel_mse": pixel_metrics.get("pixel_mse"),
            "pixel_rmse": pixel_metrics.get("pixel_rmse"),
            "pixel_acc_pct": pixel_metrics.get("pixel_acc"),
        }

        if stats is not None and len(stats) >= 9:
            result.update(
                {
                    "coco_ap": stats[0] * 100,
                    "coco_ap50": stats[1] * 100,
                    "coco_ap75": stats[2] * 100,
                    "coco_ap_small": stats[3] * 100,
                    "coco_ap_medium": stats[4] * 100,
                    "coco_ap_large": stats[5] * 100,
                    "coco_ar_1": stats[6] * 100,
                    "coco_ar_10": stats[7] * 100,
                    "coco_ar_100": stats[8] * 100,
                }
            )

        results.append(result)

    # Save JSON
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save CSV (flat)
    out_csv = Path(args.out_csv)
    headers = [
        "model_name",
        "pred_path",
        "gt_path",
        "num_pred",
        "num_pred_images",
        "num_gt_images",
        "num_gt_instances",
        "mAP_pct",
        "wmAP_pct",
        "mAP_tree_pct",
        "mIoU_bestmatch_pct",
        "pixel_mse",
        "pixel_rmse",
        "pixel_iou_pct",
        "pixel_f1_pct",
        "pixel_precision_pct",
        "pixel_recall_pct",
        "pixel_acc_pct",
        "coco_ap",
        "coco_ap50",
        "coco_ap75",
        "coco_ap_small",
        "coco_ap_medium",
        "coco_ap_large",
        "coco_ar_1",
        "coco_ar_10",
        "coco_ar_100",
    ]
    lines = [",".join(headers)]
    for r in results:
        row = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                row.append(f"{v:.6g}")
            else:
                row.append(str(v) if v is not None else "")
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save analysis
    analysis_text = render_analysis(results)
    Path(args.analysis_out).write_text(analysis_text + "\n", encoding="utf-8")

    # Console summary
    print("==== Metrics summary ====")
    for r in results:
        print(
            f"{r['model_name']}: mAP={r['mAP_pct']:.2f} | wmAP={r['wmAP_pct']:.2f} | "
            f"mAP_tree={r['mAP_tree_pct']:.2f} | mIoU={r['mIoU_bestmatch_pct']:.2f}"
        )
    print(f"\nWrote: {out_json}, {out_csv}, {args.analysis_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
