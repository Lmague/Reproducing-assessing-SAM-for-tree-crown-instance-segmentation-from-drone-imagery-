"""
Minimal inference demo for tree crown instance segmentation.

This script is the recommended entry point for new users. It runs inference
on a single RGB (+ optional DSM) tile using the trained DINOv3 + Mask R-CNN model
and saves a visualisation of predicted masks.

Usage:
    # RGB only (will use model trained on RGB)
    python src/inference/demo.py \
        --rgb path/to/image.tif \
        --output prediction.png

    # RGB + DSM (4-channel input)
    python src/inference/demo.py \
        --rgb path/to/rgb.tif \
        --dsm path/to/dsm.tif \
        --output prediction.png

    # With ground-truth COCO annotations to compute AP50
    python src/inference/demo.py \
        --rgb path/to/rgb.tif \
        --dsm path/to/dsm.tif \
        --gt path/to/gt.json \
        --output prediction.png

Author: Lmague
Date: 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

# Try to import optional demo dependencies
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _COCO_AVAILABLE = True
except ImportError:
    _COCO_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--rgb", required=True,
        help="Path to input RGB image (GeoTIFF or PNG)"
    )
    ap.add_argument(
        "--dsm",
        default=None,
        help="Path to DSM image aligned with RGB (optional; omit for RGB-only models)"
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to model checkpoint .pth file. "
            "Defaults to the best DINOv3 checkpoint in the default output directory."
        )
    )
    ap.add_argument(
        "--output", default="demo_output.png",
        help="Path to save the visualisation (default: demo_output.png)"
    )
    ap.add_argument(
        "--gt",
        default=None,
        help="Path to COCO-format ground-truth JSON for AP50 computation"
    )
    ap.add_argument(
        "--score-thr", type=float, default=0.5,
        help="Confidence threshold for displaying predictions (default: 0.5)"
    )
    return ap.parse_args()


def load_rgb(rgb_path: str) -> np.ndarray:
    """Load an RGB image from disk and return a uint8 BGR numpy array."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load images: pip install Pillow")

    pil_img = Image.open(rgb_path)
    # Convert to RGB then BGR for OpenCV compatibility
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def load_dsm(dsm_path: str) -> np.ndarray:
    """Load a DSM image, normalise it to [0,1], and return a HxW float32 array."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load images: pip install Pillow")

    pil_dsm = Image.open(dsm_path)
    dsm = np.array(pil_dsm).astype(np.float32)

    # Handle multi-band DSM (take first channel)
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]

    # Replace NaN / Inf with 0
    dsm = np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise to [0, 1] per-sample
    max_val = dsm.max()
    if max_val > 0:
        dsm = dsm / max_val

    return dsm


def overlay_masks(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    boxes: list,
    scores: list[float],
    score_thr: float = 0.5,
) -> np.ndarray:
    """
    Draw predicted masks, bounding boxes, and scores on the image.

    Args:
        image_bgr: HxWx3 uint8 BGR image.
        masks: List of HxW boolean or uint8 masks.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        scores: List of confidence scores.
        score_thr: Only visualise predictions above this threshold.

    Returns:
        uint8 BGR image with overlaid visualisations.
    """
    vis = image_bgr.copy()

    # Colour palette for different instance IDs
    np.random.seed(42)
    colours = {
        i: (int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)))
        for i in range(200)
    }

    for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if score < score_thr:
            continue

        colour = colours[idx % len(colours)]

        # Draw filled mask with transparency
        mask_u8 = (mask * 0.4 * 255).astype(np.uint8)
        mask_coloured = np.zeros_like(vis)
        mask_coloured[:, :] = colour
        mask_coloured = cv2.bitwise_and(mask_coloured, mask_coloured, mask=mask_u8)
        vis = cv2.addWeighted(vis, 1.0, mask_coloured, 1.0, 0)

        # Draw contour
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, colour, 2)

        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Draw score label
        label = f"{score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - lh - 4), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return vis


def compute_ap50(predictions: list, gt_path: str) -> float | None:
    """
    Compute AP50 against a COCO-format ground-truth JSON.

    Args:
        predictions: List of COCO-format prediction dicts.
        gt_path: Path to ground-truth COCO JSON.

    Returns:
        AP50 as a float, or None if computation fails.
    """
    if not _COCO_AVAILABLE:
        print("pycocotools not available; skipping AP50 computation.")
        return None

    try:
        with open(gt_path) as f:
            gt_dict = json.load(f)
    except Exception as e:
        print(f"Could not load ground-truth from {gt_path}: {e}")
        return None

    gt_dict.setdefault("info", {})
    gt_dict.setdefault("licenses", [])

    from copy import deepcopy
    gt_coco = COCO()
    gt_coco.dataset = dict(gt_dict)
    gt_coco.dataset.setdefault("info", {})
    gt_coco.dataset.setdefault("licenses", [])
    gt_coco.createIndex()

    # Filter predictions to known categories and valid image IDs
    valid_cat_ids = {c["id"] for c in gt_dict.get("categories", [])}
    valid_img_ids = {img["id"] for img in gt_dict.get("images", [])}
    filtered = [
        p for p in predictions
        if p.get("category_id") in valid_cat_ids
        and p.get("image_id") in valid_img_ids
    ]

    if not filtered:
        print("No valid predictions to evaluate.")
        return None

    dt_coco = gt_coco.loadRes(filtered)
    coco_eval = COCOeval(gt_coco, dt_coco, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # AP50 is stats[1] (index 1 in the stats array)
    ap50 = coco_eval.stats[1]
    return float(ap50) * 100.0


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.rgb):
        print(f"ERROR: RGB image not found: {args.rgb}")
        return 1

    rgb_dir = os.path.dirname(args.rgb)
    image_name = os.path.basename(args.rgb)

    # Determine default checkpoint if not provided
    if args.checkpoint is None:
        # Search for the best DINOv3 checkpoint in known output directories
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "training", "DINOv3_MaskRCNN_Trees"),
            os.path.join(os.path.dirname(__file__), "..", "..", "Résulats post training", "Output_DINOv3_MaskRCNN_Trees"),
        ]
        for d in search_dirs:
            best = os.path.join(d, "model_best.pth")
            final = os.path.join(d, "model_final.pth")
            if os.path.exists(best):
                args.checkpoint = best
                break
            elif os.path.exists(final):
                args.checkpoint = final
                break

    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print("WARNING: No checkpoint found. Running with randomly-initialised model.")
        print("         (This will produce nonsense predictions.)")
        print("         Provide --checkpoint to load a trained model.")
        args.checkpoint = None

    # Load images
    print(f"Loading RGB: {args.rgb}")
    image_bgr = load_rgb(args.rgb)
    print(f"  Image shape: {image_bgr.shape}")

    has_dsm = args.dsm is not None
    if has_dsm:
        if not os.path.exists(args.dsm):
            print(f"ERROR: DSM image not found: {args.dsm}")
            return 1
        print(f"Loading DSM: {args.dsm}")
        dsm = load_dsm(args.dsm)
        print(f"  DSM shape: {dsm.shape}")
    else:
        dsm = None
        print("No DSM provided; running RGB-only inference.")

    # -----------------------------------------------------------------
    # NOTE:
    #   The full Detectron2-based inference pipeline (loading the trained
    #   DINOv3 + Mask R-CNN model, applying the custom 4-channel mapper,
    #   and calling the model) is non-trivial and requires matching the
    #   exact model architecture and config used during training.
    #
    #   Below we provide a minimal working demo using SAM directly.
    #   To run the full trained DINOv3 pipeline, see:
    #     - src/training/train_dino.py --eval-only for evaluation on val/test
    #     - src/inference/test_dsm_prompter.py for DSMPrompter inference
    #
    #   Replace the block below with the appropriate Detectron2 predictor
    #   once you have a trained checkpoint and have registered the model.
    # -----------------------------------------------------------------
    print("\nRunning inference (using SAM automatic mask generation as placeholder)...")

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from utils.paths import SAM_CHECKPOINT
        sam_ckpt = SAM_CHECKPOINT

        if not os.path.exists(sam_ckpt):
            print(f"ERROR: SAM checkpoint not found at {sam_ckpt}")
            print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return 1

        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.0,
            stability_score_thresh=0.0,
            nms_thresh=0.5,
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            sam_masks = generator.generate(image_rgb)

        # Convert SAM masks to visualisation format
        masks, boxes, scores = [], [], []
        for m in sam_masks:
            if m["predicted_iou"] < args.score_thr:
                continue
            masks.append(m["segmentation"].astype(np.uint8) * 255)
            x, y, w, h = m["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(float(m["predicted_iou"]))

        print(f"  SAM generated {len(masks)} masks above threshold {args.score_thr}")

    except Exception as e:
        print(f"SAM inference failed: {e}")
        print("No visualisation will be produced.")
        masks, boxes, scores = [], [], []

    # Visualise and save
    if masks:
        vis = overlay_masks(image_bgr, [m.astype(bool) for m in masks], boxes, scores, args.score_thr)
        cv2.imwrite(args.output, vis)
        print(f"Visualisation saved: {args.output}")
    else:
        print("No masks to visualise.")

    # Compute AP50 if ground-truth is provided
    if args.gt and masks:
        # Build a minimal COCO-format prediction list for AP evaluation
        h, w = image_bgr.shape[:2]
        predictions = []
        for mask, box, score in zip(masks, boxes, scores):
            import pycocotools.mask as mask_util
            mask_binary = (mask > 127).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask_binary))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": 1,
                "category_id": 1,
                "segmentation": rle,
                "score": score,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "bbox_mode": 1,
            })

        # Find the image ID for the GT file (use first image in GT)
        with open(args.gt) as f:
            gt_data = json.load(f)
        if gt_data.get("images"):
            img_id = gt_data["images"][0]["id"]
            for pred in predictions:
                pred["image_id"] = img_id

        ap50 = compute_ap50(predictions, args.gt)
        if ap50 is not None:
            print(f"AP50: {ap50:.2f}%")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
