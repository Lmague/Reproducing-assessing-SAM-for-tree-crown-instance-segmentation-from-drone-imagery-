"""
run_all_inferences.py — Inférence sur le test set pour les 3 modèles manquants

Génère les fichiers JSON COCO pour :
1. Mask R-CNN RGB+DSM  → mask_rcnn_dsm_predictions.json
2. RSPrompter          → rsprompter_predictions.json
3. DSMPrompter v2      → dsm_prompter_v2_predictions.json

Usage:
    python3 src/inference/run_all_inferences.py
    python3 src/inference/run_all_inferences.py --model mask_rcnn_dsm
    python3 src/inference/run_all_inferences.py --model rsprompter
    python3 src/inference/run_all_inferences.py --model dsm_prompter_v2
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pycocotools.mask as mask_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))
MODELS_DIR = os.path.join(REPO_DIR, "src/models")

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from utils.paths import (
    DATA_ROOT, TEST_JSON,
    TEST_RGB as _TEST_RGB, TEST_DSM as _TEST_DSM,
    SAM_CHECKPOINT, SAM2_CHECKPOINT,
)
TEST_RGB = _TEST_RGB
TEST_DSM = _TEST_DSM
OUTPUT_DIR  = os.path.join(MODELS_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# HELPER — NMS sur boîtes XYXY
# ============================================================================
def nms_boxes(boxes, scores, iou_thr=0.5):
    """Simple NMS. boxes: (N,4) XYXY, scores: (N,). Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)
    kept = []
    while len(order) > 0:
        i = order[0].item()
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = x1[rest].clamp(min=x1[i])
        iy1 = y1[rest].clamp(min=y1[i])
        ix2 = x2[rest].clamp(max=x2[i])
        iy2 = y2[rest].clamp(max=y2[i])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= iou_thr]
    return kept


def encode_mask_rle(binary_mask):
    """Encode a binary uint8 HxW mask to COCO RLE."""
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ============================================================================
# 1. MASK R-CNN RGB+DSM
# ============================================================================
def run_mask_rcnn_dsm():
    print("\n" + "=" * 60)
    print("  Mask R-CNN RGB+DSM — Inférence test set")
    print("=" * 60)

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    # Register dataset
    try:
        register_coco_instances("trees_test",  {}, TEST_JSON,  DATA_ROOT)
    except AssertionError:
        pass

    from utils.coco_helpers import fix_coco_json_if_needed
    fix_coco_json_if_needed(TEST_JSON)

    # Import the model to get the custom backbone registered
    import Mask_RCNN_RGB_DSM as mrcnn_dsm
    cfg = mrcnn_dsm.build_cfg(num_classes=7, batch_size=4)
    cfg.DATASETS.TEST = ("trees_test",)
    cfg.MODEL.WEIGHTS = os.path.join(MODELS_DIR, "output_mask_rcnn_RGB_DSM", "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    output_file = os.path.join(OUTPUT_DIR, "mask_rcnn_dsm_predictions.json")
    evaluator = COCOEvaluator("trees_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = mrcnn_dsm.Trainer4ch.build_test_loader(cfg, "trees_test")

    results = inference_on_dataset(model, test_loader, evaluator)
    print(results)

    # Copy coco_instances_results.json to standard name
    src = os.path.join(cfg.OUTPUT_DIR, "coco_instances_results.json")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, output_file)
        print(f"✅ Sauvegardé: {output_file}")
    else:
        print(f"⚠️  coco_instances_results.json introuvable dans {cfg.OUTPUT_DIR}")

    return output_file


# ============================================================================
# 2. RSPrompter
# ============================================================================
def run_rsprompter(score_thr=0.1, nms_iou=0.5):
    print("\n" + "=" * 60)
    print("  RSPrompter — Inférence test set")
    print("=" * 60)

    from segment_anything import sam_model_registry
    from RSPrompter import RSPrompterFast

    MODEL_WEIGHTS  = os.path.join(MODELS_DIR, "output_rsprompter_fast", "best.pth")
    output_file    = os.path.join(OUTPUT_DIR, "rsprompter_predictions.json")

    # Load full SAM1 (encoder + decoder)
    print("  Chargement SAM1 ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
    sam.eval()

    # Load RSPrompter (shares decoder with SAM1)
    print("  Chargement RSPrompter weights...")
    model = RSPrompterFast(SAM_CHECKPOINT, num_proposals=10).to(DEVICE)
    state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Load test images
    with open(TEST_JSON) as f:
        coco_gt = json.load(f)

    results = []
    print(f"  {len(coco_gt['images'])} images à traiter...")

    with torch.no_grad():
        for img_info in tqdm(coco_gt["images"]):
            image_id   = img_info["id"]
            img_path   = os.path.join(TEST_RGB, os.path.basename(img_info["file_name"]))
            orig_h     = img_info.get("height", 1024)
            orig_w     = img_info.get("width",  1024)

            if not os.path.exists(img_path):
                continue

            # Read + preprocess image for SAM1
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (1024, 1024))

            # SAM1 image encoder
            img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            # SAM1 expects values in [0,255] after pixel_mean/std normalization done internally
            # Actually segment_anything normalizes internally when using set_image, but for raw encoder:
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
            pixel_std  = torch.tensor([58.395,  57.12,  57.375]).view(3, 1, 1).to(DEVICE)
            img_norm   = (img_tensor - pixel_mean) / pixel_std
            embedding  = sam.image_encoder(img_norm)  # (1, 256, 64, 64)

            # RSPrompter forward
            masks, ious, boxes, scores = model(embedding)
            # masks: (1, K, 1, 256, 256), boxes: (1, K, 4) XYXY in 1024 space

            K = boxes.shape[1]
            masks_up = F.interpolate(
                masks[0].float(),             # (K, 1, 256, 256)
                size=(1024, 1024), mode="bilinear", align_corners=False
            )
            binary_masks = (masks_up[:, 0] > 0).cpu().numpy()  # (K, 1024, 1024)
            box_arr      = boxes[0].cpu()                       # (K, 4)
            score_arr    = (ious[0, :, 0]).cpu()                # (K,)
            cat_arr      = torch.zeros(K, dtype=torch.long)     # RSPrompter = class-agnostic

            # NMS
            kept = nms_boxes(box_arr, score_arr, iou_thr=nms_iou)

            for idx in kept:
                s = score_arr[idx].item()
                if s < score_thr:
                    continue

                # Resize mask to original image size
                msk = binary_masks[idx]
                if orig_h != 1024 or orig_w != 1024:
                    msk = cv2.resize(msk.astype(np.uint8), (orig_w, orig_h)) > 0

                rle = encode_mask_rle(msk)
                x1, y1, x2, y2 = box_arr[idx].tolist()
                # Scale box to original size
                sx = orig_w / 1024.0
                sy = orig_h / 1024.0
                results.append({
                    "image_id":    image_id,
                    "category_id": 1,  # class-agnostic "tree"
                    "bbox":        [x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy],
                    "score":       s,
                    "segmentation": rle,
                })

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"✅ {len(results)} prédictions → {output_file}")
    return output_file


# ============================================================================
# 3. DSMPrompter v2
# ============================================================================
def run_dsm_prompter_v2(score_thr=0.1, nms_iou=0.5, num_proposals=5):
    print("\n" + "=" * 60)
    print("  DSMPrompter v2 — Inférence test set")
    print("=" * 60)

    from DSM_Prompter_v2 import DSMPrompterInference

    SAM2_CONFIG    = "configs/sam2.1/sam2.1_hiera_l.yaml"  # relatif au CWD (REPO_DIR)
    SAM2_CKPT      = SAM2_CHECKPOINT
    MODEL_WEIGHTS  = os.path.join(REPO_DIR, "src/training/output_dsm_prompter_v2/model_best.pth")
    output_file    = os.path.join(OUTPUT_DIR, "dsm_prompter_v2_predictions.json")

    print("  Chargement DSMPrompterInference (SAM2 + trained weights)...")
    model = DSMPrompterInference(SAM2_CONFIG, SAM2_CKPT, num_proposals=num_proposals)
    model.load_training_weights(MODEL_WEIGHTS)
    model.to(DEVICE)
    model.eval()

    with open(TEST_JSON) as f:
        coco_gt = json.load(f)

    results = []
    print(f"  {len(coco_gt['images'])} images à traiter...")

    with torch.no_grad():
        for img_info in tqdm(coco_gt["images"]):
            image_id = img_info["id"]
            orig_h   = img_info.get("height", 1024)
            orig_w   = img_info.get("width",  1024)
            img_path = os.path.join(TEST_RGB, os.path.basename(img_info["file_name"]))

            if not os.path.exists(img_path):
                continue

            # Load RGB
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (1024, 1024))
            rgb_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

            # Load DSM
            base_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
            dsm = None
            for ext in [".tif", ".TIF", ".png", ".PNG"]:
                p = os.path.join(TEST_DSM, base_name + ext)
                if os.path.exists(p):
                    dsm = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    break
            if dsm is None:
                dsm = np.zeros((1024, 1024), dtype=np.float32)
            else:
                dsm = dsm.astype(np.float32)
                dsm = np.nan_to_num(dsm, nan=0.0)
                m = dsm.max()
                if m > 0:
                    dsm /= m
            dsm_tensor = torch.from_numpy(cv2.resize(dsm, (1024, 1024))).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Forward
            low_res_masks, iou_preds, proposals, scores = model(rgb_tensor, dsm_tensor)
            # low_res_masks: (1, K, 1, 256, 256), proposals: (1, K, 4)

            K = proposals.shape[1]
            masks_up = F.interpolate(
                low_res_masks[0].float(),   # (K, 1, 256, 256)
                size=(1024, 1024), mode="bilinear", align_corners=False
            )
            binary_masks = (masks_up[:, 0] > 0).cpu().numpy()
            box_arr      = proposals[0].cpu()
            score_arr    = (scores[0] * iou_preds[0, :, 0]).cpu()  # combine box score + iou

            # NMS
            kept = nms_boxes(box_arr, score_arr, iou_thr=nms_iou)

            for idx in kept:
                s = score_arr[idx].item()
                if s < score_thr:
                    continue

                msk = binary_masks[idx]
                if orig_h != 1024 or orig_w != 1024:
                    msk = cv2.resize(msk.astype(np.uint8), (orig_w, orig_h)) > 0

                rle = encode_mask_rle(msk)
                x1, y1, x2, y2 = box_arr[idx].tolist()
                sx = orig_w / 1024.0
                sy = orig_h / 1024.0
                results.append({
                    "image_id":    image_id,
                    "category_id": 1,  # class-agnostic (DSMPrompter v2 is box-only)
                    "bbox":        [x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy],
                    "score":       s,
                    "segmentation": rle,
                })

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"✅ {len(results)} prédictions → {output_file}")
    return output_file


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mask_rcnn_dsm", "rsprompter", "dsm_prompter_v2", "all"],
                    default="all", help="Quel modèle inférer (défaut: all)")
    ap.add_argument("--score-thr", type=float, default=0.1, help="Score threshold (défaut=0.1)")
    ap.add_argument("--num-proposals", type=int, default=5, help="Nombre de proposals DSMPrompter (défaut=5)")
    args = ap.parse_args()

    run_all = args.model == "all"

    if run_all or args.model == "mask_rcnn_dsm":
        try:
            run_mask_rcnn_dsm()
        except Exception as e:
            print(f"❌ Mask R-CNN DSM ERREUR: {e}")
            import traceback; traceback.print_exc()

    if run_all or args.model == "rsprompter":
        try:
            run_rsprompter(score_thr=args.score_thr)
        except Exception as e:
            print(f"❌ RSPrompter ERREUR: {e}")
            import traceback; traceback.print_exc()

    if run_all or args.model == "dsm_prompter_v2":
        try:
            run_dsm_prompter_v2(score_thr=args.score_thr, num_proposals=args.num_proposals)
        except Exception as e:
            print(f"❌ DSMPrompter v2 ERREUR: {e}")
            import traceback; traceback.print_exc()

    print("\n✅ Inférences terminées. JSONs dans:", OUTPUT_DIR)
