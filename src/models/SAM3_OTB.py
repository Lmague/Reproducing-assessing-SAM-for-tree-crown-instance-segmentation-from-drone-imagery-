"""
SAM3 OTB (Out-of-the-Box) - Inférence zéro-shot multimodale

SAM3 est multimodal: il accepte des prompts texte + image.
On exploite les 7 espèces d'arbres comme prompts texte.

Prérequis:
  pip install sam3
  huggingface-cli login   # puis accepter la licence sur hf.co/facebook/sam3

API:
  state = processor.set_image(image)
  state = processor.set_text_prompt(prompt, state)
  masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
"""

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_util

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(BASE_DIR, "sam3_otb_predictions.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Score threshold pour filtrer les détections
CONFIDENCE_THRESHOLD = 0.3

# NMS IoU threshold
NMS_IOU_THRESH = 0.5

# ============================================================================
# PROMPTS TEXTE PAR ESPÈCE
# category_id → prompt texte (selon catégories COCO du projet)
# ============================================================================
SPECIES_PROMPTS = [
    (1, "tree crown aerial view"),
    (2, "jack pine tree crown aerial view"),
    (3, "white spruce tree crown aerial view"),
    (4, "black spruce tree crown aerial view"),
    (5, "eastern white pine tree crown aerial view"),
    (6, "eastern white cedar tree crown aerial view"),
    (7, "american elm tree crown aerial view"),
]


def apply_nms_masks(masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """NMS sur masques binaires."""
    if len(masks) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)
    return keep


def run_sam3_otb_inference(species_prompts=None, output_file=None):
    """
    Inférence SAM3 OTB avec prompts texte par espèce d'arbre.
    
    Flux par image:
      1. set_image() → encode vision features (une fois)
      2. Pour chaque espèce: set_text_prompt() → masques + scores
      3. NMS global toutes espèces, encode en RLE, sauvegarde COCO
    """
    if species_prompts is None:
        species_prompts = SPECIES_PROMPTS
    if output_file is None:
        output_file = OUTPUT_FILE

    print("=" * 60)
    print("SAM3 OTB - Inférence multimodale texte+image")
    print(f"Output: {output_file}")
    print("=" * 60)

    # 1. Charger le modèle (télécharge depuis HuggingFace si besoin)
    print(f"Chargement SAM3 sur {DEVICE}...")
    print("(Si erreur d'authentification: huggingface-cli login)")
    try:
        model = build_sam3_image_model(device=DEVICE, eval_mode=True, load_from_HF=True)
        processor = Sam3Processor(model, device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD)
        print("SAM3 chargé.")
    except Exception as e:
        print(f"ERREUR chargement SAM3: {e}")
        print("Vérifiez: huggingface-cli login + accès à hf.co/facebook/sam3")
        return

    # 2. Charger le split test
    with open(TEST_JSON, 'r') as f:
        coco_gt = json.load(f)

    images_info = coco_gt['images']
    coco_results = []

    print(f"Traitement de {len(images_info)} images × {len(species_prompts)} prompts...")

    for img_info in tqdm(images_info):
        image_id = img_info['id']
        file_name = img_info['file_name']

        img_path = os.path.join(TEST_IMG_DIR, os.path.basename(file_name))
        if not os.path.exists(img_path):
            continue

        # Ouvrir l'image en RGB via PIL (attendu par Sam3Processor)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        orig_h, orig_w = image_bgr.shape[:2]

        # 3. Encoder les features visuelles une seule fois
        try:
            state = processor.set_image(image_pil)
        except Exception:
            continue

        all_masks = []
        all_scores = []
        all_cat_ids = []

        # 4. Passer chaque prompt texte (réutilise les features visuelles)
        for cat_id, text_prompt in species_prompts:
            try:
                result_state = processor.set_text_prompt(text_prompt, state.copy())
            except Exception:
                continue

            masks_tensor = result_state.get("masks")   # (N, 1, H, W) bool
            scores_tensor = result_state.get("scores")  # (N,)

            if masks_tensor is None or len(masks_tensor) == 0:
                continue

            masks_np = masks_tensor.squeeze(1).cpu().numpy().astype(bool)  # (N, H, W)
            scores_np = scores_tensor.cpu().numpy()                         # (N,)

            for mask, score in zip(masks_np, scores_np):
                all_masks.append(mask)
                all_scores.append(float(score))
                all_cat_ids.append(cat_id)

        if len(all_masks) == 0:
            continue

        # 5. NMS global (toutes espèces confondues)
        keep_indices = apply_nms_masks(
            np.array(all_masks), np.array(all_scores), NMS_IOU_THRESH
        )

        # 6. Encoder résultats au format COCO
        for idx in keep_indices:
            mask = all_masks[idx]
            score = all_scores[idx]
            cat_id = all_cat_ids[idx]

            # RLE
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')

            # Bounding box depuis masque
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any():
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_xywh = [float(cmin), float(rmin),
                         float(cmax - cmin + 1), float(rmax - rmin + 1)]

            coco_results.append({
                "image_id": image_id,
                "category_id": cat_id,
                "segmentation": rle,
                "score": score,
                "bbox": bbox_xywh,
                "bbox_mode": 1,  # XYWH
            })

    # Sauvegarde
    with open(output_file, 'w') as f:
        json.dump(coco_results, f)

    print(f"Prédictions sauvegardées: {output_file}")
    print(f"Total: {len(coco_results)} instances détectées")
    return coco_results


# ============================================================================
# MAIN
# ============================================================================

SPECIES_PROMPTS_GENERIC = [
    (1, "tree crown"),
    (2, "jack pine"),
    (3, "white spruce"),
    (4, "black spruce"),
    (5, "eastern white pine"),
    (6, "eastern white cedar"),
    (7, "american elm"),
]

SPECIES_PROMPTS_DESCRIPTIVE = [
    (1, "tree crown"),
    (2, "jack pine sparse yellowish-green crown"),
    (3, "white spruce dense blue-green oval crown"),
    (4, "black spruce narrow dark green spire crown"),
    (5, "eastern white pine large flat blue-green crown"),
    (6, "eastern white cedar dense bright green crown"),
    (7, "american elm broad rounded green crown"),
]

if __name__ == "__main__":
    run_sam3_otb_inference(
        species_prompts=SPECIES_PROMPTS_GENERIC,
        output_file=os.path.join(BASE_DIR, "sam3_otb_generic_predictions.json"),
    )
    run_sam3_otb_inference(
        species_prompts=SPECIES_PROMPTS_DESCRIPTIVE,
        output_file=os.path.join(BASE_DIR, "sam3_otb_descriptive_predictions.json"),
    )
