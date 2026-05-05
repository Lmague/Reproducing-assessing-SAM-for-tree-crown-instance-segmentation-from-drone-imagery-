"""
SAM + DSM Prompts - Inference with DSM-based local maxima prompts.

See NeurIPS 2025 paper, Appendix C:
- Architecture: SAM with ViT-Huge backbone
- Input: RGB image + DSM
- Prompt strategy:
    1. Compute local maxima on DSM using scipy.ndimage.maximum_filter (size=100)
    2. Use resulting (x, y) coordinates as point prompts for SAM mask decoder
- Post-processing: NMS with score threshold 0.5 and IoU threshold 0.5

Author: Lmague
Date: 2026
"""

import os
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from scipy.ndimage import maximum_filter
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import nms
import pycocotools.mask as mask_util

# --- CONFIGURATION (Appendix C) ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TEST_JSON, TEST_RGB as TEST_IMG_DIR, TEST_DSM as TEST_DSM_DIR,
    SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(BASE_DIR, "sam_dsm_prompts_predictions.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paramètres du papier (Appendix C)
# "scipy.ndimage.maximum_filter (size=100)"
DSM_FILTER_SIZE = 100
# "NMS with score threshold 0.5 and IoU threshold 0.5"
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

# --- FONCTIONS UTILITAIRES ---

def get_dsm_peaks(dsm_path, size=100):
    """Extrait les coordonnées (x, y) des sommets des arbres depuis le DSM."""
    if not os.path.exists(dsm_path):
        return np.array([])
        
    # Lecture en mode 'unchanged' pour avoir les valeurs float/int brutes
    dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    if dsm is None:
        return np.array([])

    dsm = np.nan_to_num(dsm, nan=0.0)
    
    # Filtre Maximum (Dilatation)
    local_max = maximum_filter(dsm, size=size)
    
    # Les pics sont là où la valeur originale égale la valeur dilatée
    # On ajoute un seuil minimal (moyenne) pour éviter de prendre le sol
    peaks_mask = (dsm == local_max) & (dsm > np.mean(dsm))
    
    y_indices, x_indices = np.where(peaks_mask)
    
    # Format [x, y] pour SAM
    input_points = np.column_stack((x_indices, y_indices))
    return input_points

def run_inference_on_image(predictor, image, input_points):
    """Lance SAM sur une liste de points (Batch Prompting)."""
    if len(input_points) == 0:
        return [], [], []

    # On encode l'image une seule fois
    predictor.set_image(image)
    
    # Astuce : Transformer les inputs en Tenseurs pour le mode batch rapide
    # SAM permet de prédire plusieurs objets d'un coup si on utilise l'API native
    # Mais pour faire simple et robuste, on boucle (le décodeur est léger)
    # Ou on utilise predict_torch si on veut aller très vite.
    
    # Ici, on fait une prédiction par point.
    # Note : input_points doit être transformé pour le prédicteur
    input_labels = np.ones(len(input_points)) # 1 = Foreground
    
    # On utilise predictor.predict_torch pour le batching (beaucoup plus rapide que la boucle)
    # On doit transformer les coordonnées selon la taille de l'image
    trans_points = predictor.transform.apply_coords(input_points, image.shape[:2])
    coords_torch = torch.as_tensor(trans_points, dtype=torch.float, device=predictor.device)
    labels_torch = torch.as_tensor(input_labels, dtype=torch.int, device=predictor.device)
    
    # On reshape pour avoir (Batch_Size, 1, 2) -> Chaque point est un prompt indépendant
    coords_torch = coords_torch[:, None, :]
    labels_torch = labels_torch[:, None]
    
    masks_batch, scores_batch, _ = predictor.predict_torch(
        point_coords=coords_torch,
        point_labels=labels_torch,
        multimask_output=False # On veut 1 seul masque par point
    )
    
    # masks_batch: (N, 1, H, W) -> On veut les masques binaires
    # scores_batch: (N, 1)
    
    return masks_batch.squeeze(1), scores_batch.squeeze(1)


def main():
    """Run SAM + DSM prompt inference on the test set."""
    print(f"Loading SAM (ViT-H) on {DEVICE}...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    with open(TEST_JSON, 'r') as f:
        coco_gt = json.load(f)

    images_info = coco_gt['images']
    coco_results = []

    print(f"Processing {len(images_info)} images...")

    for img_info in tqdm(images_info):
        image_id = img_info['id']
        file_name = img_info['file_name']

        img_path = os.path.join(TEST_IMG_DIR, os.path.basename(file_name))
        dsm_path = os.path.join(TEST_DSM_DIR, os.path.basename(file_name))

        if not os.path.exists(img_path) or not os.path.exists(dsm_path):
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = get_dsm_peaks(dsm_path, size=DSM_FILTER_SIZE)

        if len(points) > 0:
            masks_tensor, scores_tensor = run_inference_on_image(predictor, image, points)

            keep_indices = scores_tensor > SCORE_THRESHOLD
            masks_tensor = masks_tensor[keep_indices]
            scores_tensor = scores_tensor[keep_indices]

            if masks_tensor.shape[0] > 0:
                boxes = []
                for m in masks_tensor:
                    y, x = torch.where(m)
                    if len(x) > 0:
                        boxes.append([x.min().item(), y.min().item(), x.max().item(), y.max().item()])
                    else:
                        boxes.append([0, 0, 0, 0])

                boxes_tensor = torch.tensor(boxes, dtype=torch.float, device=DEVICE)
                keep = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)

                final_masks = masks_tensor[keep].cpu().numpy()
                final_scores = scores_tensor[keep].cpu().numpy()
                final_boxes = boxes_tensor[keep].cpu().numpy()

                for i, mask in enumerate(final_masks):
                    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    box = final_boxes[i]
                    xywh = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]

                    res = {
                        "image_id": image_id,
                        "category_id": 1,  # Zero-shot model: single class "tree"
                        "segmentation": rle,
                        "score": float(final_scores[i]),
                        "bbox": xywh,
                        "bbox_mode": 0
                    }
                    coco_results.append(res)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco_results, f)

    print(f"Done! {len(coco_results)} instances saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()