"""
Inference script for DSMPrompter model on test set.

See NeurIPS 2025 paper, Appendix C, and src/models/DSM_Prompter.py.

Author: Lmague
Date: 2026
"""

import os
import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

from src.models.DSM_Prompter import DSMPrompterSAM2

# --- CONFIGURATION ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.paths import (
    DATA_ROOT, TEST_JSON,
    TEST_RGB as TEST_IMG_DIR, TEST_DSM as TEST_DSM_DIR,
    SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le modèle entraîné (à adapter selon ton epoch)
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "output_dsm_prompter/model_epoch_20.pth")
OUTPUT_FILE = os.path.join(BASE_DIR, "dsm_prompter_predictions.json")
DEVICE = "cuda"

def main():
    print(f"Chargement du modèle DSMPrompter depuis {TRAINED_MODEL_PATH}...")
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    model = DSMPrompterSAM2(config_file=sam2_config, ckpt_path=SAM_CHECKPOINT, num_proposals=100)
    
    # Charger les poids (en gérant le cas où on a sauvé tout le state_dict ou juste le model)
    checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()

    # 2. Charger la liste des images de test
    with open(TEST_JSON, 'r') as f:
        coco_gt = json.load(f)
    images_info = coco_gt['images']
    
    coco_results = []
    
    print("Lancement de l'inférence sur le Test Set...")
    
    with torch.no_grad():
        for img_info in tqdm(images_info):
            image_id = img_info['id']
            file_name = img_info['file_name']
            
            # --- Préparation des données (copié du dataloader) ---
            rgb_path = os.path.join(TEST_IMG_DIR, os.path.basename(file_name))
            dsm_path = os.path.join(TEST_DSM_DIR, os.path.basename(file_name))
            
            if not os.path.exists(dsm_path):
                # Fallback extension
                base = os.path.splitext(os.path.basename(file_name))[0]
                dsm_path = os.path.join(TEST_DSM_DIR, base + ".tif")
            
            if not os.path.exists(rgb_path) or not os.path.exists(dsm_path):
                continue

            # RGB
            image = cv2.imread(rgb_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]
            
            # DSM
            dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
            dsm = np.nan_to_num(dsm)
            if dsm.max() > 0:
                dsm = dsm / dsm.max()
                
            # Resize 1024x1024 (taille d'entrée SAM)
            image_1024 = cv2.resize(image, (1024, 1024))
            dsm_1024 = cv2.resize(dsm, (1024, 1024))
            
            # Tenseurs
            img_tensor = torch.as_tensor(image_1024.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
            dsm_tensor = torch.as_tensor(dsm_1024).float().unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 1024, 1024)

            # --- Inférence ---
            # low_res_masks: (1, K, 1, 256, 256)
            # scores: (1, K)
            low_res_masks, iou_preds, boxes, scores = model(img_tensor, dsm_tensor)
            
            # --- Post-Processing ---
            # On récupère le batch 0
            pred_masks = low_res_masks[0, :, 0] # (K, 256, 256)
            pred_scores = scores[0]             # (K,)
            
            # Filtre par score (ex: > 0.1 pour garder des candidats, le vrai tri se fait après)
            keep = pred_scores > 0.0
            pred_masks = pred_masks[keep]
            pred_scores = pred_scores[keep]
            
            if len(pred_scores) == 0:
                continue

            # Interpolation des masques vers la taille originale
            # Attention: SAM sort des masques 256x256 logits.
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0), 
                size=(original_h, original_w), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)
            
            # Binarisation
            pred_masks = (pred_masks > 0.0).cpu().numpy().astype(np.uint8)
            pred_scores = pred_scores.cpu().numpy()
            
            # Formatage COCO
            for k in range(len(pred_scores)):
                mask = pred_masks[k]
                if mask.sum() < 10: # Filtre tout petit bruit
                    continue
                
                rle = mask_util.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Bbox depuis le masque
                y_indices, x_indices = np.where(mask)
                x1, x2 = np.min(x_indices), np.max(x_indices)
                y1, y2 = np.min(y_indices), np.max(y_indices)
                w, h = x2 - x1, y2 - y1
                
                res = {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(pred_scores[k]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "bbox_mode": 0
                }
                coco_results.append(res)

    # Sauvegarde
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco_results, f)
    print(f"Prédictions sauvegardées dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()