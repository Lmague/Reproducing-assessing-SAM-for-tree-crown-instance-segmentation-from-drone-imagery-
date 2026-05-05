"""
STEP 1: Pre-compute SAM embeddings.

Saves SAM ViT-H image encoder features to disk for faster RSPrompter training.
See NeurIPS 2025 paper, Section 3.2 (RSPrompter).

Author: Lmague
Date: 2026
"""
import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_RGB as TRAIN_IMG_DIR, VAL_RGB as VAL_IMG_DIR,
    SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "sam_embeddings_vith")
os.makedirs(os.path.join(EMBED_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(EMBED_DIR, "val"), exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # A40 a 48GB, on peut augmenter le batch pour l'inférence seule

class ImageListDataset(Dataset):
    def __init__(self, json_file, img_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.images = data['images']
        self.img_dir = img_dir
        
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        file_name = info['file_name']
        img_id = info['id']
        
        path = os.path.join(self.img_dir, os.path.basename(file_name))
        img = cv2.imread(path)
        if img is None:
            # Image noire si erreur (ne devrait pas arriver)
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(img, (1024, 1024))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        
        return tensor, str(img_id)

def extract_features(loader, model, save_dir):
    with torch.no_grad():
        for images, img_ids in tqdm(loader):
            images = images.to(DEVICE)
            # Encoder via SAM
            embeddings = model.image_encoder(images) # (B, 256, 64, 64)
            
            # Sauvegarder chaque embedding individuellement
            # On utilise float16 pour gagner 50% d'espace disque (très sûr pour SAM)
            embeddings = embeddings.half().cpu()
            
            for i, img_id in enumerate(img_ids):
                save_path = os.path.join(save_dir, f"{img_id}.pt")
                torch.save(embeddings[i].clone(), save_path)

def main():
    print("Chargement de SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    sam.eval()
    
    print("--- Traitement TRAIN ---")
    train_ds = ImageListDataset(TRAIN_JSON, TRAIN_IMG_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
    extract_features(train_loader, sam, os.path.join(EMBED_DIR, "train"))
    
    print("--- Traitement VAL ---")
    val_ds = ImageListDataset(VAL_JSON, VAL_IMG_DIR)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
    extract_features(val_loader, sam, os.path.join(EMBED_DIR, "val"))
    
    print(f"Terminé ! Embeddings sauvegardés dans {EMBED_DIR}")

if __name__ == "__main__":
    main()