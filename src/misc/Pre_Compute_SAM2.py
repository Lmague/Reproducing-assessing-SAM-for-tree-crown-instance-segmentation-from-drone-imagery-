"""
Pre_Compute_SAM2.py — Pré-calcul des embeddings SAM2 Hiera Large.

Sauvegarde uniquement vision_features (256x64x64) en float16.
backbone_fpn (haute résolution) est trop volumineux (~675GB) donc ignoré.
DSMPrompter sera modifié pour fonctionner sans high_res_features.
"""
import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_RGB as TRAIN_IMG_DIR, VAL_RGB as VAL_IMG_DIR,
    SAM2_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "sam2_embeddings")
os.makedirs(os.path.join(EMBED_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(EMBED_DIR, "val"), exist_ok=True)
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 4


class ImageListDataset(Dataset):
    def __init__(self, json_file, img_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.images = data['images']
        self.img_dir = img_dir
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.pixel_std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_id = info['id']
        path = os.path.join(self.img_dir, os.path.basename(info['file_name']))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        return tensor, str(img_id)


def extract_features(loader, sam, save_dir):
    """Sauvegarde vision_features (256x64x64) en float16."""
    already_done = set(f.replace('.pt', '') for f in os.listdir(save_dir) if f.endswith('.pt'))
    skipped = 0
    with torch.no_grad():
        for images, img_ids in tqdm(loader):
            # Skip si déjà calculé
            todo = [i for i, iid in enumerate(img_ids) if iid not in already_done]
            if not todo:
                skipped += len(img_ids)
                continue
            images = images.to(DEVICE)
            out = sam.forward_image(images)
            vision_features = out["vision_features"].half().cpu()
            for i in todo:
                torch.save(vision_features[i].clone(),
                           os.path.join(save_dir, f"{img_ids[i]}.pt"))
    if skipped:
        print(f"  ({skipped} images déjà calculées, skippées)")


def main():
    print("Chargement SAM2 Hiera Large...")
    from sam2.build_sam import build_sam2
    sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE, apply_postprocessing=False)
    sam.eval()

    # Vérifier la taille output
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 1024, 1024, device=DEVICE)
        out = sam.forward_image(dummy)
        vf_shape = out["vision_features"].shape
        fpn_shapes = [f.shape for f in out["backbone_fpn"]]
    print(f"vision_features shape : {vf_shape}")
    print(f"backbone_fpn shapes   : {fpn_shapes}")
    vf_mb = (vf_shape[1]*vf_shape[2]*vf_shape[3]*2) / 1e6
    fpn_mb = sum((f[1]*f[2]*f[3]*2)/1e6 for f in fpn_shapes)
    total_imgs = 18746 + 4392
    print(f"Taille vision_features par image  : {vf_mb:.1f} MB → total {vf_mb*total_imgs/1024:.1f} GB")
    print(f"Taille backbone_fpn par image     : {fpn_mb:.1f} MB → total {fpn_mb*total_imgs/1024:.1f} GB (non sauvegardé)")

    print("\n--- Traitement TRAIN ---")
    train_ds = ImageListDataset(TRAIN_JSON, TRAIN_IMG_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
    extract_features(train_loader, sam, os.path.join(EMBED_DIR, "train"))

    print("\n--- Traitement VAL ---")
    val_ds = ImageListDataset(VAL_JSON, VAL_IMG_DIR)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
    extract_features(val_loader, sam, os.path.join(EMBED_DIR, "val"))

    print(f"\nTerminé ! Embeddings dans {EMBED_DIR}")


if __name__ == "__main__":
    main()
