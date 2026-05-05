"""
Pre_Compute_DINOv3.py — Pré-calcul des features DINOv3 ViT-L/16 Satellite

Sauvegarde les features du ViT-L gelé (1024×64×64, float16) sur disque.
Même pattern que Pre_Compute_SAM2.py.

Usage:
    python src/misc/Pre_Compute_DINOv3.py

Stockage estimé: ~150 GB train + ~35 GB val = ~185 GB total.
Temps estimé: ~4-6h sur RTX 3090 (batch 2).
"""

import os
import sys
import json
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_RGB as TRAIN_IMG_DIR, VAL_RGB as VAL_IMG_DIR,
    DINO_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

EMBED_DIR = os.path.join(BASE_DIR, "dinov3_embeddings")
TRAIN_EMBED_DIR = os.path.join(EMBED_DIR, "train")
VAL_EMBED_DIR = os.path.join(EMBED_DIR, "val")

BATCH_SIZE = 2   # ViT-L at 1024×1024 ≈ 8-10 GB per image. OOM fallback to 1.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detectron2 normalization (0-255 scale)
PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)


# ==============================================================================
# DATASET
# ==============================================================================

class ImageListDataset(Dataset):
    """Charge les images RGB depuis un JSON COCO (pas d'annotations nécessaires)."""

    def __init__(self, json_file, img_dir):
        with open(json_file, "r") as f:
            coco = json.load(f)
        self.img_dir = img_dir
        self.images = coco["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = str(img_info["id"])
        path = os.path.join(self.img_dir, os.path.basename(img_info["file_name"]))

        image = cv2.imread(path)
        if image is None:
            print(f"Warning: image introuvable: {path}")
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (1024, 1024))

        # Tensor float32 en échelle 0-255 (comme Detectron2)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, 1024, 1024)
        # Normalisation Detectron2
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        return tensor, img_id


def collate_images(batch):
    tensors, ids = zip(*batch)
    return torch.stack(tensors), list(ids)


# ==============================================================================
# MODEL
# ==============================================================================

def load_dinov3_model():
    """Charge DINOv3 ViT-L/16 avec les poids satellite."""
    print(f"Chargement DINOv3 ViT-L/16...")
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        img_size=224,
    )

    if os.path.exists(DINO_CHECKPOINT):
        print(f"Checkpoint: {DINO_CHECKPOINT}")
        state = torch.load(DINO_CHECKPOINT, map_location="cpu", weights_only=True)
        # Key remapping (same as train_dino.py)
        clean = {}
        for k, v in state.items():
            new_k = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
            clean[new_k] = v
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print(f"  Loaded {len(clean)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"WARNING: Checkpoint introuvable: {DINO_CHECKPOINT}")
        print("  Utilisation des poids ImageNet par défaut (timm)")
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

    model.eval()
    model.to(DEVICE)
    return model


def extract_vit_features(model, images):
    """
    Extrait les features ViT avant le FPN.

    Args:
        model: timm ViT-L model
        images: (B, 3, 1024, 1024) float32 normalisé

    Returns:
        (B, 1024, 64, 64) float32
    """
    B, _, H, W = images.shape
    patch_size = 16

    # Padding to multiple of patch_size
    h_pad = (patch_size - H % patch_size) % patch_size
    w_pad = (patch_size - W % patch_size) % patch_size
    if h_pad > 0 or w_pad > 0:
        images = torch.nn.functional.pad(images, (0, w_pad, 0, h_pad))

    Hp = images.shape[2] // patch_size  # 64
    Wp = images.shape[3] // patch_size  # 64

    out = model.forward_features(images)

    # out can be (B, N+1, D) with class token, or (B, N, D) without
    if out.ndim == 3:
        expected = Hp * Wp
        if out.shape[1] == expected + 1:
            out = out[:, 1:, :]  # Remove class token at index 0
        elif out.shape[1] > expected:
            out = out[:, -expected:, :]
        # else: already correct

    # Reshape to spatial: (B, N, 1024) → (B, 1024, Hp, Wp)
    out = out.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

    return out  # (B, 1024, 64, 64)


# ==============================================================================
# EXTRACTION LOOP
# ==============================================================================

def extract_and_save(loader, model, save_dir, split_name):
    """Pré-calcule et sauvegarde les features ViT pour toutes les images."""
    os.makedirs(save_dir, exist_ok=True)

    # Pre-scan for already computed files (resume support)
    already_done = set(f.replace(".pt", "") for f in os.listdir(save_dir) if f.endswith(".pt"))
    print(f"[{split_name}] {len(already_done)} fichiers déjà calculés, à traiter: {len(loader.dataset) - len(already_done)}")

    skipped = 0
    saved = 0

    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc=f"Pre-compute {split_name}"):
            # Filter already-computed
            todo = [i for i, iid in enumerate(img_ids) if iid not in already_done]
            if not todo:
                skipped += len(img_ids)
                continue

            images = images.to(DEVICE)
            features = extract_vit_features(model, images)  # (B, 1024, 64, 64)

            for i in todo:
                feat = features[i].cpu().half()  # float16
                torch.save(feat, os.path.join(save_dir, f"{img_ids[i]}.pt"))
                saved += 1

            skipped += len(img_ids) - len(todo)

    print(f"[{split_name}] Sauvegardé: {saved}, Skippé: {skipped}")
    return saved


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Pre-Compute DINOv3 ViT-L/16 Satellite Features")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {DINO_CHECKPOINT}")
    print(f"Data:       {DATA_ROOT}")
    print(f"Output:     {EMBED_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)

    model = load_dinov3_model()

    # Dummy forward to verify output shape and estimate storage
    dummy = torch.randn(1, 3, 1024, 1024, device=DEVICE)
    with torch.no_grad():
        dummy_out = extract_vit_features(model, dummy)
    print(f"\nOutput shape: {dummy_out.shape}")  # Expected: (1, 1024, 64, 64)
    size_mb = dummy_out.half().nelement() * 2 / 1e6
    print(f"Size per image (float16): {size_mb:.1f} MB")
    print(f"Estimated train ({18746} images): {18746 * size_mb / 1e3:.0f} GB")
    print(f"Estimated val ({4392} images): {4392 * size_mb / 1e3:.0f} GB")
    del dummy, dummy_out
    torch.cuda.empty_cache()

    # --- Train ---
    print(f"\n--- Train split ---")
    train_dataset = ImageListDataset(TRAIN_JSON, TRAIN_IMG_DIR)
    print(f"  Images: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_images, num_workers=4, pin_memory=True,
    )
    extract_and_save(train_loader, model, TRAIN_EMBED_DIR, "train")

    # --- Val ---
    print(f"\n--- Val split ---")
    val_dataset = ImageListDataset(VAL_JSON, VAL_IMG_DIR)
    print(f"  Images: {len(val_dataset)}")
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_images, num_workers=4, pin_memory=True,
    )
    extract_and_save(val_loader, model, VAL_EMBED_DIR, "val")

    # --- Verify ---
    train_count = len([f for f in os.listdir(TRAIN_EMBED_DIR) if f.endswith(".pt")])
    val_count = len([f for f in os.listdir(VAL_EMBED_DIR) if f.endswith(".pt")])
    print(f"\n{'='*60}")
    print(f"TERMINÉ!")
    print(f"  Train: {train_count} fichiers dans {TRAIN_EMBED_DIR}")
    print(f"  Val:   {val_count} fichiers dans {VAL_EMBED_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # OOM fallback: try batch 2, then 1
    for bs in [BATCH_SIZE, 1]:
        try:
            BATCH_SIZE = bs
            main()
            break
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] Batch size {bs} trop grand.")
            torch.cuda.empty_cache()
            gc.collect()
            if bs == 1:
                raise
