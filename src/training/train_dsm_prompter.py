"""
Training script for DSMPrompter (SAM2 + DSM).

See NeurIPS 2025 paper, Appendix C, and src/models/DSM_Prompter.py.

Author: Lmague
Date: 2026
"""

import os
import torch
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pycocotools import mask as coco_mask
from tqdm import tqdm

# Import du modèle SAM2+DSM
from src.models.DSM_Prompter import DSMPrompterSAM2, DSMLoss, HungarianMatcher, collate_fn

# --- CONFIGURATION (Appendix C) ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_RGB as TRAIN_IMG_DIR, VAL_RGB as VAL_IMG_DIR,
    TRAIN_DSM as TRAIN_DSM_DIR, VAL_DSM as VAL_DSM_DIR,
    SAM2_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_dsm_prompter")
TRAINING_LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparamètres du papier (Appendix C) - Adaptés pour A40 (48GB)
# "batch size 2" pour RSPrompter/SAM-based methods
# A40: Batch 2 devrait passer avec SAM2, sinon réduire à 1
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2  # batch effectif = 16 * 2 = 32
# "AdamW optimizer with learning rate 1e-5"
LR = 1e-5
# "weight decay 0.1"
WEIGHT_DECAY = 0.1
# "trained for a maximum of 100 epochs"
EPOCHS = 100
# Validation tous les N epochs
VAL_EVERY = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embeddings SAM2 pré-calculés (activé automatiquement si le dossier existe)
SAM2_EMBED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../misc/sam2_embeddings")
USE_PRECOMPUTED = os.path.exists(os.path.join(SAM2_EMBED_DIR, "train"))
if USE_PRECOMPUTED:
    print(f"[INFO] Embeddings SAM2 pré-calculés détectés dans {SAM2_EMBED_DIR}")

class PlantationsDataset(Dataset):
    """Dataset pour RGB + DSM avec annotations COCO.
    Normalisation ImageNet pour SAM2.
    """
    def __init__(self, json_file, img_dir, dsm_dir):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        
        self.img_dir = img_dir
        self.dsm_dir = dsm_dir
        
        # Normalisation ImageNet (standard pour SAM2)
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.img_ids = list(self.images.keys())
        
        # Récupérer les dimensions originales pour le scaling
        self.orig_sizes = {}
        for img in self.coco['images']:
            self.orig_sizes[img['id']] = (img.get('height', 1024), img.get('width', 1024))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        orig_h, orig_w = self.orig_sizes[img_id]
        
        # --- 1. RGB ---
        rgb_path = os.path.join(self.img_dir, os.path.basename(img_info['file_name']))
        image = cv2.imread(rgb_path)
        if image is None:
            # Fallback si image introuvable
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            print(f"Warning: Image non trouvée: {rgb_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize 1024x1024 (Requis par SAM2)
        # image = cv2.resize(image, (1024, 1024))  # deja 1024x1024
        
        # Normalisation ImageNet (0-1 puis normalize)
        input_image = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0
        input_image = (input_image - self.pixel_mean) / self.pixel_std

        # --- 2. DSM ---
        base_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        # Essayer plusieurs extensions
        dsm_path = None
        for ext in ['.tif', '.png', '.TIF', '.PNG']:
            candidate = os.path.join(self.dsm_dir, base_name + ext)
            if os.path.exists(candidate):
                dsm_path = candidate
                break
        
        if dsm_path is None:
            dsm = np.zeros((1024, 1024), dtype=np.float32)
        else:
            dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
            if dsm is None:
                dsm = np.zeros((1024, 1024), dtype=np.float32)
            else:
                dsm = np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)
                dsm = dsm.astype(np.float32)  # deja 1024x1024
                # Normalisation per-sample (Appendix C du papier)
                max_val = np.max(dsm)
                if max_val > 0:
                    dsm = dsm / max_val

        dsm_tensor = torch.as_tensor(dsm).float().unsqueeze(0)

        # --- 3. Targets (Boxes & Masks) ---
        anns = self.annotations.get(img_id, [])
        boxes = []
        masks = []
        
        # Facteurs de scale pour passer des coords originales à 1024x1024
        scale_x = 1024.0 / orig_w
        scale_y = 1024.0 / orig_h
        
        for ann in anns:
            # Box (XYWH -> XYXY) avec scaling
            x, y, w, h = ann['bbox']
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])

            # Masque
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygone -> RLE -> Bitmap
                    rle = coco_mask.frPyObjects(ann['segmentation'], orig_h, orig_w)
                    m = coco_mask.decode(rle)
                    if len(m.shape) == 3:
                        m = m.sum(axis=2) > 0
                else:
                    # Déjà RLE
                    m = coco_mask.decode(ann['segmentation'])
                
                # Resize le masque à 1024x1024
                m = cv2.resize(m.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
                masks.append(m)
            else:
                masks.append(np.zeros((1024, 1024), dtype=np.uint8))

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.tensor(np.array(masks), dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, 1024, 1024), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes, 
            "labels": labels,
            "masks": masks
        }

        return {
            "image": input_image, 
            "dsm": dsm_tensor, 
            "targets": target
        }


def local_collate_fn(batch):
    """Custom collate car targets ont des tailles variables."""
    images = torch.stack([b['image'] for b in batch])
    dsms = torch.stack([b['dsm'] for b in batch])
    targets = [b['targets'] for b in batch]
    return {"image": images, "dsm": dsms, "targets": targets}


def _save_config_snapshot(config_dict, checkpoint_path: str) -> None:
    """
    Saves a JSON snapshot of the training configuration alongside a checkpoint.

    Args:
        config_dict: Dictionary of hyperparameters.
        checkpoint_path: Path to the checkpoint .pth file.
    """
    import json as _json
    json_path = checkpoint_path + ".config.json"
    with open(json_path, "w") as f:
        _json.dump(config_dict, f, indent=2)


def validate(model, dataloader, criterion, device):
    """Évaluation sur le set de validation."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["image"].to(device)
            dsm = batch["dsm"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            
            precomputed_emb = None
            if USE_PRECOMPUTED and "img_id" in batch:
                embs = []
                for img_id in batch["img_id"]:
                    p = os.path.join(SAM2_EMBED_DIR, "val", f"{img_id}.pt")
                    embs.append(torch.load(p, map_location="cpu") if os.path.exists(p) else None)
                if all(e is not None for e in embs):
                    precomputed_emb = torch.stack(embs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                masks, ious, boxes, scores = model(rgb, dsm, precomputed_emb=precomputed_emb)
                outputs = {"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    model.train()
    return total_loss / max(len(dataloader), 1)


def main():
    print(f"=" * 60)
    print(f"DSM Prompter Training - SAM2 + DSM")
    print(f"=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {SAM2_CHECKPOINT}")
    print(f"Data: {DATA_ROOT}")
    print(f"=" * 60)
    
    # Vérification du checkpoint
    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint SAM2 introuvable: {SAM2_CHECKPOINT}")
    
    # 1. Datasets & Loaders
    print("Chargement des datasets...")
    train_dataset = PlantationsDataset(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_DSM_DIR)
    val_dataset = PlantationsDataset(VAL_JSON, VAL_IMG_DIR, VAL_DSM_DIR)
    
    print(f"  - Train: {len(train_dataset)} images")
    print(f"  - Val: {len(val_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=local_collate_fn, 
        num_workers=12, prefetch_factor=4,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=local_collate_fn, 
        num_workers=12, prefetch_factor=4,
        pin_memory=True, persistent_workers=True
    )

    # 2. Modèle SAM2 + DSM
    print("Initialisation du modèle SAM2+DSM...")
    model = DSMPrompterSAM2(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CHECKPOINT,
        num_proposals=5  # moyenne=3.7 arbres/image, 78.5% images <= 5 arbres
    )
    model.to(DEVICE)
    # Compiler le mask decoder pour accelerer le forward

    # 3. Optimizer & Scheduler (Appendix C du papier)
    # "AdamW with linear warmup followed by cosine annealing"
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch de warmup
    
    # "Warmup linéaire (start 10^-8) pendant 1 epoch"
    # start_factor = 1e-8 / 1e-5 = 0.001
    scheduler1 = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
    # "suivi d'un cosine annealing"
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    # 4. Loss
    criterion = DSMLoss(HungarianMatcher())

    # 5. Boucle d'entraînement
    print(f"Démarrage de l'entraînement: {EPOCHS} epochs, {len(train_loader)} batches/epoch")
    
    best_val_loss = float('inf')
    PATIENCE = 10
    epochs_no_improve = 0
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            rgb = batch["image"].to(DEVICE)
            dsm = batch["dsm"].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
            
            # Charger embeddings pré-calculés si disponibles
            precomputed_emb = None
            if USE_PRECOMPUTED and "img_id" in batch:
                embs = []
                for img_id in batch["img_id"]:
                    p = os.path.join(SAM2_EMBED_DIR, "train", f"{img_id}.pt")
                    embs.append(torch.load(p, map_location="cpu") if os.path.exists(p) else None)
                if all(e is not None for e in embs):
                    precomputed_emb = torch.stack(embs)
            
            # Mixed precision pour la vitesse
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                masks, ious, boxes, scores = model(rgb, dsm, precomputed_emb=precomputed_emb)
                outputs = {"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}
                loss = criterion(outputs, targets)
            
            # Normaliser la loss pour l'accumulation
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(pbar):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        # Validation périodique
        val_loss = None
        if (epoch + 1) % VAL_EVERY == 0:
            val_loss = validate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pth"))
                print(f"  -> Nouveau meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += VAL_EVERY
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping à l'epoch {epoch+1} (pas d'amélioration depuis {PATIENCE} epochs)")
                    break
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

        # Log to CSV
        import csv as _csv
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
            "learning_rate": f"{current_lr:.2e}",
        }
        with open(TRAINING_LOG_CSV, "a", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        # Sauvegarde régulière
        ckpt_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Config snapshot alongside every checkpoint
        _save_config_snapshot({
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "val_every": VAL_EVERY,
            "num_proposals": 5,
            "device": str(DEVICE),
            "sam2_config": SAM2_CONFIG,
            "sam2_checkpoint": SAM2_CHECKPOINT,
            "data_root": DATA_ROOT,
            "output_dir": OUTPUT_DIR,
        }, ckpt_path + ".config.json")
    
    # Sauvegarde finale
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))
    print(f"\nEntraînement terminé! Modèles sauvegardés dans {OUTPUT_DIR}")


if __name__ == "__main__":
    import gc
    batch_configs = [
        (16, 2),  # batch=16, accum=2 -> effectif 32
        (8, 4),   # batch=8, accum=4 -> effectif 32
        (4, 8),   # batch=4, accum=8 -> effectif 32
    ]
    for batch_size, accum_steps in batch_configs:
        try:
            globals()['BATCH_SIZE'] = batch_size
            globals()['GRAD_ACCUM_STEPS'] = accum_steps
            print(f"\n{'='*60}")
            print(f"Tentative avec BATCH_SIZE={batch_size}, GRAD_ACCUM_STEPS={accum_steps}")
            print(f"{'='*60}")
            main()
            break  # Succès, on sort de la boucle
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] CUDA Out of Memory avec batch={batch_size}. On réessaie avec un batch plus petit...")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"\n[ERREUR] {e}")
            raise