"""
train_dsm_prompter_v2.py — Entraînement léger DSMPrompter (sans décodeur SAM2)

Architecture:
  - DSMEncoder (entraînable) + PromptProposalHead (entraînable)
  - SAM2 NON chargé pendant l'entraînement
  - Embeddings vision_features pré-calculés depuis le disque (~58 GB)
  - Loss: BCE (classification) + L1 (boîtes) — pas de loss masque

Vitesse: ~2-4 min/epoch (vs ~21 min avec le décodeur SAM2)

Pour l'inférence: utiliser DSMPrompterInference dans DSM_Prompter_v2.py
"""

import os
import sys
import csv
import gc
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.models.DSM_Prompter_v2 import (
    PlantationsDatasetBoxOnly,
    collate_fn_boxonly,
    DSMPrompterTraining,
    DSMBoxLoss,
    HungarianMatcher,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_DSM as TRAIN_DSM_DIR, VAL_DSM as VAL_DSM_DIR,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAM2_EMBED_DIR = os.path.join(BASE_DIR, "../misc/sam2_embeddings")
TRAIN_EMBED_DIR = os.path.join(SAM2_EMBED_DIR, "train")
VAL_EMBED_DIR   = os.path.join(SAM2_EMBED_DIR, "val")

OUTPUT_DIR = os.path.join(BASE_DIR, "output_dsm_prompter_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparamètres
BATCH_SIZE   = 32    # Batch effectif — ajusté par OOM recovery
LR           = 1e-5
WEIGHT_DECAY = 0.1
EPOCHS       = 100
VAL_EVERY    = 2
PATIENCE     = 10    # Early stopping
NUM_PROPOSALS = 5    # Moyenne=3.7 arbres/image, 78.5% ≤ 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# VALIDATE
# ==============================================================================

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            dsm = batch["dsm"].to(device)
            emb = batch["embedding"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]

            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                proposals, scores = model(dsm, emb)
                outputs = {
                    "pred_logits": scores.unsqueeze(-1),
                    "pred_boxes":  proposals,
                }
                loss = criterion(outputs, targets)

            total_loss += loss.item()

    model.train()
    return total_loss / max(len(dataloader), 1)


# ==============================================================================
# MAIN
# ==============================================================================

def main(batch_size=32, grad_accum=1):
    print("=" * 60)
    print("DSMPrompter v2 — Entraînement box-only (sans SAM2)")
    print("=" * 60)
    print(f"Device:        {DEVICE}")
    print(f"Batch size:    {batch_size}  (accum={grad_accum}, effectif={batch_size * grad_accum})")
    print(f"Embeddings:    {SAM2_EMBED_DIR}")
    print(f"Output:        {OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.isdir(TRAIN_EMBED_DIR):
        raise RuntimeError(f"Embeddings introuvables: {TRAIN_EMBED_DIR}\n"
                           "Lance d'abord Pre_Compute_SAM2.py")

    # --- Datasets ---
    print("Chargement des datasets...")
    train_dataset = PlantationsDatasetBoxOnly(TRAIN_JSON, TRAIN_DSM_DIR, TRAIN_EMBED_DIR)
    val_dataset   = PlantationsDatasetBoxOnly(VAL_JSON,   VAL_DSM_DIR,   VAL_EMBED_DIR)
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_boxonly,
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_boxonly,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Modèle ---
    print("Initialisation du modèle (DSMEncoder + PromptProposalHead)...")
    model = DSMPrompterTraining(num_proposals=NUM_PROPOSALS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables: {total_params / 1e6:.1f}M")

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch de warmup
    scheduler1 = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler  = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2],
                               milestones=[warmup_steps])

    # --- Loss ---
    criterion = DSMBoxLoss(HungarianMatcher())

    # --- CSV log ---
    log_csv = os.path.join(OUTPUT_DIR, "training_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss", "learning_rate"]

    best_val_loss = float("inf")
    epochs_no_improve = 0
    model.train()

    print(f"\nDémarrage: {EPOCHS} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            dsm     = batch["dsm"].to(DEVICE)
            emb     = batch["embedding"].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]

            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                proposals, scores = model(dsm, emb)
                outputs = {
                    "pred_logits": scores.unsqueeze(-1),
                    "pred_boxes":  proposals,
                }
                loss = criterion(outputs, targets)

            (loss / grad_accum).backward()
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if (step + 1) % grad_accum == 0 or (step + 1) == len(pbar):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        current_lr     = scheduler.get_last_lr()[0]

        # --- Validation ---
        val_loss = None
        if (epoch + 1) % VAL_EVERY == 0:
            val_loss = validate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1}/{EPOCHS} — Train: {avg_train_loss:.4f}  Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pth"))
                print(f"  -> Nouveau meilleur modèle (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += VAL_EVERY
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping à l'epoch {epoch+1} "
                          f"(pas d'amélioration depuis {PATIENCE} epochs)")
                    break
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} — Train: {avg_train_loss:.4f}")

        # CSV
        row = {
            "epoch":         epoch + 1,
            "train_loss":    f"{avg_train_loss:.6f}",
            "val_loss":      f"{val_loss:.6f}" if val_loss is not None else "",
            "learning_rate": f"{current_lr:.2e}",
        }
        write_header = not os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Checkpoint par epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))
    print(f"\nEntraînement terminé! Modèles dans {OUTPUT_DIR}")


# ==============================================================================
# ENTRY POINT avec OOM recovery
# ==============================================================================

if __name__ == "__main__":
    batch_configs = [
        (32, 1),  # 32 direct
        (16, 2),  # batch=16, accum=2 → effectif 32
        (8,  4),  # batch=8,  accum=4 → effectif 32
    ]
    for batch_size, accum_steps in batch_configs:
        try:
            print(f"\n{'='*60}")
            print(f"Tentative: BATCH_SIZE={batch_size}, GRAD_ACCUM={accum_steps}")
            print(f"{'='*60}")
            main(batch_size=batch_size, grad_accum=accum_steps)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] batch={batch_size} → trop grand. Réduction...")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[ERREUR] {e}")
            raise
