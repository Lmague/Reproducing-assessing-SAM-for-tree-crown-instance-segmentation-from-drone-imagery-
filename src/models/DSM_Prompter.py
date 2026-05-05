"""
DSMPrompter - Parameter-Efficient Tuning avec SAM2

Selon l'Appendix C du papier (basé sur RSPrompter):
- L'encodeur d'image et le décodeur de masque de SAM sont GELÉS (frozen)
- Architecture: RSPrompter + DSM Encoder
    - Le DSM Encoder a la même architecture que le "dense prompt encoder" de SAM
    - Il est entraînable
- Fusion: L'embedding du DSM est ajouté (somme) à l'embedding de l'image
- Hyperparamètres:
    - Optimizer: AdamW (weight_decay=0.1)
    - Batch size: 2
    - Lr: Base 0.00001 (1e-5)
    - Warmup linéaire (start 10^-8) pendant 1 epoch, suivi d'un cosine annealing
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as coco_mask
from tqdm import tqdm

# --- IMPORT SAM2 ---
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

# ==============================================================================
# 1. CONFIGURATION & CHEMINS (Appendix C)
# ==============================================================================

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON,
    TRAIN_RGB as TRAIN_IMG_DIR, VAL_RGB as VAL_IMG_DIR,
    TRAIN_DSM as TRAIN_DSM_DIR, VAL_DSM as VAL_DSM_DIR,
    SAM2_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(BASE_DIR, "output_dsm_sam2")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- HYPERPARAMÈTRES DU PAPIER (Appendix C) - Adaptés pour A40 (48GB) ---
# "batch size 2" pour RSPrompter/SAM-based methods
# A40: Batch 2 devrait passer, mais si OOM réduire à 1
BATCH_SIZE   = 2  # Papier: 2, A40: 2 (ou 1 si OOM)
# "AdamW optimizer with learning rate 1e-5" (0.00001)
LR           = 1e-5  # Pas de changement car batch reste 2
# "weight decay 0.1"
WEIGHT_DECAY = 0.1
# "trained for a maximum of 100 epochs" 
# Note: Le papier mentionne early stopping vers 20 epochs pour RSPrompter
EPOCHS       = 100
# Validation tous les N epochs
VAL_EVERY    = 2
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# A40 Tip: Si OOM, réduire num_proposals dans le modèle (ex: 10 -> 5)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DATASET
# ==============================================================================
class PlantationsDataset(Dataset):
    def __init__(self, json_file, img_dir, dsm_dir):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        self.dsm_dir = dsm_dir
        # Normalisation ImageNet (standard pour SAM2)
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.pixel_std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations: self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        self.img_ids = list(self.images.keys())
        
        # Récupérer les dimensions originales pour le scaling
        self.orig_sizes = {}
        for img in self.coco['images']:
            self.orig_sizes[img['id']] = (img.get('height', 1024), img.get('width', 1024))

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        orig_h, orig_w = self.orig_sizes[img_id]
        
        # RGB
        rgb_path = os.path.join(self.img_dir, os.path.basename(img_info['file_name']))
        image = cv2.imread(rgb_path)
        if image is None: 
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            print(f"Warning: Image non trouvée: {rgb_path}")
        else: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (1024, 1024))  # deja 1024x1024
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std

        # DSM - gestion robuste des extensions
        base_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        dsm_path = None
        for ext in ['.tif', '.TIF', '.png', '.PNG']:
            candidate = os.path.join(self.dsm_dir, base_name + ext)
            if os.path.exists(candidate):
                dsm_path = candidate
                break

        if dsm_path and os.path.exists(dsm_path):
            dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
            if dsm is not None:
                dsm = np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)
                dsm = dsm.astype(np.float32)  # deja 1024x1024
                # Normalisation per-sample (Appendix C)
                max_val = np.max(dsm)
                if max_val > 0: dsm = dsm / max_val
            else:
                dsm = np.zeros((1024, 1024), dtype=np.float32)
        else: 
            dsm = np.zeros((1024, 1024), dtype=np.float32)
        dsm_tensor = torch.from_numpy(dsm).float().unsqueeze(0)

        # Targets avec scaling des annotations
        anns = self.annotations.get(img_id, [])
        boxes, masks = [], []
        
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
                    rle = coco_mask.frPyObjects(ann['segmentation'], orig_h, orig_w)
                    m = coco_mask.decode(rle)
                    if len(m.shape) == 3: m = m.sum(axis=2) > 0
                else: 
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

        return {"image": img_tensor, "dsm": dsm_tensor, "img_id": str(img_id), "targets": {"boxes": boxes, "masks": masks, "labels": labels}}

def collate_fn(batch):
    return {
        "image": torch.stack([b['image'] for b in batch]),
        "dsm": torch.stack([b['dsm'] for b in batch]),
        "targets": [b['targets'] for b in batch]
    }

# ==============================================================================
# 3. MODULES
# ==============================================================================
class DSMEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.LayerNorm([16, 512, 512]), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.LayerNorm([32, 256, 256]), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LayerNorm([64, 128, 128]), nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, 2, 1), nn.LayerNorm([embed_dim, 64, 64]), nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
    def forward(self, x): return self.net(x)

class PromptProposalHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv_box = nn.Conv2d(embed_dim, 4, 1)
        self.conv_score = nn.Conv2d(embed_dim, 1, 1)
    def forward(self, x): return self.conv_box(x), self.conv_score(x)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class, self.cost_bbox = cost_class, cost_bbox
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        B, N, _ = pred_logits.shape
        indices = []
        for b in range(B):
            tgt_boxes = targets[b]["boxes"]
            if tgt_boxes.shape[0] == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue
            prob = pred_logits[b].sigmoid().squeeze(-1)
            cost_class = -prob.unsqueeze(1)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)
            C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox).cpu()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return indices

class DSMLoss(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher, self.bce, self.l1 = matcher, nn.BCEWithLogitsLoss(), nn.L1Loss()
    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        indices = self.matcher(outputs, targets)
        indices = [(src.to(device), tgt.to(device)) for (src, tgt) in indices]
        batch_idx = torch.cat([torch.full_like(src, i, device=device) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        pred_logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(pred_logits)
        if batch_idx.numel() > 0: target_classes[batch_idx, src_idx, 0] = 1.0
        loss_class = self.bce(pred_logits, target_classes)

        loss_bbox, loss_mask = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        if batch_idx.numel() > 0:
            src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
            tgt_boxes = torch.cat([t["boxes"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)])
            loss_bbox = self.l1(src_boxes, tgt_boxes)
            src_masks = outputs["pred_masks"][batch_idx, src_idx]
            tgt_masks = torch.cat([t["masks"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)])
            if src_masks.shape[-2:] != tgt_masks.shape[-2:]:
                src_masks = F.interpolate(src_masks, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
            loss_mask = F.binary_cross_entropy_with_logits(src_masks.squeeze(1), tgt_masks.float())
        return 2.0*loss_class + 5.0*loss_bbox + 5.0*loss_mask

# ==============================================================================
# 4. MODÈLE PRINCIPAL (SAM2 + DSM)
# ==============================================================================
class DSMPrompterSAM2(nn.Module):
    def __init__(self, config_file, ckpt_path, num_proposals=10):
        super().__init__()
        print("Chargement de SAM2...")
        self.sam = build_sam2(config_file, ckpt_path, device="cpu", apply_postprocessing=False)
        for p in self.sam.parameters(): p.requires_grad = False
        self.dsm_encoder = DSMEncoder(embed_dim=256)
        self.prompt_head = PromptProposalHead(embed_dim=256)
        self.num_proposals = num_proposals

    def decode_boxes(self, box_offsets, scores):
        B, _, H, W = box_offsets.shape
        device = box_offsets.device
        stride = 1024.0 / H 
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        cx, cy = (x_grid * stride + stride/2).flatten(), (y_grid * stride + stride/2).flatten()
        offsets = box_offsets.permute(0, 2, 3, 1).reshape(B, -1, 4)
        flat_scores = scores.view(B, -1)
        scale = stride * 4.0
        dx, dy = offsets[..., 0]*scale, offsets[..., 1]*scale
        w, h = torch.exp(offsets[..., 2])*scale, torch.exp(offsets[..., 3])*scale
        x1, y1 = (cx.unsqueeze(0)+dx) - w/2, (cy.unsqueeze(0)+dy) - h/2
        x2, y2 = (cx.unsqueeze(0)+dx) + w/2, (cy.unsqueeze(0)+dy) + h/2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1024)
        k = min(self.num_proposals, flat_scores.shape[1])
        topk_vals, topk_idx = torch.topk(flat_scores, k=k, dim=1)
        batch_idx = torch.arange(B, device=device)[:, None]
        return boxes[batch_idx, topk_idx], flat_scores[batch_idx, topk_idx]

    def forward(self, rgb, dsm, precomputed_emb=None):
        with torch.no_grad():
            if precomputed_emb is not None:
                img_emb = precomputed_emb.to(rgb.device).float()
                high_res_features = []
            else:
                out_dict = self.sam.forward_image(rgb)
                img_emb, high_res_features = out_dict["vision_features"], out_dict["backbone_fpn"]

        dsm_emb = self.dsm_encoder(dsm)
        fused_emb = img_emb + dsm_emb
        deltas, logits = self.prompt_head(fused_emb)
        proposals, scores = self.decode_boxes(deltas, logits) 
        
        B, K, _ = proposals.shape
        flat_proposals = proposals.view(B * K, 1, 4)
        flat_fused_emb = fused_emb.repeat_interleave(K, dim=0)
        flat_high_res = [feat.repeat_interleave(K, dim=0) for feat in high_res_features]
        if len(flat_high_res) > 2: flat_high_res = flat_high_res[:2]

        sparse, dense = self.sam.sam_prompt_encoder(points=None, boxes=flat_proposals, masks=None)
        image_pe = self.sam.sam_prompt_encoder.get_dense_pe()
        
        low_res_masks, iou_preds, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=flat_fused_emb, image_pe=image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=flat_high_res
        )
        return low_res_masks.view(B, K, 1, 256, 256), iou_preds.view(B, K, 1), proposals, scores

# ==============================================================================
# 5. MAIN LOOP (COMPLETE ET CORRIGÉE)
# ==============================================================================

def validate(model, dataloader, criterion, device):
    """Évaluation sur le set de validation."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["image"].to(device)
            dsm = batch["dsm"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                masks, ious, boxes, scores = model(rgb, dsm)
                outputs = {"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    model.train()
    return total_loss / max(len(dataloader), 1)


def main():
    print("=" * 60)
    print("DSM Prompter Training - SAM2 + DSM")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {SAM2_CHECKPOINT}")
    print(f"Data: {DATA_ROOT}")
    print("=" * 60)
    
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"ERREUR: Checkpoint introuvable: {SAM2_CHECKPOINT}")
        return
    
    print(f"Chargement Data depuis : {DATA_ROOT}")
    train_dataset = PlantationsDataset(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_DSM_DIR)
    val_dataset = PlantationsDataset(VAL_JSON, VAL_IMG_DIR, VAL_DSM_DIR)
    
    if len(train_dataset) == 0:
        print("ERREUR: Le dataset train est vide ! Vérifie tes chemins JSON/Images.")
        return
    
    print(f"  - Train: {len(train_dataset)} images")
    print(f"  - Val: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print("Initialisation Modèle...")
    model = DSMPrompterSAM2(SAM2_CONFIG, SAM2_CHECKPOINT, num_proposals=10)
    model.to(DEVICE)
    
    # Optimizer & Scheduler (Appendix C)
    # "AdamW with weight_decay=0.1"
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch de warmup
    
    # "Warmup linéaire (start 10^-8) pendant 1 epoch"
    # start_factor = 1e-8 / 1e-5 = 0.001
    scheduler1 = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
    # "suivi d'un cosine annealing"
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    
    criterion = DSMLoss(HungarianMatcher())
    
    print(f"Démarrage entraînement: {EPOCHS} epochs, {len(train_loader)} batchs/epoch")
    
    best_val_loss = float('inf')
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            rgb = batch["image"].to(DEVICE)
            dsm = batch["dsm"].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                masks, ious, boxes, scores = model(rgb, dsm)
                outputs = {"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation périodique
        if (epoch + 1) % VAL_EVERY == 0 and len(val_loader) > 0:
            val_loss = validate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pth"))
                print(f"  -> Nouveau meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")
        
        # Sauvegarde régulière
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth"))
    
    # Sauvegarde finale
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))
    print(f"\nEntraînement terminé! Modèles sauvegardés dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()