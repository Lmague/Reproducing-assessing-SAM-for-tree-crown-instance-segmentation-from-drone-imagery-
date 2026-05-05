"""
STEP 2: Training RSPrompter on pre-computed SAM embeddings.

See NeurIPS 2025 paper, Section 3.2 (RSPrompter).
First run src/misc/Pre_Compute_SAM.py to generate embeddings.

Author: Lmague
Date: 2026
"""
import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as coco_mask
from tqdm import tqdm

from segment_anything import sam_model_registry

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON, SAM_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "sam_embeddings_vith")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_rsprompter_fast")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HYPERPARAMÈTRES ---
BATCH_SIZE = 16  # On peut augmenter massivement car on n'a plus le gros encodeur !
LR = 1e-4        # On peut tenter un peu plus haut, ou garder 1e-5
WEIGHT_DECAY = 0.1
EPOCHS = 100     # Maintenant, 100 epochs prendront ~1h30
VAL_EVERY = 5
DEVICE = "cuda"

# ==============================================================================
# DATASET (Charge les .pt au lieu des images)
# ==============================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, json_file, embed_dir):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.embed_dir = embed_dir
        
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        self.img_ids = list(self.images.keys())
        
        self.orig_sizes = {}
        for img in self.coco['images']:
            self.orig_sizes[img['id']] = (img.get('height', 1024), img.get('width', 1024))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        orig_h, orig_w = self.orig_sizes[img_id]
        
        # Chargement de l'embedding pré-calculé
        embed_path = os.path.join(self.embed_dir, f"{img_id}.pt")
        if not os.path.exists(embed_path):
            # Fallback (ne devrait pas arriver si step 1 ok)
            embedding = torch.zeros((256, 64, 64), dtype=torch.float32)
        else:
            embedding = torch.load(embed_path) # (256, 64, 64)

        # Targets (identique à avant)
        anns = self.annotations.get(img_id, [])
        boxes, masks = [], []
        
        scale_x = 1024.0 / orig_w
        scale_y = 1024.0 / orig_h
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rle = coco_mask.frPyObjects(ann['segmentation'], orig_h, orig_w)
                    m = coco_mask.decode(rle)
                    if len(m.shape) == 3: m = m.sum(axis=2) > 0
                else:
                    m = coco_mask.decode(ann['segmentation'])
                m = cv2.resize(m.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
                masks.append(m)
            else:
                masks.append(np.zeros((1024, 1024), dtype=np.uint8))

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.tensor(np.array(masks), dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, 1024, 1024), dtype=torch.float32)

        return {
            "embedding": embedding, # Notez le changement de clé
            "targets": {"boxes": boxes, "masks": masks}
        }

def collate_fn(batch):
    return {
        "embedding": torch.stack([b['embedding'] for b in batch]),
        "targets": [b['targets'] for b in batch]
    }

# ==============================================================================
# MODELS (PromptEncoder + Matcher + RSPrompterLight)
# ==============================================================================
# PromptEncoder et Matcher sont identiques au code précédent
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LayerNorm([embed_dim, 64, 64]), nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LayerNorm([embed_dim, 64, 64]), nn.GELU(),
        )
        self.box_head = nn.Conv2d(embed_dim, 4, 1)
        self.score_head = nn.Conv2d(embed_dim, 1, 1)
        
    def forward(self, img_embeddings):
        x = self.conv_layers(img_embeddings)
        return self.box_head(x), self.score_head(x)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B = pred_logits.shape[0]
        indices = []
        for b in range(B):
            tgt_boxes = targets[b]["boxes"]
            if tgt_boxes.shape[0] == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            prob = pred_logits[b].sigmoid().squeeze(-1)
            cost_class = -prob.unsqueeze(1)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)
            C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox).cpu()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return indices

class RSPrompterLoss(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        indices = self.matcher(outputs, targets)
        indices = [(src.to(device), tgt.to(device)) for (src, tgt) in indices]
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        pred_logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(pred_logits)
        if batch_idx.numel() > 0: target_classes[batch_idx, src_idx, 0] = 1.0
        loss_class = self.bce(pred_logits, target_classes)

        loss_bbox = torch.tensor(0.0, device=device)
        loss_mask = torch.tensor(0.0, device=device)
        
        if batch_idx.numel() > 0:
            src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
            tgt_boxes = torch.cat([t["boxes"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)])
            loss_bbox = self.l1(src_boxes, tgt_boxes)
            
            src_masks = outputs["pred_masks"][batch_idx, src_idx]
            tgt_masks = torch.cat([t["masks"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)])
            if src_masks.shape[-2:] != tgt_masks.shape[-2:]:
                src_masks = F.interpolate(src_masks, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
            loss_mask = F.binary_cross_entropy_with_logits(src_masks.squeeze(1), tgt_masks.float())
        
        return 2.0 * loss_class + 5.0 * loss_bbox + 5.0 * loss_mask

class RSPrompterFast(nn.Module):
    def __init__(self, sam_checkpoint, num_proposals=10):
        super().__init__()
        # On charge SAM juste pour le decoder (prompt encoder + mask decoder)
        # On SUPPRIME l'image encoder pour libérer de la mémoire
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.mask_decoder = sam.mask_decoder
        self.sam_prompt_encoder = sam.prompt_encoder
        del sam.image_encoder # Libère 600M params en VRAM !
        
        for p in self.parameters(): p.requires_grad = False # Geler SAM parts
        
        self.prompt_encoder_learnable = PromptEncoder(embed_dim=256)
        self.num_proposals = num_proposals
    
    def decode_boxes(self, box_offsets, scores):
        B, _, H, W = box_offsets.shape
        device = box_offsets.device
        stride = 1024.0 / H
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        cx, cy = (x_grid * stride + stride / 2).flatten(), (y_grid * stride + stride / 2).flatten()
        
        offsets = box_offsets.permute(0, 2, 3, 1).reshape(B, -1, 4)
        flat_scores = scores.view(B, -1)
        scale = stride * 4.0
        dx, dy, w, h = offsets[..., 0]*scale, offsets[..., 1]*scale, torch.exp(offsets[..., 2].clamp(-10, 10))*scale, torch.exp(offsets[..., 3].clamp(-10, 10))*scale
        
        boxes = torch.stack([(cx.unsqueeze(0)+dx)-w/2, (cy.unsqueeze(0)+dy)-h/2, (cx.unsqueeze(0)+dx)+w/2, (cy.unsqueeze(0)+dy)+h/2], dim=-1).clamp(0, 1024)
        k = min(self.num_proposals, flat_scores.shape[1])
        topk_vals, topk_idx = torch.topk(flat_scores, k=k, dim=1)
        batch_idx = torch.arange(B, device=device)[:, None]
        return boxes[batch_idx, topk_idx], flat_scores[batch_idx, topk_idx]

    def forward(self, embeddings):
        # embeddings: (B, 256, 64, 64) - Vient direct du disque !
        B = embeddings.shape[0]
        
        # 1. Générer prompts
        box_offsets, logits = self.prompt_encoder_learnable(embeddings)
        boxes, scores = self.decode_boxes(box_offsets, logits)
        
        # 2. Decoder
        mask_outputs, iou_outputs = [], []
        dense_pe = self.sam_prompt_encoder.get_dense_pe()
        
        for b in range(B):
            img_emb = embeddings[b:b+1]
            current_boxes = boxes[b].view(self.num_proposals, 1, 4)
            sparse_emb, dense_emb = self.sam_prompt_encoder(points=None, boxes=current_boxes, masks=None)
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=img_emb, image_pe=dense_pe, 
                sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, 
                multimask_output=False
            )
            mask_outputs.append(low_res_masks)
            iou_outputs.append(iou_predictions)
            
        return torch.stack(mask_outputs), torch.stack(iou_outputs), boxes, scores

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(os.path.join(EMBED_DIR, "train")):
        print(f"ERREUR: Lancez l'étape 1 d'abord pour créer {EMBED_DIR}")
        return

    train_dataset = EmbeddingDataset(TRAIN_JSON, os.path.join(EMBED_DIR, "train"))
    val_dataset = EmbeddingDataset(VAL_JSON, os.path.join(EMBED_DIR, "val"))
    
    # Batch size bien plus grand possible ici !
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = RSPrompterFast(SAM_CHECKPOINT, num_proposals=10).to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))
    criterion = RSPrompterLoss(HungarianMatcher())
    
    PATIENCE = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Training FAST: {EPOCHS} epochs max, {len(train_loader)} batches/epoch (early stopping patience={PATIENCE})")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            emb = batch["embedding"].to(DEVICE).float()
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
            
            optimizer.zero_grad()
            masks, ious, boxes, scores = model(emb)
            loss = criterion({"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.2f}"})
            
        # Validation + early stopping
        if (epoch+1) % VAL_EVERY == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    emb = batch["embedding"].to(DEVICE).float()
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
                    masks, ious, boxes, scores = model(emb)
                    loss = criterion({"pred_logits": scores.unsqueeze(-1), "pred_boxes": boxes, "pred_masks": masks}, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Ep {epoch+1} — train_loss={total_loss/len(train_loader):.2f}  val_loss={val_loss:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pth"))
                print(f"  -> Nouveau meilleur modèle sauvegardé (val_loss={val_loss:.2f})")
            else:
                epochs_no_improve += VAL_EVERY
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last.pth"))
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping à l'epoch {epoch+1} (pas d'amélioration depuis {PATIENCE} epochs)")
                    break

if __name__ == "__main__":
    main()