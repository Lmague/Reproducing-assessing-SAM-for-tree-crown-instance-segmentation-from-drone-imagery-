"""
DSMPrompter v2 — Entraînement léger sans décodeur SAM2

Approche inspirée de RSPrompter:
  - Pre_Compute_SAM2 (déjà fait): embeddings vision_features (256x64x64) sur disque
  - Training: DSMEncoder + PromptProposalHead seulement (pas de SAM2 pendant l'entraînement)
  - Loss: BCE (classification) + L1 (boîtes) — pas de loss masque
  - Inference: charger les poids entraînés + brancher le décodeur SAM2

Vitesse estimée: ~2-4 min/epoch vs ~21 min/epoch avec le décodeur SAM2.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as coco_mask

# ==============================================================================
# 1. DATASET — DSM + Embeddings pré-calculés + Boxes uniquement (pas de masques)
# ==============================================================================

class PlantationsDatasetBoxOnly(Dataset):
    """
    Dataset allégé pour l'entraînement box-only.
    Charge: DSM (1x1024x1024) + embedding SAM2 pré-calculé (256x64x64) + boxes GT.
    Ne charge PAS: RGB, masques (inutiles pour la loss box-only).
    """
    def __init__(self, json_file, dsm_dir, embed_dir):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.dsm_dir = dsm_dir
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
        img_info = self.images[img_id]
        orig_h, orig_w = self.orig_sizes[img_id]

        # --- DSM ---
        base_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        dsm_path = None
        for ext in ['.tif', '.TIF', '.png', '.PNG']:
            candidate = os.path.join(self.dsm_dir, base_name + ext)
            if os.path.exists(candidate):
                dsm_path = candidate
                break

        if dsm_path is not None:
            dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
            if dsm is None:
                dsm = np.zeros((1024, 1024), dtype=np.float32)
            else:
                dsm = np.nan_to_num(dsm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                max_val = np.max(dsm)
                if max_val > 0:
                    dsm = dsm / max_val
        else:
            dsm = np.zeros((1024, 1024), dtype=np.float32)

        dsm_tensor = torch.from_numpy(dsm).float().unsqueeze(0)  # (1, 1024, 1024)

        # --- Embedding SAM2 pré-calculé ---
        embed_path = os.path.join(self.embed_dir, f"{img_id}.pt")
        if os.path.exists(embed_path):
            embedding = torch.load(embed_path, map_location="cpu")  # (256, 64, 64)
        else:
            embedding = torch.zeros(256, 64, 64, dtype=torch.float16)

        # --- Boxes GT uniquement (pas de masques) ---
        anns = self.annotations.get(img_id, [])
        boxes = []
        scale_x = 1024.0 / orig_w
        scale_y = 1024.0 / orig_h

        for ann in anns:
            x, y, w, h = ann['bbox']
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "dsm": dsm_tensor,
            "embedding": embedding,
            "targets": {"boxes": boxes, "labels": labels},
        }


def collate_fn_boxonly(batch):
    dsms = torch.stack([b['dsm'] for b in batch])
    embeddings = torch.stack([b['embedding'] for b in batch])
    targets = [b['targets'] for b in batch]
    return {"dsm": dsms, "embedding": embeddings, "targets": targets}


# ==============================================================================
# 2. MODULES (identiques à v1)
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
    def forward(self, x):
        return self.net(x)


class PromptProposalHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv_box = nn.Conv2d(embed_dim, 4, 1)
        self.conv_score = nn.Conv2d(embed_dim, 1, 1)
    def forward(self, x):
        return self.conv_box(x), self.conv_score(x)


# ==============================================================================
# 3. HUNGARIAN MATCHER (boîtes uniquement)
# ==============================================================================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]  # (B, N, 1)
        pred_boxes = outputs["pred_boxes"]    # (B, N, 4)
        B = pred_logits.shape[0]
        indices = []
        for b in range(B):
            tgt_boxes = targets[b]["boxes"]
            if tgt_boxes.shape[0] == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                torch.tensor([], dtype=torch.int64)))
                continue
            prob = pred_logits[b].sigmoid().squeeze(-1)  # (N,)
            cost_class = -prob.unsqueeze(1)               # (N, T)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # (N, T)
            C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox).cpu()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.int64),
                            torch.as_tensor(col, dtype=torch.int64)))
        return indices


# ==============================================================================
# 4. LOSS — Box + Classification uniquement (pas de masque, pas de décodeur SAM2)
# ==============================================================================

class DSMBoxLoss(nn.Module):
    """
    Loss légère: BCE (classification) + L1 (régression de boîtes).
    Pas de mask loss → pas besoin du décodeur SAM2 pendant l'entraînement.
    """
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        indices = self.matcher(outputs, targets)
        indices = [(src.to(device), tgt.to(device)) for (src, tgt) in indices]

        batch_idx = torch.cat([
            torch.full_like(src, i, device=device)
            for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])

        # Classification loss
        pred_logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(pred_logits)
        if batch_idx.numel() > 0:
            target_classes[batch_idx, src_idx, 0] = 1.0
        loss_class = self.bce(pred_logits, target_classes)

        # Box regression loss
        loss_bbox = torch.tensor(0.0, device=device)
        if batch_idx.numel() > 0:
            src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
            tgt_boxes = torch.cat([
                t["boxes"][tgt_idx]
                for t, (_, tgt_idx) in zip(targets, indices)
            ])
            loss_bbox = self.l1(src_boxes, tgt_boxes)

        return 2.0 * loss_class + 5.0 * loss_bbox


# ==============================================================================
# 5. MODÈLE D'ENTRAÎNEMENT (sans SAM2)
# ==============================================================================

class DSMPrompterTraining(nn.Module):
    """
    Modèle léger pour l'entraînement: DSMEncoder + PromptProposalHead uniquement.
    SAM2 n'est PAS chargé — on utilise les embeddings pré-calculés depuis le disque.

    À l'inférence, charger ces poids dans DSMPrompterInference qui branche SAM2.
    """
    def __init__(self, num_proposals=5):
        super().__init__()
        self.dsm_encoder = DSMEncoder(embed_dim=256)
        self.prompt_head = PromptProposalHead(embed_dim=256)
        self.num_proposals = num_proposals

    def decode_boxes(self, box_offsets, scores):
        B, _, H, W = box_offsets.shape
        device = box_offsets.device
        stride = 1024.0 / H
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        cx = (x_grid * stride + stride / 2).flatten()
        cy = (y_grid * stride + stride / 2).flatten()
        offsets = box_offsets.permute(0, 2, 3, 1).reshape(B, -1, 4)
        flat_scores = scores.view(B, -1)
        scale = stride * 4.0
        dx, dy = offsets[..., 0] * scale, offsets[..., 1] * scale
        w = torch.exp(offsets[..., 2].clamp(-4, 4)) * scale
        h = torch.exp(offsets[..., 3].clamp(-4, 4)) * scale
        x1 = (cx.unsqueeze(0) + dx) - w / 2
        y1 = (cy.unsqueeze(0) + dy) - h / 2
        x2 = (cx.unsqueeze(0) + dx) + w / 2
        y2 = (cy.unsqueeze(0) + dy) + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1024)
        k = min(self.num_proposals, flat_scores.shape[1])
        topk_vals, topk_idx = torch.topk(flat_scores, k=k, dim=1)
        batch_idx = torch.arange(B, device=device)[:, None]
        return boxes[batch_idx, topk_idx], flat_scores[batch_idx, topk_idx]

    def forward(self, dsm, precomputed_emb):
        """
        Args:
            dsm: (B, 1, 1024, 1024) float32
            precomputed_emb: (B, 256, 64, 64) float16 ou float32

        Returns:
            proposals: (B, K, 4) boîtes XYXY
            scores: (B, K) logits de confiance
        """
        dsm_emb = self.dsm_encoder(dsm)                    # (B, 256, 64, 64)
        fused_emb = precomputed_emb.float() + dsm_emb      # (B, 256, 64, 64)
        deltas, logits = self.prompt_head(fused_emb)        # (B,4,H,W), (B,1,H,W)
        proposals, scores = self.decode_boxes(deltas, logits)
        return proposals, scores


# ==============================================================================
# 6. MODÈLE D'INFÉRENCE (avec SAM2) — à utiliser APRÈS entraînement
# ==============================================================================

class DSMPrompterInference(nn.Module):
    """
    Modèle d'inférence complet: charge les poids entraînés (DSMEncoder + Head)
    puis branche le décodeur SAM2 pour produire des masques.

    Usage:
        model = DSMPrompterInference(SAM2_CONFIG, SAM2_CHECKPOINT)
        model.load_training_weights("model_best.pth")
        masks, scores = model(rgb, dsm)
    """
    def __init__(self, config_file, ckpt_path, num_proposals=5):
        super().__init__()
        from sam2.build_sam import build_sam2
        print("Chargement SAM2 pour inférence...")
        self.sam = build_sam2(config_file, ckpt_path, device="cpu", apply_postprocessing=False)
        for p in self.sam.parameters():
            p.requires_grad = False
        self.dsm_encoder = DSMEncoder(embed_dim=256)
        self.prompt_head = PromptProposalHead(embed_dim=256)
        self.num_proposals = num_proposals

    def load_training_weights(self, checkpoint_path):
        """Charge les poids entraînés depuis DSMPrompterTraining."""
        state = torch.load(checkpoint_path, map_location="cpu")
        self.dsm_encoder.load_state_dict(
            {k.replace("dsm_encoder.", ""): v
             for k, v in state.items() if k.startswith("dsm_encoder.")})
        self.prompt_head.load_state_dict(
            {k.replace("prompt_head.", ""): v
             for k, v in state.items() if k.startswith("prompt_head.")})
        print(f"Poids chargés depuis {checkpoint_path}")

    def decode_boxes(self, box_offsets, scores):
        B, _, H, W = box_offsets.shape
        device = box_offsets.device
        stride = 1024.0 / H
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        cx = (x_grid * stride + stride / 2).flatten()
        cy = (y_grid * stride + stride / 2).flatten()
        offsets = box_offsets.permute(0, 2, 3, 1).reshape(B, -1, 4)
        flat_scores = scores.view(B, -1)
        scale = stride * 4.0
        dx, dy = offsets[..., 0] * scale, offsets[..., 1] * scale
        w = torch.exp(offsets[..., 2].clamp(-4, 4)) * scale
        h = torch.exp(offsets[..., 3].clamp(-4, 4)) * scale
        x1 = (cx.unsqueeze(0) + dx) - w / 2
        y1 = (cy.unsqueeze(0) + dy) - h / 2
        x2 = (cx.unsqueeze(0) + dx) + w / 2
        y2 = (cy.unsqueeze(0) + dy) + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1024)
        k = min(self.num_proposals, flat_scores.shape[1])
        _, topk_idx = torch.topk(flat_scores, k=k, dim=1)
        batch_idx = torch.arange(B, device=device)[:, None]
        return boxes[batch_idx, topk_idx], flat_scores[batch_idx, topk_idx]

    def forward(self, rgb, dsm):
        """
        Inférence complète avec masques SAM2.
        Args:
            rgb: (B, 3, 1024, 1024)
            dsm: (B, 1, 1024, 1024)
        Returns:
            masks: (B, K, 1, 256, 256) masques SAM2 basse résolution
            iou_preds: (B, K, 1)
            proposals: (B, K, 4)
            scores: (B, K)
        """
        with torch.no_grad():
            out_dict = self.sam.forward_image(rgb)
            img_emb = out_dict["vision_features"]
            high_res_features = []  # backbone_fpn non utilisé (~189 GB)

        dsm_emb = self.dsm_encoder(dsm)
        fused_emb = img_emb + dsm_emb
        deltas, logits = self.prompt_head(fused_emb)
        proposals, scores = self.decode_boxes(deltas, logits)

        B, K, _ = proposals.shape
        flat_proposals = proposals.view(B * K, 1, 4)
        flat_fused_emb = fused_emb.repeat_interleave(K, dim=0)

        with torch.no_grad():
            sparse, dense = self.sam.sam_prompt_encoder(
                points=None, boxes=flat_proposals, masks=None)
            image_pe = self.sam.sam_prompt_encoder.get_dense_pe()
            low_res_masks, iou_preds, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=flat_fused_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=[]
            )

        return (low_res_masks.view(B, K, 1, 256, 256),
                iou_preds.view(B, K, 1),
                proposals,
                scores)
