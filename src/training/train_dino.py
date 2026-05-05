"""
DINOv3 + Mask R-CNN for tree crown instance segmentation.

Mask R-CNN with DINOv3 ViT-Large backbone pretrained on 493M satellite images.
Optimized for:
- 1024x1024 native tiles
- Tree crowns ~300px (median diameter 257px, bbox ~310x315)
- ~3-4 trees per tile
- 7 tree species classes
- ~18750 training images
- NVIDIA A40 (48GB VRAM)

Learning Rate Schedule (frozen backbone):
- Warmup: 0 → 1000 iters
- Phase 1: 1000 → 60000 iters (LR = 0.0002)
- Phase 2: 60000 → 85000 iters (LR = 0.00002, /10)
- Phase 3: 85000 → 100000 iters (LR = 0.000002, /100)
- Total: ~21 epochs, ~25-30h on A40

Pre-computed mode (--use-precomputed):
- ViT-L not loaded → only FPN + Mask R-CNN heads
- Features loaded from disk (1024×64×64 float16, pre-computed by Pre_Compute_DINOv3.py)
- Geometric augmentations (flips, rotations) applied to feature maps
- ~5-7x faster than standard mode

Usage:
  python src/training/train_dino.py                          # Run training
  python src/training/train_dino.py --resume                 # Resume training
  python src/training/train_dino.py --unfreeze-blocks 2    # Fine-tuning phase 2
  python src/training/train_dino.py --eval-only             # Evaluation only
  python src/training/train_dino.py --demo-image img.png   # Inference on one image
  python src/training/train_dino.py --use-precomputed \\
    --embed-dir src/misc/dinov3_embeddings                  # Pre-computed mode

Author: Lmague
Date: 2026
"""

import os
import copy
import math
import json
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from fvcore.common.config import CfgNode as CN

from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    DatasetCatalog, DatasetMapper,
    build_detection_train_loader, build_detection_test_loader,
    detection_utils as utils,
)
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

torch.backends.cudnn.benchmark = True

# ==============================================================================
# CHEMINS ET CONFIGURATION PAR DÉFAUT
# ==============================================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON, DINO_CHECKPOINT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "DINOv3_MaskRCNN_Trees")

NUM_CLASSES = 7
UNFREEZE_BLOCKS = 0


# ==============================================================================
# 1) FPN amélioré avec connexions latérales (style ViTDet)
# ==============================================================================
class ImprovedFeaturePyramid(nn.Module):
    """
    FPN avec connexions latérales pour mieux fusionner les échelles.
    Adapté pour DINOv3 ViT-Large (patch 16).

    Entrée: feature map à stride 16 (ViT-L patch16)
    Sortie: [P2, P3, P4, P5] avec strides [4, 8, 16, 32]
    """
    def __init__(self, in_channels:  int, out_channels: int = 256):
        super().__init__()

        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.up_p2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        self.conv_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        self.down_p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        self.out_convs = nn.ModuleDict({
            'p2': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            'p3': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            'p4': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            'p5': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        lat = self.lateral(x)

        p4 = self.conv_p4(lat)
        p3 = self.up_p3(p4)
        p2 = self.up_p2(p3)
        p5 = self.down_p5(p4)

        return {
            'p2': self.out_convs['p2'](p2),
            'p3': self.out_convs['p3'](p3),
            'p4': self.out_convs['p4'](p4),
            'p5': self.out_convs['p5'](p5),
        }


# ==============================================================================
# 2) Custom DatasetMapper pour features pré-calculées
# ==============================================================================
class DINOv3EmbeddingMapper:
    """
    DatasetMapper qui charge les features ViT pré-calculées depuis disque
    et applique les augmentations géométriques sur les feature maps.

    Les features sont (1024, 64, 64) float16 sauvegardées par Pre_Compute_DINOv3.py.
    Les augmentations couleur sont ignorées (ViT est robuste).
    """

    def __init__(self, cfg, is_train=True, embed_dir=""):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.embed_dir = embed_dir

        if is_train:
            self.augmentations = T.AugmentationList([
                T.RandomFlip(horizontal=True, vertical=False),
                T.RandomFlip(horizontal=False, vertical=True),
                T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice"),
            ])
        else:
            self.augmentations = T.AugmentationList([])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Read image (needed for annotation processing and ImageList sizes)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        # Store image tensor
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)).astype("float32")
        )

        # Load pre-computed ViT features
        image_id = str(dataset_dict["image_id"])
        feat_path = os.path.join(self.embed_dir, f"{image_id}.pt")
        features = torch.load(feat_path, map_location="cpu", weights_only=True).float()

        # Apply same geometric transforms to features
        for t in transforms.transforms:
            t_name = type(t).__name__
            if t_name == "HFlipTransform":
                features = features.flip(-1)
            elif t_name == "VFlipTransform":
                features = features.flip(-2)
            elif t_name == "RotationTransform":
                angle = int(round(t.angle))
                if angle == 90:
                    features = features.rot90(1, [-2, -1])
                elif angle == 180:
                    features = features.rot90(2, [-2, -1])
                elif angle == 270:
                    features = features.rot90(3, [-2, -1])

        dataset_dict["dino_features"] = features

        # Process annotations
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="polygon")
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


# ==============================================================================
# 3) Backbone DINOv3 Satellite + ImprovedFPN
# ==============================================================================
@BACKBONE_REGISTRY.register()
class DINOv3_Satellite_Backbone(Backbone):
    """
    Backbone DINOv3 ViT-Large Satellite avec FPN amélioré.

    Caractéristiques DINOv3 Satellite:
    - Architecture: ViT-Large (24 blocs, embed_dim=1024, 16 heads)
    - Patch size: 16
    - Pré-entraîné sur 493M images satellites

    Mode pré-calculé (USE_PRECOMPUTED=True):
    - ViT non chargé → économise ~1.2 GB VRAM
    - forward_from_precomputed() passe les features directement au FPN
    """
    def __init__(self, cfg, input_shape):
        super().__init__()
        print("🛰️  Initialisation DINOv3 Satellite Backbone...")

        dcfg = cfg.MODEL.DINO
        ckpt_path = dcfg.CHECKPOINT_PATH

        self.patch_size = int(dcfg.PATCH_SIZE)
        self.out_channels = int(dcfg.OUT_CHANNELS)
        self.freeze_backbone = bool(dcfg.FREEZE)
        self.unfreeze_last_n_blocks = int(dcfg.UNFREEZE_LAST_N_BLOCKS)
        self.use_precomputed = bool(dcfg.USE_PRECOMPUTED)

        if self.use_precomputed:
            # ===== Mode pré-calculé: pas de ViT, FPN + LayerNorm uniquement =====
            assert self.unfreeze_last_n_blocks == 0, \
                "--use-precomputed est incompatible avec --unfreeze-blocks > 0 (ViT non chargé)"
            self.embed_dim = 1024  # ViT-L/16
            self.dino = None
            self.norm = nn.LayerNorm(self.embed_dim)
            print("⚡ Mode pré-calculé: ViT non chargé, FPN + LayerNorm uniquement")
        else:
            # ===== Mode standard: charge ViT complet =====
            self.dino = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                img_size=224,
            )

            self.embed_dim = self.dino.num_features

            if ckpt_path and os.path.exists(ckpt_path):
                print(f"📦 Chargement du checkpoint: {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

                if 'model' in state:
                    state = state['model']
                elif 'state_dict' in state:
                    state = state['state_dict']

                clean = {}
                for k, v in state.items():
                    new_k = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
                    clean[new_k] = v

                missing, unexpected = self.dino.load_state_dict(clean, strict=False)

                print(f"✅ DINOv3 Satellite chargé!")
                print(f"   - Poids chargés: {len(clean) - len(missing)}/{len(clean)}")
                print(f"   - Missing keys: {len(missing)}")
                if 0 < len(missing) < 10:
                    print(f"   - Missing:  {missing}")
                print(f"   - Unexpected keys: {len(unexpected)}")
            else:
                print(f"⚠️  Checkpoint introuvable: {ckpt_path}")
                print("   Utilisation des poids ImageNet par défaut (non recommandé)")
                self.dino = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=True,
                    num_classes=0,
                    dynamic_img_size=True,
                )

            self.norm = nn.LayerNorm(self.embed_dim)
            self._setup_freeze()

        # FPN (toujours créé, toujours trainable)
        self.fpn = ImprovedFeaturePyramid(
            in_channels=self.embed_dim,
            out_channels=self.out_channels,
        )

        self._out_features = ["p2", "p3", "p4", "p5"]
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16, "p5":  32}
        self._out_feature_channels = {k: self.out_channels for k in self._out_features}

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Backbone:  {n_params/1e6:.1f}M params total, {n_trainable/1e6:.1f}M trainables")

    def _setup_freeze(self):
        """Configure le freeze/unfreeze du backbone."""
        if self.freeze_backbone and self.unfreeze_last_n_blocks == 0:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()
            print("🧊 DINOv3 entièrement gelé (features extraction)")

        elif self.unfreeze_last_n_blocks > 0:
            for p in self.dino.parameters():
                p.requires_grad = False

            n_blocks = len(self.dino.blocks)
            start_unfreeze = n_blocks - self.unfreeze_last_n_blocks

            unfrozen_tensors = 0
            for i, block in enumerate(self.dino.blocks):
                if i >= start_unfreeze:
                    for p in block.parameters():
                        p.requires_grad = True
                        unfrozen_tensors += 1

            if hasattr(self.dino, 'norm'):
                for p in self.dino.norm.parameters():
                    p.requires_grad = True
                    unfrozen_tensors += 1

            self.dino.eval()

            unfrozen_params = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
            print(f"🔓 DINOv3: blocs {start_unfreeze}-{n_blocks-1} dégelés")
            print(f"   - {unfrozen_tensors} tensors, {unfrozen_params/1e6:.1f}M params trainables dans backbone")

        else:
            for p in self.dino.parameters():
                p.requires_grad = True
            self.dino.train()
            print("🔥 DINOv3 en full finetune (attention: très coûteux! )")

    def forward(self, x):
        """Forward standard: image → ViT → LayerNorm → FPN."""
        if self.dino is None:
            raise RuntimeError(
                "forward() appelé en mode pré-calculé (ViT non chargé). "
                "Utilisez forward_from_precomputed() ou désactivez --use-precomputed."
            )

        B, _, H, W = x.shape

        h_pad = (self.patch_size - H % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - W % self.patch_size) % self.patch_size
        if h_pad > 0 or w_pad > 0:
            x = F.pad(x, (0, w_pad, 0, h_pad))

        Hp = x.shape[2] // self.patch_size
        Wp = x.shape[3] // self.patch_size

        if self.freeze_backbone and self.unfreeze_last_n_blocks == 0:
            with torch.no_grad():
                out = self.dino.forward_features(x)
        else:
            out = self.dino.forward_features(x)

        if out.ndim == 3:
            expected = Hp * Wp
            if out.shape[1] == expected + 1:
                out = out[:, 1:, :]
            elif out.shape[1] > expected:
                out = out[:, -expected:, :]

            out = self.norm(out)
            out = out.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        fp_outputs = self.fpn(out)

        results = {}
        for name in self._out_features:
            feat = fp_outputs[name]
            stride = self._out_feature_strides[name]
            target_h = int(math.ceil(H / stride))
            target_w = int(math.ceil(W / stride))

            if feat.shape[2] >= target_h and feat.shape[3] >= target_w:
                feat = feat[: , :, :target_h, :target_w]
            else:
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)

            results[name] = feat

        return results

    def forward_from_precomputed(self, vit_features):
        """
        Forward depuis features ViT pré-calculées.

        Args:
            vit_features: (B, 1024, 64, 64) — sortie ViT reshape, SANS LayerNorm.
                          Produit par Pre_Compute_DINOv3.py.

        Returns:
            dict {p2, p3, p4, p5} avec les strides habituels.
        """
        B = vit_features.shape[0]
        H = W = 1024  # taille image originale (fixe pour pré-calculé)

        # Appliquer LayerNorm trainable (même que dans forward() standard)
        # (B, 1024, 64, 64) → (B, 64, 64, 1024) → norm → (B, 1024, 64, 64)
        x = vit_features.permute(0, 2, 3, 1).contiguous()  # (B, 64, 64, 1024)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 1024, 64, 64)

        fp_outputs = self.fpn(x)

        results = {}
        for name in self._out_features:
            feat = fp_outputs[name]
            stride = self._out_feature_strides[name]
            target_h = int(math.ceil(H / stride))
            target_w = int(math.ceil(W / stride))

            if feat.shape[2] >= target_h and feat.shape[3] >= target_w:
                feat = feat[:, :, :target_h, :target_w]
            else:
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)

            results[name] = feat

        return results

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


# ==============================================================================
# 4) DINOv3_MaskRCNN — GeneralizedRCNN avec support pré-calculé
# ==============================================================================
@META_ARCH_REGISTRY.register()
class DINOv3_MaskRCNN(GeneralizedRCNN):
    """
    Wrapper autour de GeneralizedRCNN qui supporte les features DINOv3 pré-calculées.

    Quand 'dino_features' est présent dans batched_inputs, utilise
    backbone.forward_from_precomputed() au lieu de backbone(images.tensor).
    Sinon, tombe en arrière sur le forward standard (pour l'inférence finale).
    """

    def _get_features(self, batched_inputs, images):
        """Obtient les features FPN depuis features pré-calculées ou backbone standard."""
        if batched_inputs and "dino_features" in batched_inputs[0]:
            dino_features = torch.stack([
                x["dino_features"].to(self.device) for x in batched_inputs
            ])
            return self.backbone.forward_from_precomputed(dino_features)
        return self.backbone(images.tensor)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self._get_features(batched_inputs, images)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self._get_features(batched_inputs, images)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results


# ==============================================================================
# 5) Dataset registration
# ==============================================================================
def _guess_image_root_from_json(json_path, candidate_roots):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        coco = json.load(f)
    imgs = coco.get("images", [])
    if not imgs:
        return None
    sample = imgs[: min(20, len(imgs))]
    file_names = [im.get("file_name", "") for im in sample if "file_name" in im]

    for root in candidate_roots:
        ok = 0
        for fn in file_names:
            if os.path.exists(os.path.join(root, fn)):
                ok += 1
        if ok >= max(1, len(file_names) // 2):
            return root
    return None


def register_datasets():
    """Enregistre les datasets train/val/test."""
    candidate_roots = [
        DATA_ROOT,
        os.path.join(DATA_ROOT, "train"),
        os.path.join(DATA_ROOT, "train", "RGB"),
        os.path.join(DATA_ROOT, "val"),
        os.path.join(DATA_ROOT, "val", "RGB"),
        os.path.join(DATA_ROOT, "test"),
        os.path.join(DATA_ROOT, "test", "RGB"),
    ]

    train_root = _guess_image_root_from_json(TRAIN_JSON, candidate_roots) or DATA_ROOT
    val_root = _guess_image_root_from_json(VAL_JSON, candidate_roots) or DATA_ROOT
    test_root = _guess_image_root_from_json(TEST_JSON, candidate_roots) or DATA_ROOT

    print(f"📌 train image_root = {train_root}")
    print(f"📌 val   image_root = {val_root}")
    print(f"📌 test  image_root = {test_root}")

    if "trees_train" not in DatasetCatalog.list():
        register_coco_instances("trees_train", {}, TRAIN_JSON, train_root)
    if "trees_val" not in DatasetCatalog.list():
        register_coco_instances("trees_val", {}, VAL_JSON, val_root)
    if "trees_test" not in DatasetCatalog.list():
        register_coco_instances("trees_test", {}, TEST_JSON, test_root)

    d = DatasetCatalog.get("trees_train")[0]
    if not os.path.exists(d["file_name"]):
        print("⚠️  Exemple d'image introuvable:", d["file_name"])
        print("   -> Vérifie JSON file_name et/ou image_root.")
    else:
        print(f"✅ Dataset OK, exemple:  {d['file_name']}")


# ==============================================================================
# 6) Configuration optimisée
# ==============================================================================
def add_dino_config(cfg, ckpt_path:  str = ""):
    """Ajoute les paramètres DINOv3 à la config Detectron2."""
    cfg.MODEL.DINO = CN()
    cfg.MODEL.DINO.CHECKPOINT_PATH = ckpt_path
    cfg.MODEL.DINO.PATCH_SIZE = 16
    cfg.MODEL.DINO.OUT_CHANNELS = 256
    cfg.MODEL.DINO.FREEZE = True
    cfg.MODEL.DINO.UNFREEZE_LAST_N_BLOCKS = 0
    cfg.MODEL.DINO.USE_PRECOMPUTED = False
    cfg.MODEL.DINO.EMBED_DIR = ""


def build_config(num_classes: int, output_dir: str, dino_ckpt_path: str,
                 unfreeze_blocks: int = 0, use_precomputed: bool = False,
                 embed_dir: str = ""):
    """
    Configuration optimisée pour:
    - Tuiles 1024x1024 natives
    - Arbres de ~300px (diamètre médian 257px, bbox ~310x315)
    - ~3-4 arbres par tuile
    - 7 classes d'arbres
    - ~18750 images d'entraînement

    Learning Rate Schedule:
    - Backbone gelé: 100k iters (~21 époques), LR decay à 60k et 85k
    - Fine-tuning: 140k iters (~15 époques), LR decay à 90k et 120k
    - Pré-calculé: 60k iters (~13 époques batch 4), même LR decay
    """
    cfg = get_cfg()
    add_dino_config(cfg, dino_ckpt_path)

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Datasets
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST = ("trees_val",)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.1

    # ============================================================
    # INPUT
    # ============================================================
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    cfg.INPUT.MIN_SIZE_TRAIN = (896, 960, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = False

    # ============================================================
    # Backbone
    # ============================================================
    cfg.MODEL.BACKBONE.NAME = "DINOv3_Satellite_Backbone"
    cfg.MODEL.DINO.PATCH_SIZE = 16

    # ============================================================
    # Configuration selon mode freeze/unfreeze
    # ============================================================
    # Calcul:  18746 images
    # - Batch 4: 4687 iters/époque
    # - Batch 2: 9373 iters/époque

    if unfreeze_blocks > 0:
        # ===== Fine-tuning partiel du backbone =====
        cfg.MODEL.DINO.FREEZE = False
        cfg.MODEL.DINO.UNFREEZE_LAST_N_BLOCKS = unfreeze_blocks

        # Batch réduit (plus de VRAM pour gradients backbone)
        cfg.SOLVER.IMS_PER_BATCH = 2

        # LR plus bas pour fine-tuning
        cfg.SOLVER.BASE_LR = 5e-5

        # Schedule:  ~15 époques (9373 × 15 ≈ 140k)
        cfg.SOLVER.MAX_ITER = 140000
        cfg.SOLVER.STEPS = (90000, 120000)
        cfg.SOLVER.WARMUP_ITERS = 2000

    else:
        # ===== Backbone gelé (défaut) =====
        cfg.MODEL.DINO.FREEZE = True
        cfg.MODEL.DINO.UNFREEZE_LAST_N_BLOCKS = 0

        # Batch plus grand possible
        cfg.SOLVER.IMS_PER_BATCH = 2  # RTX3090: 24GB

        # LR pour FPN + têtes
        cfg.SOLVER.BASE_LR = 1e-4

        # Schedule: ~6 époques (9373 × 6 ≈ 56k → arrondi 60k) — réduit pour coût GPU
        cfg.SOLVER.MAX_ITER = 60000
        cfg.SOLVER.STEPS = (40000, 55000)
        cfg.SOLVER.WARMUP_ITERS = 1000

    # ============================================================
    # Mode pré-calculé (overrides)
    # ============================================================
    if use_precomputed:
        cfg.MODEL.DINO.USE_PRECOMPUTED = True
        cfg.MODEL.DINO.EMBED_DIR = os.path.abspath(embed_dir)
        cfg.MODEL.META_ARCHITECTURE = "DINOv3_MaskRCNN"

        # Taille fixe (features pré-calculées pour 1024×1024 uniquement)
        cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
        cfg.INPUT.MAX_SIZE_TRAIN = 1024

        # Batch plus grand (pas de ViT → beaucoup moins de VRAM)
        cfg.SOLVER.IMS_PER_BATCH = 4

        # Plus de workers (I/O .pt files)
        cfg.DATALOADER.NUM_WORKERS = 8

    # FPN features
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # ============================================================
    # ANCRES:  Adaptées aux arbres de ~300px
    # ============================================================
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.8, 1.0, 1.25]]

    # ============================================================
    # ROI Heads
    # ============================================================
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28

    # ============================================================
    # RPN
    # ============================================================
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]

    # ============================================================
    # Solver - Optimizer & LR Schedule
    # ============================================================
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.GAMMA = 0.1

    # Gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    # Checkpointing (~1 fois par époque)
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 5000  # plus fréquent car moins d iters

    # Output
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    return cfg


# ==============================================================================
# 7) Augmentations
# ==============================================================================
def build_train_augmentations(cfg):
    """Augmentations adaptées aux images aériennes."""
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            "choice",
        ),
    ]

    augs.append(T.RandomFlip(horizontal=True, vertical=False))
    augs.append(T.RandomFlip(horizontal=False, vertical=True))
    augs.append(T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice"))
    augs.append(T.RandomBrightness(0.8, 1.2))
    augs.append(T.RandomContrast(0.8, 1.2))
    augs.append(T.RandomSaturation(0.8, 1.2))

    return augs


# ==============================================================================
# 8) Early stopping hook
# ==============================================================================
class EarlyStoppingHook(HookBase):
    """Arrête l'entraînement si pas d'amélioration après patience évaluations."""

    def __init__(self, eval_period, patience=10, metric_name="segm/AP", mode="max"):
        self.eval_period = int(eval_period)
        self.patience = int(patience)
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.eval_period != 0:
            return
        try:
            v = self.trainer.storage.latest().get(self.metric_name, None)
            if v is None:
                return
            curr = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
            improved = (self.mode == "max" and curr > self.best_metric) or \
                       (self.mode == "min" and curr < self.best_metric)

            if improved:
                self.best_metric = curr
                self.counter = 0
                print(f"\n✅ Nouveau meilleur:  {self.metric_name}={curr:.4f}")
            else:
                self.counter += 1
                print(f"\n⏳ Pas d'amélioration ({self.counter}/{self.patience}).Meilleur={self.best_metric:.4f}")

            if self.counter >= self.patience:
                print("\n🛑 EARLY STOPPING - Arrêt de l'entraînement.")
                self.trainer.iter = self.trainer.max_iter - 1
        except Exception:
            pass


# ==============================================================================
# 9) LR Logging Hook
# ==============================================================================
class LRLoggingHook(HookBase):
    """Affiche le learning rate périodiquement."""

    def __init__(self, log_period=1000):
        self.log_period = log_period

    def after_step(self):
        if (self.trainer.iter + 1) % self.log_period == 0:
            lr = self.trainer.optimizer.param_groups[0]["lr"]
            print(f"LR LoggingHook iter {self.trainer.iter + 1}: LR = {lr:.2e}")


class TrainingLogCSVHook(HookBase):
    """
    Logs training metrics (loss, val metrics, LR) to a CSV file alongside checkpoints.

    The CSV file is saved at {output_dir}/training_log.csv with columns:
    epoch, train_loss, val_loss, val_AP50, val_AP75, val_segm_AP, learning_rate
    """

    def __init__(self, log_period_iters: int, output_dir: str):
        self.log_period = log_period_iters
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self._written_header = False

    def after_step(self):
        iter_num = self.trainer.iter + 1
        if iter_num % self.log_period != 0:
            return

        storage = self.trainer.storage
        lr = self.trainer.optimizer.param_groups[0]["lr"]

        latest = storage.latest()
        train_loss = None
        val_loss = None
        val_ap50 = None
        val_ap75 = None
        val_segm_ap = None

        if "loss" in latest:
            train_loss = latest["loss"][0]

        if "validation_loss" in latest:
            val_loss = latest["validation_loss"][0]
        elif "val_loss" in latest:
            val_loss = latest["val_loss"][0]

        if "segm/AP50" in latest:
            val_ap50 = latest["segm/AP50"][0]
        if "segm/AP75" in latest:
            val_ap75 = latest["segm/AP75"][0]
        if "segm/AP" in latest:
            val_segm_ap = latest["segm/AP"][0]

        row = {
            "iter": iter_num,
            "train_loss": train_loss if train_loss is not None else "",
            "val_loss": val_loss if val_loss is not None else "",
            "val_AP50": val_ap50 if val_ap50 is not None else "",
            "val_AP75": val_ap75 if val_ap75 is not None else "",
            "val_segm_AP": val_segm_ap if val_segm_ap is not None else "",
            "learning_rate": lr,
        }

        import csv
        mode = "a"
        if not self._written_header:
            mode = "w"
            self._written_header = True

        try:
            with open(self.csv_path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if mode == "w":
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass


def _save_config_snapshot(cfg, path: str) -> None:
    """
    Saves a snapshot of the full training configuration to a JSON file.
    """
    import json as _json

    def _sanitise(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_sanitise(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return _sanitise(vars(obj))
        return str(obj)

    snapshot = {
        "num_classes": int(cfg.MODEL.ROI_HEADS.NUM_CLASSES),
        "data_root": str(cfg.DATASETS.TRAIN),
        "output_dir": str(cfg.OUTPUT_DIR),
        "batch_size": int(cfg.SOLVER.IMS_PER_BATCH),
        "base_lr": float(cfg.SOLVER.BASE_LR),
        "max_iter": int(cfg.SOLVER.MAX_ITER),
        "steps": [int(s) for s in cfg.SOLVER.STEPS],
        "gamma": float(cfg.SOLVER.GAMMA),
        "warmup_iters": int(cfg.SOLVER.WARMUP_ITERS),
        "eval_period": int(cfg.TEST.EVAL_PERIOD),
        "dino_checkpoint": str(cfg.MODEL.DINO.CHECKPOINT_PATH),
        "dino_patch_size": int(cfg.MODEL.DINO.PATCH_SIZE),
        "dino_out_channels": int(cfg.MODEL.DINO.OUT_CHANNELS),
        "dino_freeze": bool(cfg.MODEL.DINO.FREEZE),
        "dino_unfreeze_blocks": int(cfg.MODEL.DINO.UNFREEZE_LAST_N_BLOCKS),
        "dino_use_precomputed": bool(cfg.MODEL.DINO.USE_PRECOMPUTED),
        "dino_embed_dir": str(cfg.MODEL.DINO.EMBED_DIR),
        "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
        "optimizer": str(cfg.SOLVER.OPTIMIZER),
        "input_min_size": list(cfg.INPUT.MIN_SIZE_TRAIN),
        "input_max_size": int(cfg.INPUT.MAX_SIZE_TRAIN),
        "pixel_mean": list(cfg.MODEL.PIXEL_MEAN),
        "pixel_std": list(cfg.MODEL.PIXEL_STD),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        _json.dump(snapshot, f, indent=2)


# ==============================================================================
# 10) Trainer personnalisé
# ==============================================================================
class TreeDetectionTrainer(DefaultTrainer):
    """Trainer avec augmentations spécialisées et hooks personnalisés."""

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.DINO.USE_PRECOMPUTED:
            embed_dir = os.path.join(cfg.MODEL.DINO.EMBED_DIR, "train")
            mapper = DINOv3EmbeddingMapper(cfg, is_train=True, embed_dir=embed_dir)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            augs = build_train_augmentations(cfg)
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.DINO.USE_PRECOMPUTED:
            embed_dir = os.path.join(cfg.MODEL.DINO.EMBED_DIR, "val")
            mapper = DINOv3EmbeddingMapper(cfg, is_train=False, embed_dir=embed_dir)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        ret = super().build_hooks()

        eval_period = int(self.cfg.TEST.EVAL_PERIOD)

        # Sauvegarde du meilleur modèle
        ret.append(
            hooks.BestCheckpointer(
                eval_period=eval_period,
                checkpointer=self.checkpointer,
                val_metric="segm/AP",
                mode="max",
                file_prefix="model_best",
            )
        )

        # Early stopping (patience augmentée pour long training)
        ret.append(
            EarlyStoppingHook(
                eval_period=eval_period,
                patience=10,
                metric_name="segm/AP",
                mode="max"
            )
        )

        # LR logging
        ret.append(LRLoggingHook(log_period=2000))

        # CSV training log
        ret.append(TrainingLogCSVHook(log_period_iters=eval_period, output_dir=self.cfg.OUTPUT_DIR))

        # Config snapshot (saved alongside every checkpoint)
        config_snapshot_path = os.path.join(self.cfg.OUTPUT_DIR, "config_snapshot.json")
        _save_config_snapshot(self.cfg, config_snapshot_path)

        return ret


# ==============================================================================
# 11) Fonctions utilitaires
# ==============================================================================
def train(cfg, resume:  bool = False):
    """Lance l'entraînement."""
    trainer = TreeDetectionTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()
    return trainer


def inference_demo(cfg, image_path: str, output_path: str = None):
    """Fait une inférence sur une image."""
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    import cv2

    cfg_clone = cfg.clone()
    best_model = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    if os.path.exists(best_model):
        cfg_clone.MODEL.WEIGHTS = best_model
    else:
        cfg_clone.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg_clone.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg_clone)

    img = cv2.imread(image_path)
    outputs = predictor(img)

    v = Visualizer(
        img[: , :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = out.get_image()[:, :, ::-1]

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"✅ Résultat sauvegardé:  {output_path}")

    return outputs, result


def print_lr_schedule(cfg):
    """Affiche le planning du learning rate."""
    base_lr = cfg.SOLVER.BASE_LR
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    warmup_factor = cfg.SOLVER.WARMUP_FACTOR
    steps = cfg.SOLVER.STEPS
    gamma = cfg.SOLVER.GAMMA
    max_iter = cfg.SOLVER.MAX_ITER
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    iters_per_epoch = 18746 // batch_size

    print("\n📈 Learning Rate Schedule:")
    print(f"   {'Phase':<20} {'Iters':<20} {'Époques':<15} {'LR':<15}")
    print("   " + "-" * 70)

    # Warmup
    print(f"   {'Warmup':<20} {'0 → ' + str(warmup_iters):<20} {'0 → ' + f'{warmup_iters/iters_per_epoch:.1f}':<15} {base_lr * warmup_factor:.2e} → {base_lr:.2e}")

    # Phases principales
    prev_step = warmup_iters
    current_lr = base_lr
    for i, step in enumerate(steps):
        epoch_start = prev_step / iters_per_epoch
        epoch_end = step / iters_per_epoch
        print(f"   {'Phase ' + str(i+1):<20} {str(prev_step) + ' → ' + str(step):<20} {f'{epoch_start:.1f} → {epoch_end:.1f}':<15} {current_lr:.2e}")
        current_lr *= gamma
        prev_step = step

    # Phase finale
    epoch_start = prev_step / iters_per_epoch
    epoch_end = max_iter / iters_per_epoch
    print(f"   {'Phase finale':<20} {str(prev_step) + ' → ' + str(max_iter):<20} {f'{epoch_start:.1f} → {epoch_end:.1f}':<15} {current_lr:.2e}")

    print(f"\n   Total: {max_iter: ,} iters ≈ {max_iter/iters_per_epoch:.1f} époques")


# ==============================================================================
# 12) Main
# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DINOv3_MaskRCNN_Trees - Détection d'arbres")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--dino-ckpt", type=str, default=DINO_CHECKPOINT)
    ap.add_argument("--unfreeze-blocks", type=int, default=UNFREEZE_BLOCKS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--demo-image", type=str, default="")
    ap.add_argument("--use-precomputed", action="store_true",
                    help="Utiliser les features ViT pré-calculées (Pre_Compute_DINOv3.py)")
    ap.add_argument("--embed-dir", type=str, default="",
                    help="Chemin vers le dossier dinov3_embeddings/ (contient train/ et val/)")
    args = ap.parse_args()

    # Validation
    if args.use_precomputed:
        if not args.embed_dir:
            # Default embed dir
            args.embed_dir = os.path.join(BASE_DIR, "../../src/misc/dinov3_embeddings")
        if not os.path.isdir(args.embed_dir):
            print(f"❌ ERREUR: --embed-dir introuvable: {args.embed_dir}")
            print("   Lancez d'abord Pre_Compute_DINOv3.py pour pré-calculer les features.")
            exit(1)
        train_embed = os.path.join(args.embed_dir, "train")
        val_embed = os.path.join(args.embed_dir, "val")
        if not os.path.isdir(train_embed) or not os.path.isdir(val_embed):
            print(f"❌ ERREUR: --embed-dir doit contenir train/ et val/")
            print(f"   Trouvé: {os.listdir(args.embed_dir)}")
            exit(1)
        if args.unfreeze_blocks > 0:
            print("❌ ERREUR: --use-precomputed est incompatible avec --unfreeze-blocks > 0")
            exit(1)

    # Enregistrer les datasets
    register_datasets()

    # Construire la config
    cfg = build_config(
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        dino_ckpt_path=args.dino_ckpt,
        unfreeze_blocks=args.unfreeze_blocks,
        use_precomputed=args.use_precomputed,
        embed_dir=args.embed_dir,
    )

    # Affichage configuration
    print("\n" + "=" * 75)
    print("🌳 DINOv3_MaskRCNN_Trees - Détection et classification d'arbres")
    print("=" * 75)
    if args.use_precomputed:
        print(f"   Mode          : ⚡ PRÉ-CALCULÉ (ViT non chargé, features depuis disque)")
        print(f"   Embed dir     : {args.embed_dir}")
        n_train = len([f for f in os.listdir(os.path.join(args.embed_dir, "train")) if f.endswith(".pt")])
        n_val = len([f for f in os.listdir(os.path.join(args.embed_dir, "val")) if f.endswith(".pt")])
        print(f"   Features      : {n_train} train, {n_val} val")
        print(f"   Meta arch     : DINOv3_MaskRCNN")
    else:
        print(f"   Backbone       : DINOv3 ViT-Large (patch 16) - 493M images satellites")
        print(f"   Checkpoint     : {os.path.basename(args.dino_ckpt)}")
        print(f"   Mode           : {'Fine-tuning (' + str(args.unfreeze_blocks) + ' blocs)' if args.unfreeze_blocks > 0 else 'Backbone gelé'}")
    print(f"   Classes        : {args.num_classes}")
    print(f"   Batch size     : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   Max iters      : {cfg.SOLVER.MAX_ITER: ,}")
    print(f"   Base LR        : {cfg.SOLVER.BASE_LR:.2e}")
    print(f"   LR steps       : {cfg.SOLVER.STEPS}")
    print(f"   Input size     : {cfg.INPUT.MIN_SIZE_TRAIN} → {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"   Anchors        : {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"   Data root      : {DATA_ROOT}")
    print(f"   Output         : {args.output_dir}")
    print("=" * 75)

    # Afficher le schedule LR
    print_lr_schedule(cfg)

    # Estimation temps
    if args.use_precomputed:
        est_time = "~6-8h (pre-computed)"
    elif cfg.SOLVER.IMS_PER_BATCH == 4:
        est_time = "~25-30h"
    else:
        est_time = "~40-50h"
    print(f"\n   ⏱️  Temps estimé sur RTX 3090: {est_time}")
    print("=" * 75 + "\n")

    if args.eval_only:
        from detectron2.evaluation import inference_on_dataset

        trainer = TreeDetectionTrainer(cfg)
        trainer.resume_or_load(resume=True)

        evaluator = COCOEvaluator("trees_val", output_dir=cfg.OUTPUT_DIR)
        val_loader = TreeDetectionTrainer.build_test_loader(cfg, "trees_val")
        results = inference_on_dataset(trainer.model, val_loader, evaluator)
        print(results)

    elif args.demo_image:
        outputs, result = inference_demo(cfg, args.demo_image,
                                         output_path=os.path.join(cfg.OUTPUT_DIR, "demo_result.png"))
        print(f"Détections: {len(outputs['instances'])}")

    else:
        mode_str = "pré-calculé" if args.use_precomputed else "standard"
        print(f"🚀 Démarrage de l'entraînement DINOv3_MaskRCNN_Trees (mode {mode_str})...")
        train(cfg, resume=args.resume)
        print("\n✅ Entraînement terminé!")
