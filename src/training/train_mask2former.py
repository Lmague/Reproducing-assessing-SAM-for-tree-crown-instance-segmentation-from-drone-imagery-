"""
Mask2Former + Swin-T — Instance segmentation de couronnes d'arbres

Architecture SOTA pour la segmentation d'instance (CVPR 2022).
Utilise un backbone Swin-Tiny avec un pixel decoder MSDeformAttn
et un transformer decoder avec masked cross-attention.

Avantages vs Mask R-CNN:
- Transformer decoder + queries apprises (pas d'anchors)
- Hungarian matching (prédiction ensembliste)
- Masked cross-attention (meilleure per-instance feature extraction)
- Gains typiques: +3-5 AP vs Mask R-CNN sur COCO

Specs:
- Backbone: Swin-Tiny (28M params)
- Pixel decoder: MSDeformAttn FPN (~10M params)
- Transformer decoder: 6 layers, 100 queries (~8M params)
- Total: ~47M params
- VRAM: ~18-22 GB en batch=2 avec AMP à 1024×1024
- Temps estimé: ~12-20h sur RTX 3090 (COCO pretrained + early stop)

Usage:
  python src/training/train_mask2former.py                    # Train
  python src/training/train_mask2former.py --resume           # Resume
  python src/training/train_mask2former.py --eval-only        # Eval only
  python src/training/train_mask2former.py --batch-size 1     # OOM fallback

Requires: Mask2Former repo cloned at /workspace/Mask2Former
          (setup_mask2former.sh handles this automatically)
"""

import os
import sys
import json
import copy
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

# ============================================================================
# PATH SETUP — Add Mask2Former repo to sys.path
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK2FORMER_DIR = "/workspace/Mask2Former"
if os.path.exists(MASK2FORMER_DIR):
    sys.path.insert(0, MASK2FORMER_DIR)

# Detectron2
from detectron2.engine import DefaultTrainer, hooks, HookBase, default_setup
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    MetadataCatalog, DatasetCatalog,
    DatasetMapper, build_detection_train_loader, build_detection_test_loader,
    detection_utils as utils,
)
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.solver.build import maybe_add_gradient_clipping
import itertools

# Mask2Former imports (registered via side-effect)
try:
    from mask2former import add_maskformer2_config
    from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
        COCOInstanceNewBaselineDatasetMapper,
        build_transform_gen,
    )
    MASK2FORMER_AVAILABLE = True
except ImportError:
    MASK2FORMER_AVAILABLE = False
    print("⚠️  Mask2Former non trouvé. Lancez setup_mask2former.sh d'abord.")

sys.path.insert(0, os.path.join(BASE_DIR, ".."))
from utils.coco_helpers import fix_coco_json_if_needed

# ============================================================================
# CONFIGURATION
# ============================================================================
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON,
)
OUTPUT_DIR = os.path.join(BASE_DIR, "Mask2Former_SwinT_Trees")

NUM_CLASSES = 7
N_TRAIN_IMAGES = 18746

# COCO pre-trained Mask2Former Swin-T checkpoint
COCO_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
    "maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
)
# Local path (downloaded by setup script)
COCO_CHECKPOINT_LOCAL = "/workspace/checkpoints/mask2former_swin_tiny_coco_instance.pkl"


# ============================================================================
# EARLY STOPPING
# ============================================================================
class EarlyStoppingHook(HookBase):
    def __init__(self, eval_period, patience=3, metric_name="segm/AP", mode="max"):
        self.eval_period = int(eval_period)
        self.patience = int(patience)
        self.metric_name = metric_name
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.eval_period != 0:
            return
        try:
            v = self.trainer.storage.latest().get(self.metric_name)
            if v is None:
                return
            curr = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
            improved = (self.mode == "max" and curr > self.best) or \
                       (self.mode == "min" and curr < self.best)
            if improved:
                self.best = curr
                self.counter = 0
                print(f"\n✅ Nouveau meilleur {self.metric_name}={curr:.4f}")
            else:
                self.counter += 1
                print(f"\n⏳ Pas d'amélioration ({self.counter}/{self.patience}) "
                      f"meilleur={self.best:.4f}")
            if self.counter >= self.patience:
                print("\n🛑 EARLY STOPPING — arrêt de l'entraînement")
                self.trainer.iter = self.trainer.max_iter - 1
        except Exception:
            pass


# ============================================================================
# CUSTOM TRAINER
# ============================================================================
class Mask2FormerTrainer(DefaultTrainer):
    """Trainer Mask2Former avec AMP, BestCheckpointer et early stopping."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        # LSJ augmentation: RandomFlip + ResizeScale + FixedSizeCrop
        tfm_gens = build_transform_gen(cfg, is_train=True)
        mapper = COCOInstanceNewBaselineDatasetMapper(
            is_train=True, tfm_gens=tfm_gens, image_format=cfg.INPUT.FORMAT
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # No augmentation for eval — use default detectron2 mapper
        mapper = DatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """AdamW with per-parameter weight decay (backbone lower LR)."""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d,
            nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm,
            nn.LocalResponseNorm,
        )

        params = []
        memo = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                # Backbone gets lower LR (0.1x)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER

                # Norm layers: separate weight decay
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                # Embedding layers: separate weight decay
                if "position_embedding" in module_param_name or \
                   "query_embed" in module_param_name or \
                   "query_feat" in module_param_name or \
                   "level_embed" in module_param_name:
                    hyperparams["weight_decay"] = weight_decay_embed

                params.append({"params": [value], **hyperparams})

        # Full-model gradient clipping (Mask2Former specific, not in detectron2 0.6)
        def maybe_add_full_model_gradient_clipping(optim_cls):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim_cls):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim_cls

        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, **defaults
        )
        # Standard clipping for non-full_model modes
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """WarmupMultiStepLR (same as detectron2 default)."""
        from detectron2.solver import build_lr_scheduler
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        ret = super().build_hooks()
        eval_period = int(self.cfg.TEST.EVAL_PERIOD)

        # BestCheckpointer
        ret.append(hooks.BestCheckpointer(
            eval_period=eval_period,
            checkpointer=self.checkpointer,
            val_metric="segm/AP",
            mode="max",
            file_prefix="model_best",
        ))

        # Early stopping: patience=3 évals × 5 epochs = 15 epochs sans amélioration
        ret.append(EarlyStoppingHook(
            eval_period=eval_period,
            patience=3,
            metric_name="segm/AP",
            mode="max",
        ))

        return ret


# ============================================================================
# CONFIG BUILDER
# ============================================================================
def build_config(batch_size=2, resume=False):
    """Build Mask2Former config for tree crown segmentation."""
    cfg = get_cfg()
    add_maskformer2_config(cfg)

    # ── Base Mask2Former Swin-T config ─────────────────────────────────────
    config_file = os.path.join(
        MASK2FORMER_DIR,
        "configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml"
    )
    if os.path.exists(config_file):
        # Allow new keys that Mask2Former base config adds (not in detectron2 0.6)
        cfg.MODEL.RESNETS.set_new_allowed(True)
        cfg.INPUT.set_new_allowed(True)
        cfg.merge_from_file(config_file)
    else:
        print(f"⚠️  Config file non trouvé: {config_file}")
        print("   Configuration manuelle des paramètres Mask2Former...")
        _setup_mask2former_config_manual(cfg)

    # ── Dataset ────────────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST = ("trees_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # ── Input ──────────────────────────────────────────────────────────────
    # LSJ augmentation: scale jittering + crop to IMAGE_SIZE
    # Conservative scales for 1024×1024 tree tiles (trees ~300px)
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.5    # Min 512px before crop (trees still ~150px)
    cfg.INPUT.MAX_SCALE = 1.5    # Max 1536px before crop
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_instance_lsj"

    # ── Model ──────────────────────────────────────────────────────────────
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES

    # Use local checkpoint if downloaded, otherwise URL (auto-download)
    if os.path.exists(COCO_CHECKPOINT_LOCAL):
        cfg.MODEL.WEIGHTS = COCO_CHECKPOINT_LOCAL
    else:
        cfg.MODEL.WEIGHTS = COCO_CHECKPOINT_URL

    # ── Solver ─────────────────────────────────────────────────────────────
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # LR: scale linearly from base config (base: batch=16, lr=1e-4)
    base_batch = 16
    base_lr = 1e-4
    cfg.SOLVER.BASE_LR = base_lr * (batch_size / base_batch)

    iters_per_epoch = N_TRAIN_IMAGES // batch_size
    max_epochs = 50
    max_iter = iters_per_epoch * max_epochs

    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(1000, iters_per_epoch)
    cfg.SOLVER.WARMUP_FACTOR = 0.01

    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    # AMP
    cfg.SOLVER.AMP.ENABLED = True

    # Checkpoint & eval
    eval_period = max(500, iters_per_epoch * 5)  # Every 5 epochs
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period
    cfg.TEST.EVAL_PERIOD = eval_period

    # ── Output ─────────────────────────────────────────────────────────────
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def _setup_mask2former_config_manual(cfg):
    """Fallback: manually set Mask2Former Swin-T parameters."""
    # Backbone: Swin-T
    cfg.MODEL.BACKBONE.NAME = "D2SwinTransformer"
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # Pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"

    # Transformer decoder
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8

    # Loss weights
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1

    # Meta architecture
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"


# ============================================================================
# DATASET REGISTRATION
# ============================================================================
def register_datasets():
    for json_path in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if os.path.exists(json_path):
            fix_coco_json_if_needed(json_path)
    try:
        register_coco_instances("trees_train", {}, TRAIN_JSON, DATA_ROOT)
        register_coco_instances("trees_val", {}, VAL_JSON, DATA_ROOT)
        register_coco_instances("trees_test", {}, TEST_JSON, DATA_ROOT)
    except AssertionError:
        pass  # Already registered


# ============================================================================
# TRAINING
# ============================================================================
def train(batch_size=2, resume=False):
    assert MASK2FORMER_AVAILABLE, (
        "Mask2Former non installé. Lancez setup_mask2former.sh d'abord."
    )

    print("=" * 65)
    print("  Mask2Former Swin-T — Instance Segmentation d'Arbres")
    print("=" * 65)

    register_datasets()
    cfg = build_config(batch_size=batch_size, resume=resume)

    iters_per_epoch = N_TRAIN_IMAGES // batch_size
    eval_period = cfg.TEST.EVAL_PERIOD

    print(f"  Backbone        : Swin-Tiny (28M params)")
    print(f"  Total params    : ~47M")
    print(f"  Batch size      : {batch_size}")
    print(f"  LR              : {cfg.SOLVER.BASE_LR:.2e} (backbone: {cfg.SOLVER.BASE_LR * 0.1:.2e})")
    print(f"  Optimizer       : AdamW (wd={cfg.SOLVER.WEIGHT_DECAY})")
    print(f"  Max iter        : {cfg.SOLVER.MAX_ITER:,} (~{cfg.SOLVER.MAX_ITER / iters_per_epoch:.0f} epochs)")
    print(f"  LR steps        : {cfg.SOLVER.STEPS}")
    print(f"  Eval period     : {eval_period} iters (~{eval_period / iters_per_epoch:.0f} epochs)")
    print(f"  Early stop      : patience 3 évals ({3 * eval_period / iters_per_epoch:.0f} epochs)")
    print(f"  AMP             : {'✅ ACTIVÉ' if cfg.SOLVER.AMP.ENABLED else '❌'}")
    print(f"  Weights         : COCO pretrained Swin-T")
    print(f"  Output          : {cfg.OUTPUT_DIR}")
    print("=" * 65)

    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 65)
    print("  Évaluation finale sur validation")
    print("=" * 65)
    evaluator = COCOEvaluator("trees_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = Mask2FormerTrainer.build_test_loader(cfg, "trees_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

    return results


def evaluate_only(batch_size=2):
    """Evaluate best checkpoint."""
    assert MASK2FORMER_AVAILABLE, "Mask2Former non installé."

    register_datasets()
    cfg = build_config(batch_size=batch_size)
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_best.pth")

    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    evaluator = COCOEvaluator("trees_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = Mask2FormerTrainer.build_test_loader(cfg, "trees_val")
    results = inference_on_dataset(model, val_loader, evaluator)
    print(results)
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mask2Former Swin-T — Tree Segmentation")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Batch size (défaut=2, RTX 3090 24GB)")
    ap.add_argument("--resume", action="store_true",
                    help="Reprendre depuis le dernier checkpoint")
    ap.add_argument("--eval-only", action="store_true",
                    help="Évaluer le meilleur checkpoint seulement")
    args = ap.parse_args()

    if args.eval_only:
        evaluate_only(args.batch_size)
    else:
        train(args.batch_size, args.resume)
