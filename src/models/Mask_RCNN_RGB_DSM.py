"""
Mask R-CNN + DSM (RGB+DSM) - Entraînement pour segmentation d'instances d'arbres

Selon l'Appendix C du papier:
- Backbone: ResNet-50 avec FPN
- Entrée: 4 canaux (RGB empilé avec DSM)
- Modification Backbone: Pour la première couche (conv1), COPIER les poids des
  canaux RGB vers le canal DSM pour gérer l'entrée à 4 canaux
- Normalisation DSM: Par échantillon en divisant par la valeur maximale
- Hyperparamètres: Identiques à Mask R-CNN sauf Batch size = 8
- Augmentations: RandomFlip
- Critère de sélection: Meilleur mAP de segmentation sur validation

Optimisations v2:
- AMP (mixed precision) : ~1.5-2x speedup
- BestCheckpointer : sauvegarde automatique du meilleur model
- EvalHook dédupliqué (le double eval du build_hooks original est fixé)
- Scaling automatique LR + iters selon batch size (--batch-size)
- cudnn.benchmark = True
- NUM_WORKERS = 8
"""

import os
import sys
import cv2
import json
import argparse
import torch
import numpy as np
import torch.nn as nn

torch.backends.cudnn.benchmark = True

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, hooks, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.coco_helpers import fix_coco_json_if_needed
from utils.paths import (
    DATA_ROOT, ANN_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON, TEST_RGB as TEST_IMG_DIR,
)

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "output_mask_rcnn_RGB_DSM")

NUM_CLASSES     = 7
REFERENCE_BATCH = 2      # batch de référence du papier (adapté RTX 3090)
REFERENCE_LR    = 2.5e-5 # LR correspondant au batch de référence
N_TRAIN_IMAGES  = 18746


# ============================================================================
# BACKBONE 4 CANAUX
# ============================================================================
@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_4ch(cfg, input_shape):
    """ResNet-50+FPN avec conv1 modifiée pour 4 canaux (RGB+DSM)."""
    bottom_up = build_resnet_backbone(cfg, input_shape)
    old_conv  = bottom_up.stem.conv1

    if old_conv.in_channels != 4:
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Appendix C: copier la moyenne RGB vers le canal DSM
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = nn.Parameter(old_conv.bias.clone())
        bottom_up.stem.conv1 = new_conv

    from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm="",
        top_block=LastLevelMaxPool(),
        fuse_type="sum",
    )


# ============================================================================
# DATASET MAPPER 4 CANAUX
# ============================================================================
def MyFourChannelMapper(cfg, is_train=True):
    """DatasetMapper custom pour RGB + DSM (4 canaux)."""
    def mapper(dataset_dict):
        dataset_dict = dataset_dict.copy()

        # 1) Image RGB
        rgb_path  = dataset_dict["file_name"]
        image_rgb = utils.read_image(rgb_path, format="RGB")

        # 2) DSM — chercher le fichier avec gestion robuste des extensions
        dsm_path = rgb_path.replace("/RGB/", "/DSM/")
        if not os.path.exists(dsm_path):
            base, _ = os.path.splitext(dsm_path)
            for ext in ['.tif', '.TIF', '.png', '.PNG']:
                candidate = base + ext
                if os.path.exists(candidate):
                    dsm_path = candidate
                    break

        if os.path.exists(dsm_path):
            dsm = utils.read_image(dsm_path, format="F")
            dsm = np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.nanmax(dsm)
            if max_val > 0:
                dsm = dsm / max_val  # Appendix C: normalisation par max
        else:
            dsm = np.zeros(image_rgb.shape[:2], dtype=np.float32)

        if dsm.ndim == 2:
            dsm = dsm[:, :, None]

        # 3) Concaténer → H, W, 4
        image = np.concatenate([image_rgb, dsm], axis=2)

        # 4) Augmentations
        if is_train:
            augs = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(),
            ])
        else:
            augs = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                ),
            ])

        aug_input = T.AugInput(image)
        transforms = augs(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )

        # 5) Annotations
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
    return mapper


# ============================================================================
# EARLY STOPPING
# ============================================================================
class EarlyStoppingHook(HookBase):
    def __init__(self, eval_period, patience=10, metric_name="segm/AP", mode="max"):
        self.eval_period = int(eval_period)
        self.patience    = int(patience)
        self.metric_name = metric_name
        self.mode        = mode
        self.best        = float("-inf") if mode == "max" else float("inf")
        self.counter     = 0

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
                self.best    = curr
                self.counter = 0
                print(f"\n✅ Nouveau meilleur {self.metric_name}={curr:.4f}")
            else:
                self.counter += 1
                print(f"\n⏳ Pas d'amélioration ({self.counter}/{self.patience}) meilleur={self.best:.4f}")
            if self.counter >= self.patience:
                print("\n🛑 EARLY STOPPING")
                self.trainer.iter = self.trainer.max_iter - 1
        except Exception:
            pass


# ============================================================================
# TRAINER
# ============================================================================
class Trainer4ch(DefaultTrainer):
    """Trainer RGB+DSM avec AMP, BestCheckpointer et early stopping."""

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyFourChannelMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=MyFourChannelMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        ret = super().build_hooks()

        eval_period = int(self.cfg.TEST.EVAL_PERIOD)

        # BestCheckpointer — sauvegarde automatique du meilleur modèle
        ret.append(hooks.BestCheckpointer(
            eval_period=eval_period,
            checkpointer=self.checkpointer,
            val_metric="segm/AP",
            mode="max",
            file_prefix="model_best",
        ))

        # Early stopping
        ret.append(EarlyStoppingHook(
            eval_period=eval_period,
            patience=10,
            metric_name="segm/AP",
            mode="max",
        ))

        return ret


# ============================================================================
# CONFIGURATION
# ============================================================================
def build_cfg(num_classes: int, batch_size: int = REFERENCE_BATCH) -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # Datasets
    cfg.DATASETS.TRAIN = ("trees_train",)
    cfg.DATASETS.TEST  = ("trees_val",)
    cfg.DATALOADER.NUM_WORKERS = 8

    # 4 canaux
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53, 0.0]
    cfg.MODEL.PIXEL_STD  = [58.395,  57.12,  57.375, 1.0]
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone_4ch"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # ROI Heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES      = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = 0.5

    # ── Scaling linéaire LR + iters selon batch size ──────────────────────────
    ratio = batch_size / REFERENCE_BATCH   # ex: 8/2 = 4

    iters_per_epoch  = N_TRAIN_IMAGES // batch_size          # ~2343 pour batch=8
    max_epochs       = 50                                     # plafond (early stop avant)
    max_iter         = int(iters_per_epoch * max_epochs)
    steps            = (int(max_iter * 0.6), int(max_iter * 0.8))
    eval_period      = max(500, iters_per_epoch)              # 1 eval/epoch
    checkpoint_period = eval_period

    cfg.SOLVER.IMS_PER_BATCH     = batch_size
    cfg.SOLVER.BASE_LR           = REFERENCE_LR * ratio      # LR linéairement scalé
    cfg.SOLVER.WEIGHT_DECAY      = 0.0005
    cfg.SOLVER.WARMUP_FACTOR     = 0.01
    cfg.SOLVER.WARMUP_ITERS      = max(500, int(1000 / ratio))
    cfg.SOLVER.WARMUP_METHOD     = "linear"
    cfg.SOLVER.MAX_ITER          = max_iter
    cfg.SOLVER.STEPS             = steps
    cfg.SOLVER.GAMMA             = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # AMP — mixed precision (1.5-2x speedup sur RTX 3090)
    cfg.SOLVER.AMP = type(cfg.SOLVER.AMP)()
    cfg.SOLVER.AMP.ENABLED = True

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR   = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


# ============================================================================
# DATASET REGISTRATION
# ============================================================================
def register_datasets():
    for json_path in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if os.path.exists(json_path):
            fix_coco_json_if_needed(json_path)
    try:
        register_coco_instances("trees_train", {}, TRAIN_JSON, DATA_ROOT)
        register_coco_instances("trees_val",   {}, VAL_JSON,   DATA_ROOT)
        register_coco_instances("trees_test",  {}, TEST_JSON,  DATA_ROOT)
    except AssertionError:
        pass  # Déjà enregistré


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================
def train(num_classes: int, batch_size: int = REFERENCE_BATCH, resume: bool = False):
    print("=" * 65)
    print("  Mask R-CNN RGB+DSM (4 canaux) — Entraînement")
    print("=" * 65)

    register_datasets()
    cfg = build_cfg(num_classes, batch_size)

    iters_per_epoch = N_TRAIN_IMAGES // batch_size
    est_s_per_iter  = 0.30  # AMP + batch=8 ≈ 0.3s/iter (estimation)
    est_epochs_es   = 20    # early stop typiquement à ~20-30 epochs

    print(f"  Batch size      : {batch_size}")
    print(f"  LR              : {cfg.SOLVER.BASE_LR:.2e}")
    print(f"  Max iter        : {cfg.SOLVER.MAX_ITER:,} (~{cfg.SOLVER.MAX_ITER/iters_per_epoch:.0f} epochs)")
    print(f"  LR steps        : {cfg.SOLVER.STEPS}")
    print(f"  Eval period     : {cfg.TEST.EVAL_PERIOD} iters (~1 epoch)")
    print(f"  Early stop      : patience 10 évals")
    print(f"  AMP             : {'✅ ACTIVÉ' if cfg.SOLVER.AMP.ENABLED else '❌'}")
    print(f"  Temps estimé    : ~{est_epochs_es * iters_per_epoch * est_s_per_iter / 3600:.1f}h (avec early stop)")
    print("=" * 65)

    trainer = Trainer4ch(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    # Évaluation finale
    print("\n" + "=" * 65)
    print("  Évaluation finale sur validation")
    print("=" * 65)
    evaluator  = COCOEvaluator("trees_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = Trainer4ch.build_test_loader(cfg, "trees_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mask R-CNN RGB+DSM — 4 canaux")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--batch-size",  type=int, default=8,
                    help="Batch size (défaut=8 pour RTX 3090 après DINOv3)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    train(args.num_classes, args.batch_size, args.resume)
