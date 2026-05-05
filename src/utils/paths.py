"""
Centralized path resolution for the project.

Reads `config.yaml` at the repository root and exposes absolute paths for
data, COCO annotations, and model checkpoints. Each path can be overridden
by an environment variable, which is convenient on shared GPU servers.

Environment overrides:
    TREE_SEG_DATA_ROOT       -> paths.data_root
    TREE_SEG_SAM_CHECKPOINT  -> paths.sam_checkpoint
    TREE_SEG_SAM2_CHECKPOINT -> paths.sam2_checkpoint
    TREE_SEG_DINO_CHECKPOINT -> paths.dino_checkpoint

Author: Lmague
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _find_repo_root(start: Path) -> Path:
    """Walk upwards from `start` until we find a directory containing config.yaml."""
    for candidate in [start, *start.parents]:
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate config.yaml starting from {start}. "
        "Make sure the repo root contains a config.yaml file."
    )


REPO_ROOT: Path = _find_repo_root(Path(__file__).resolve().parent)
CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def _resolve(path_str: str) -> str:
    """Resolve a path relative to the repo root if it is not absolute."""
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p.resolve())


_cfg = load_config()
_paths_cfg = _cfg.get("paths", {})

DATA_ROOT: str = _resolve(
    os.environ.get("TREE_SEG_DATA_ROOT", _paths_cfg.get("data_root", "./data"))
)
SAM_CHECKPOINT: str = _resolve(
    os.environ.get(
        "TREE_SEG_SAM_CHECKPOINT",
        _paths_cfg.get("sam_checkpoint", "./sam_vit_h_4b8939.pth"),
    )
)
SAM2_CHECKPOINT: str = _resolve(
    os.environ.get(
        "TREE_SEG_SAM2_CHECKPOINT",
        _paths_cfg.get("sam2_checkpoint", "./checkpoints/sam2.1_hiera_large.pt"),
    )
)
DINO_CHECKPOINT: str = _resolve(
    os.environ.get(
        "TREE_SEG_DINO_CHECKPOINT",
        _paths_cfg.get(
            "dino_checkpoint", "./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        ),
    )
)

ANN_ROOT: str = os.path.join(DATA_ROOT, "coco")
TRAIN_JSON: str = os.path.join(ANN_ROOT, "train.json")
VAL_JSON: str = os.path.join(ANN_ROOT, "val.json")
TEST_JSON: str = os.path.join(ANN_ROOT, "test.json")

TRAIN_RGB: str = os.path.join(DATA_ROOT, "train", "RGB")
VAL_RGB: str = os.path.join(DATA_ROOT, "val", "RGB")
TEST_RGB: str = os.path.join(DATA_ROOT, "test", "RGB")

TRAIN_DSM: str = os.path.join(DATA_ROOT, "train", "DSM")
VAL_DSM: str = os.path.join(DATA_ROOT, "val", "DSM")
TEST_DSM: str = os.path.join(DATA_ROOT, "test", "DSM")


__all__ = [
    "REPO_ROOT",
    "CONFIG_PATH",
    "load_config",
    "DATA_ROOT",
    "ANN_ROOT",
    "TRAIN_JSON",
    "VAL_JSON",
    "TEST_JSON",
    "TRAIN_RGB",
    "VAL_RGB",
    "TEST_RGB",
    "TRAIN_DSM",
    "VAL_DSM",
    "TEST_DSM",
    "SAM_CHECKPOINT",
    "SAM2_CHECKPOINT",
    "DINO_CHECKPOINT",
]
