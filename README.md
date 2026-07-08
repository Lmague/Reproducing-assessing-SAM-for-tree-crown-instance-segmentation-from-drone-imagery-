# Tree Crown Instance Segmentation — SAM, SAM2, Mask R-CNN & DINOv3

Master's side project — reproduction and extension of Teng et al., ICLR 2025 ML4RS, on the UAV Canadian Quebec Plantations dataset.

The paper benchmarks SAM variants and Mask R-CNN for individual tree crown instance segmentation from drone imagery, optionally using DSM (elevation data) as an extra input. I reproduced most of it and added a few things (SAM2, SAM3, DINOv3).

---

## What's implemented

| | |
|---|---|
| **From the paper** | SAM zero-shot, SAM + DSM prompts, Mask R-CNN RGB, Mask R-CNN + DSM, Faster R-CNN + SAM, Mask R-CNN + SAM |
| **Added** | SAM2 zero-shot, SAM3 zero-shot, DINOv3 ViT-Large backbone + FPN |
| **Implemented but not trained** | RSPrompter, DSMPrompter — the code is there but I didn't have enough GPU to actually train them (they require fine-tuning SAM end-to-end) |

---

## Results

All metrics × 100. **mAP tree** = COCO AP[0.5:0.95] with everything collapsed to one "tree" class. **mIoU** = for each ground truth tree, take the prediction with best overlap and average. Full numbers in [`metrics_summary.json`](metrics_summary.json).

### Models from the paper

| Model | mAP (tree) | mIoU | mAP | wmAP | Paper mAP (tree) |
|-------|-----------|------|-----|------|-----------------|
| **Mask R-CNN + SAM** | **52.97** | **72.94** | 48.76 | 49.67 | 57.60 ±0.19 |
| Mask R-CNN RGB | 51.70 | 73.24 | 49.30 | 49.88 | 63.65 ±0.43 |
| Mask R-CNN RGB+DSM | 39.73 | 64.09 | 36.08 | 37.35 | 64.64 ±0.69 |
| SAM OTB (pps=10) | 17.54 | 66.24 | – | – | 10.11 |
| SAM OTB (pps=5) | 14.56 | 52.79 | – | – | – |
| SAM OTB (pps=3) | 6.51 | 31.06 | – | – | – |
| SAM + DSM prompts | 15.25 | 57.48 | – | – | 9.28 |
| Faster R-CNN + SAM | 2.86 | 12.66 | 0.49 | 0.95 | 57.85 ±0.66 |
| RSPrompter | — | — | — | — | 66.37 ±0.91 |
| DSMPrompter | — | — | — | — | 65.03 ±1.76 |

> **Faster R-CNN** didn't converge during training (standalone mAP tree = 0.06%), so Faster R-CNN + SAM is also bad as a result — SAM can't compensate for bad detections.
>
> The paper averages results over 3 seeds, mine are single runs, which explains part of the gap.

### Original contributions

| Model | mAP (tree) | mIoU |
|-------|-----------|------|
| SAM2 OTB (pps=10) | **22.43** | **67.95** |
| SAM2 OTB (pps=5) | 18.27 | 53.77 |
| SAM2 OTB (pps=3) | 8.12 | 30.27 |
| SAM3 OTB (descriptive prompts) | 7.53 | 10.68 |
| SAM3 OTB (generic prompts) | 4.47 | 5.72 |
| DINOv3 ViT-Det RGB | 0.34 | 43.77 |

SAM2 beats SAM zero-shot on both metrics (22.43 vs 17.54 mAP tree).

DINOv3 results are very low — training didn't converge properly. One likely reason: the checkpoint I used (`dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`) is pretrained on **satellite images**, not drone images. Satellite and drone are quite different domains: satellite imagery is low resolution (meters/pixel), taken from much higher altitude, often multispectral — while drone data here is ~5mm/pixel RGB at low altitude. Features learned on satellite data probably don't transfer well to this scale.

### Per-class AP

| Species | Mask R-CNN + SAM | Mask R-CNN RGB | Mask R-CNN RGB+DSM |
|---------|-----------------|---------------|---------------------|
| thoc | 73.25 | 73.91 | 62.90 |
| pist | 66.18 | 66.98 | 53.10 |
| pigl | 52.25 | 53.67 | 41.61 |
| piba | 46.82 | 46.05 | 35.01 |
| others | 35.55 | 35.79 | 17.43 |
| pima | 35.17 | 35.28 | 20.20 |
| ulam | 31.89 | 33.46 | 22.30 |

---

## Installation

```bash
pip install -r requirements.txt
```

Download checkpoints into the `checkpoints/` directory at the repository root:

```bash
mkdir -p checkpoints

# SAM ViT-H (needed for all SAM-based models)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth

# SAM2 large (needed for SAM2 inference and DSMPrompter)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2.1_hiera_large.pt

# DINOv3 ViT-L/16 (satellite-pretrained) - see DINOv3 project page for the link
# Save as checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

### Configuring paths

All paths are centralized in [`config.yaml`](config.yaml) and resolved by [`src/utils/paths.py`](src/utils/paths.py). Defaults assume the layout below (everything relative to the repo root). You can override any path either by editing `config.yaml` or via environment variables:

```bash
export TREE_SEG_DATA_ROOT=/path/to/tiles
export TREE_SEG_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export TREE_SEG_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
export TREE_SEG_DINO_CHECKPOINT=/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

---

## Dataset

UAV Canadian Quebec Plantations dataset — RGB orthomosaics + DSM at ~5mm/pixel, 7 tree species + "others" class, 15 plantation sites.

Generate 1024×1024 tiles from raw data:

```bash
python src/utils/tiles_generator.py --config config.yaml
```

Expected dataset layout under `data_root` (default: `./data`):

```
data/
├── train/RGB/   train/DSM/
├── val/RGB/     val/DSM/
├── test/RGB/    test/DSM/
└── coco/        # train.json, val.json, test.json
```

---

## Usage

### Training

```bash
python src/models/Mask_RCNN_RGB.py        # Mask R-CNN baseline
python src/models/Mask_RCNN_RGB_DSM.py    # Mask R-CNN + DSM (4-channel input)
python src/training/train_dino.py         # DINOv3 + Mask R-CNN
```

Mask R-CNN + SAM uses the Mask R-CNN checkpoint — train it first, then:
```bash
python src/models/Mask_RCNN_SAM.py
```

RSPrompter needs SAM embeddings pre-computed first:
```bash
python src/misc/Pre_Compute_SAM.py   # takes a while, needs the SAM checkpoint
python src/models/RSPrompter.py
```

### Zero-shot inference (no training needed)

```bash
python src/models/SAM_OTB.py          # SAM automatic mode
python src/models/SAM_DSMPrompt.py    # SAM with DSM-based prompts
```

### Evaluation

```bash
python src/misc/evaluate_models.py \
  --gt data/coco/test.json \
  --pred path/to/predictions.json \
  --out-json metrics_summary.json
```

### Quick checks (CPU, no checkpoint needed)

```bash
jupyter notebook notebooks/cpu_tests.ipynb
```

---

## Project structure

```
├── src/
│   ├── models/      # One file per architecture
│   ├── training/    # Training scripts for DINOv3 and DSMPrompter
│   ├── inference/   # Inference scripts and demo
│   ├── utils/       # Dataset generation, COCO helpers, central path config (paths.py)
│   └── misc/        # Evaluation, SAM/DINO embedding precompute
├── notebooks/
│   ├── cpu_tests.ipynb                          # Unit tests (no GPU needed)
│   └── smoke_test_rsprompter_dsmprompter.ipynb  # Check setup before training
├── config.yaml            # Paths + per-model hyperparameters
├── requirements.txt
├── metrics_summary.json   # All evaluation results
├── metrics_summary.csv
└── metrics_analysis.md
```

---

## Notes

- **LR scaling**: batch sizes are lower than in the paper due to GPU constraints, LR is scaled accordingly: `LR = LR_paper × (batch / batch_paper)`
- **Early stopping**: Mask R-CNN training stops if val mAP doesn't improve for 10 consecutive evaluations
- **RSPrompter / DSMPrompter**: implemented but not trained — these are the paper's best models but require fine-tuning SAM end-to-end which I couldn't run
- **SAM3**: experimental — `src/models/SAM3_OTB.py` is provided for reference but needs a HuggingFace checkpoint that isn't publicly available yet

---

## Original paper & Dataset

- Lefebvre, I., & Laliberté, E. (2024). UAV LiDAR, UAV Imagery, Tree Segmentations and Ground Mesurements for Estimating Tree Biomass in Canadian (Quebec) Plantations [Jeu de données]. Federated Research Data Repository / dépôt fédéré de données de recherche. https://doi.org/10.20383/103.0979
- Teng, M., Ouaknine, A., Laliberté, E., Bengio, Y., Rolnick, D., & Larochelle, H. (2025, mars 26). Assessing SAM for Tree Crown Instance Segmentation from Drone Imagery. ICLR 2025 ML4RS workshop. https://doi.org/10.48550/arXiv.2503.20199

