# Models

One file per architecture. Quick map between the files here, the entries in the
paper (Teng et al., ICLR 2025 ML4RS) and the models added on top of it.

## From the paper

| File | Model | Notes |
|---|---|---|
| `SAM_OTB.py` | SAM zero-shot | Automatic mask generation, no training. `pps` = points per side. |
| `SAM_DSMPrompt.py` | SAM + DSM prompts | Builds prompts from the elevation data (DSM). |
| `Mask_RCNN_RGB.py` | Mask R-CNN (RGB) | RGB baseline detector. |
| `Mask_RCNN_RGB_DSM.py` | Mask R-CNN (RGB + DSM) | 4-channel input (RGB + elevation). |
| `Mask_RCNN_SAM.py` | Mask R-CNN + SAM | Refines Mask R-CNN proposals with SAM. Reuses the Mask R-CNN checkpoint. |
| `Faster_RCNN_SAM.py` | Faster R-CNN + SAM | Refines Faster R-CNN proposals with SAM. |
| `RSPrompter.py` | RSPrompter | Implemented but not trained (needs end-to-end SAM fine-tuning). |
| `DSM_Prompter.py` | DSMPrompter | Implemented but not trained. |

## Added on top of the paper

| File | Model | Notes |
|---|---|---|
| `SAM2_OTB.py` | SAM2 zero-shot | Same idea as `SAM_OTB.py` but with SAM2. Better masks here. |
| `SAM3_OTB.py` | SAM3 zero-shot | Experimental, needs a checkpoint that is not public yet. |
| `Dino_backbone.py` | DINOv3 backbone | ViT-Large + FPN, plus a small DSM encoder. Used with Mask R-CNN. |

## Variants

- `Mask_RCNN_SAM_from_json.py` / `Faster_RCNN_SAM_from_json.py`: same hybrid
  pipeline, but they read already-saved predictions (masks/boxes) from a JSON
  file instead of running the detector live.
- `DSM_Prompter_v2.py`: lighter version of DSMPrompter. Trains only the DSM
  encoder and the prompt head (no SAM2 decoder during training), which is much
  faster per epoch.
