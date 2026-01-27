# Wrinkle V2 Improvements

This folder contains the Wrinkle V2 pipeline plus four improvements:

1) Boundary-aware segmentation loss
2) Stability evaluation metric for the end-to-end pipeline
3) Region-based normalization for measurement-based scoring
4) Optional score calibration plug-in (age/skin-type aware)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset configuration

Edit `configs/default.yaml`:

- `dataset_root` points to the local Roboflow export root.
- Supported label formats:
  - YOLOv8-seg polygon txt files (one file per image).
  - COCO JSON with polygon segmentations.

Expected split folders:

```
train/
valid/
test/
```

Each split may include `images/` and `labels/` (YOLO) or `annotations/` (COCO). The loader auto-detects.

## Training

```bash
python scripts/train_seg.py --config configs/default.yaml
```

Training will auto-resume if `checkpoints/wrinkle_unet_last.pt` exists.

To train with a custom dataset path without editing the file, create a copy and change `dataset_root`, then pass it:

```bash
cp configs/default.yaml configs/local.yaml
python scripts/train_seg.py --config configs/local.yaml
```

Checkpoints are saved to `checkpoints/` and exported as TorchScript.

## Inference (CLI)

```bash
python scripts/predict.py --image /path/to/image.jpg
python scripts/predict.py --image /path/to/image.jpg --overlay-out outputs_overlay.jpg
```

## Stability evaluation

```bash
python -m wrinkle_v2_improvements.stability_eval --input_dir /path/to/images --out_dir /path/to/out --config configs/default.yaml
```

This runs controlled perturbations and writes per-image reports plus `summary.json`.

## Evaluation

```bash
python scripts/eval_seg.py --config configs/local.yaml --checkpoint checkpoints/wrinkle_unet_best.pt
```

Outputs metrics to `runs/metrics.json`.

## Threshold calibration

```bash
python scripts/calibrate_threshold.py --config configs/local.yaml --checkpoint checkpoints/wrinkle_unet_best.pt --min 0.2 --max 0.8 --step 0.05
```

Outputs a sweep to `runs/threshold_sweep.json` and prints best threshold for `inference.mask_threshold`.

## Inference (API)

```bash
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

POST an image to `/predict` and receive JSON.

## Pipeline details

### 1) Segmentation

UNet with ResNet34 encoder is used to produce a binary wrinkle mask. This is better suited to thin texture segmentation than object-centric instance models (e.g., Mask R-CNN) because it preserves fine-grained pixel-level continuity.

Boundary-aware loss (in `losses.py`) adds an edge-focused penalty via morphological gradients. Configure in `configs/default.yaml` under `loss`.

### 2) Quality gate

The global quality gate runs before any skin concern pipeline. If it fails, the output is `NO_SCORE` with reasons. Checks include:
- No face detected (configurable)
- Face too small
- Blurry image (variance of Laplacian)
- Too dark / too bright

Soft flags from the gate can reduce scores (without forcing NO_SCORE). See `docs/QUALITY_GATE.md`.

### 3) Region mapping

Regions are derived from MediaPipe FaceMesh landmark polygons. Regions used:
- Forehead
- Glabella (frown lines)
- Bunny lines (nose)
- Crow's feet (left/right)
- Under-eye (left/right)
- Nasolabial folds (left/right)
- Smoker lines (perioral)
- Chin

See landmark index lists in `src/wrinkle_v2/regions/face_regions.py`.

### 4) Measurement-based scoring (no confidence)

Global and per-region scores are computed from the predicted mask:
- Wrinkle area ratio: wrinkle pixels / face pixels
- Skeleton length: length of the thinned wrinkle lines
- Thickness: distance transform width proxy
- Density: connected components per region area

The final 0-100 score is a weighted sum of normalized region measurements. Use `scoring.region_normalization` and `scoring.region_weights` in `configs/default.yaml`.

### 5) Reasoning text

Deterministic rule-based reasoning is produced from measurements and quality signals. No external LLM calls.

### 6) Optional calibration

Calibration is a plug-in mapping from raw score to calibrated score using fixed anchors. It is disabled by default and can be enabled in `configs/default.yaml` under `calibration`.

## Metrics

- Dice coefficient
- IoU
- Boundary F1
- Region metrics: area ratio, skeleton length, thickness

## Tests

```bash
python -m pytest -q
```

## No-score behavior examples

- "No score due to quality gate: image_blurry"
- "No score due to quality gate: face_too_small"

## Smoke test

```bash
python scripts/smoke_test.py
```

Loads one test image, runs quality gate, segmentation, and prints JSON.
The smoke test disables MediaPipe to avoid native crashes; the main pipeline uses MediaPipe by default.

Tests live under `tests/` and are intentionally minimal to avoid extra dependencies.
