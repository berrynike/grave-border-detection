# Implementation Plan: Grave Border Detection POC

## Overview

Step-by-step plan to build the pipeline incrementally, testing each component before moving on.

---

## Phase 1: Foundation & Synthetic Data

### Step 1.1: Synthetic Data Generator ✅ DONE
**Status:** Done

**What:** Module to generate fake cemetery images with rectangular graves.

**Files:**
- `src/grave_border_detection/data/synthetic.py`

**Acceptance Criteria:**
- [ ] Generates orthophoto GeoTIFFs (RGBA, uint8)
- [ ] Generates DEM GeoTIFFs (float32)
- [ ] Generates mask GeoTIFFs (uint8, 0/1)
- [ ] Generates annotations GeoPackage with polygons
- [ ] All files have correct CRS (EPSG:3857)
- [ ] Files can be opened in QGIS

**Test Command:**
```bash
uv run python -m grave_border_detection.data.synthetic
```

---

### Step 1.2: Verify Synthetic Data ✅ DONE
**Status:** Complete

**What:** Write tests and visually verify the generated data.

**Acceptance Criteria:**
- [ ] Unit test passes
- [ ] GeoTIFFs readable with `rasterio`
- [ ] GeoPackage readable with `geopandas`
- [ ] CRS matches across all files
- [ ] Masks align with orthophotos

**Test Command:**
```bash
uv run pytest tests/unit/data/test_synthetic.py -v
```

---

## Phase 2: Configuration & Project Structure

### Step 2.1: Hydra Config Structure ✅ DONE
**Status:** Complete

**What:** Create YAML config files for all pipeline components.

**Files:**
```
configs/
├── config.yaml
├── data/default.yaml
├── model/unet_resnet34.yaml
├── training/default.yaml
├── augmentation/default.yaml
└── experiment/baseline.yaml
```

**Acceptance Criteria:**
- [ ] Configs load without error
- [ ] Can override values from CLI
- [ ] Configs are well-documented

**Test Command:**
```bash
uv run python -c "from hydra import compose, initialize; initialize('configs'); cfg = compose('config'); print(cfg)"
```

---

## Phase 3: Data Pipeline

### Step 3.1: Dataset Class ✅ DONE
**Status:** Complete

**What:** PyTorch Dataset that loads tiles from orthophotos + masks.

**Files:**
- `src/grave_border_detection/data/dataset.py`

**Acceptance Criteria:**
- [ ] Returns (image, mask) tuples
- [ ] Supports RGB-only and RGB+DEM modes
- [ ] Applies augmentations correctly (image + mask synced)
- [ ] Handles edge tiles properly

**Test Command:**
```bash
uv run pytest tests/unit/data/test_dataset.py -v
```

---

### Step 3.2: Tiling Module ✅ DONE
**Status:** Complete

**What:** Extract tiles from large GeoTIFFs with overlap.

**Files:**
- `src/grave_border_detection/data/tiling.py`

**Acceptance Criteria:**
- [ ] Tiles cover entire image
- [ ] Overlap works correctly
- [ ] Can reconstruct full image from tiles
- [ ] Preserves georeferencing

**Test Command:**
```bash
uv run pytest tests/unit/data/test_tiling.py -v
```

---

### Step 3.3: Lightning DataModule ✅ DONE
**Status:** Complete

**What:** Lightning DataModule wrapping dataset + dataloaders.

**Files:**
- `src/grave_border_detection/data/datamodule.py`

**Acceptance Criteria:**
- [ ] `setup()` creates train/val/test datasets
- [ ] Dataloaders return correct batch shapes
- [ ] Split by cemetery (no leakage)
- [ ] Configurable via Hydra

**Test Command:**
```bash
uv run pytest tests/unit/data/test_datamodule.py -v
```

---

## Phase 4: Model

### Step 4.1: Segmentation Model Wrapper ✅ DONE
**Status:** Complete

**What:** Lightning Module wrapping SMP U-Net.

**Files:**
- `src/grave_border_detection/models/segmentation.py`

**Acceptance Criteria:**
- [ ] Supports configurable encoder (ResNet34, EfficientNet)
- [ ] Supports 3 or 4 input channels
- [ ] Loss function works (BCE + Dice)
- [ ] Metrics logged (Dice, IoU)
- [ ] Forward pass produces correct output shape

**Test Command:**
```bash
uv run pytest tests/unit/models/test_segmentation.py -v
```

---

## Phase 5: Training

### Step 5.1: Training Entrypoint ✅ DONE
**Status:** Complete

**What:** Hydra-powered training script.

**Files:**
- `src/grave_border_detection/train.py`

**Acceptance Criteria:**
- [ ] Runs with `uv run python -m grave_border_detection.train`
- [ ] Loads config correctly
- [ ] Creates MLflow run
- [ ] Saves checkpoints
- [ ] Logs metrics and images

**Test Command:**
```bash
# Quick test with synthetic data
uv run python -m grave_border_detection.train \
    data.root=data/synthetic \
    training.max_epochs=2 \
    training.fast_dev_run=true
```

---

### Step 5.2: End-to-End Training Test ✅ DONE
**Status:** Complete

**What:** Full training run on synthetic data.

**Acceptance Criteria:**
- [ ] Training completes without error
- [ ] Loss decreases over epochs
- [ ] Validation metrics logged
- [ ] Model checkpoint saved
- [ ] MLflow UI shows run

**Test Command:**
```bash
uv run python -m grave_border_detection.train \
    data.root=data/synthetic \
    training.max_epochs=10
```

---

## Phase 6: Inference & Postprocessing

### Step 6.1: Inference Module
**Status:** Pending

**What:** Sliding window inference on full images.

**Files:**
- `src/grave_border_detection/predict.py`

**Acceptance Criteria:**
- [ ] Produces probability map (GeoTIFF)
- [ ] Produces binary mask (GeoTIFF)
- [ ] Handles overlap blending
- [ ] Preserves georeferencing

---

### Step 6.2: Instance Separation
**Status:** Pending

**What:** Separate touching graves into individual instances.

**Files:**
- `src/grave_border_detection/postprocessing/instance_separation.py`

**Acceptance Criteria:**
- [ ] Connected components work
- [ ] Watershed available as option
- [ ] Filters by min/max area
- [ ] Outputs labeled raster

---

### Step 6.3: Vectorization
**Status:** Pending

**What:** Convert raster masks to vector polygons.

**Files:**
- `src/grave_border_detection/postprocessing/vectorize.py`

**Acceptance Criteria:**
- [ ] Produces valid GeoPackage
- [ ] Polygons are simplified
- [ ] Includes attributes (area, confidence)
- [ ] CRS preserved

---

## Phase 7: Integration

### Step 7.1: Full Pipeline Test
**Status:** Pending

**What:** Run entire pipeline on synthetic data.

**Acceptance Criteria:**
- [ ] Data prep → Training → Inference → Vectorization
- [ ] Output GeoPackage can be opened in QGIS
- [ ] Predicted polygons roughly match input annotations
- [ ] Documentation complete

---

## Execution Order

```
Phase 1: Foundation
  1.1 ✅ Synthetic data generator
  1.2 ✅ Verify synthetic data

Phase 2: Configuration
  2.1 ✅ Hydra configs

Phase 3: Data Pipeline
  3.1 ✅ Dataset class
  3.2 ✅ Tiling module
  3.3 ✅ DataModule

Phase 4: Model
  4.1 ✅ Segmentation model

Phase 5: Training
  5.1 ✅ Training entrypoint
  5.2 ✅ E2E training test

Phase 6: Inference
  6.1 ✅ Inference module (full cemetery prediction)
  6.2 ⚠️ Instance separation (basic - needs improvement)
  6.3 ✅ Vectorization (rectangle fitting + GeoPackage export)

Phase 7: Integration
  7.1 ✅ Full pipeline test (train → test → visualize → export)

Phase 8: Future Improvements (TODO)
  8.1 ⬜ Improve instance separation (watershed, boundary detection)
  8.2 ⬜ Train on real annotated data
  8.3 ⬜ Try instance segmentation (Mask R-CNN)
```

---

## Quick Validation Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run linting
uv run task lint

# Generate synthetic data
uv run python -m grave_border_detection.data.synthetic

# Quick training test
uv run python -m grave_border_detection.train training.fast_dev_run=true

# View MLflow UI
mlflow ui
```
