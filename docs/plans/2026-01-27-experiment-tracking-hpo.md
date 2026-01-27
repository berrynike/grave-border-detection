# Experiment Tracking & HPO Implementation Plan

## Status: Phases 1-4 Complete

## Overview

Consolidate experiment tracking to MLflow-only, add dataset versioning, implement hyperparameter optimization with Optuna, and enhance visualization logging to include DEM data and individual artifacts.

---

## Completed Work

### Phase 1: Consolidate to MLflow-Only Tracking ✅

**Status**: Complete

Removed duplicate local file storage. Callbacks now use temp files for intermediate steps, cleaned up after logging to MLflow.

**Files Modified**:
- `src/grave_border_detection/callbacks/image_logger.py`
- `src/grave_border_detection/callbacks/full_cemetery_viz.py`

### Phase 2: Enhanced Image Logging (DEM + Individual Artifacts) ✅

**Status**: Complete

Added DEM visualization to logged images (when DEM is present) and log each visualization component as a separate artifact. Works for both RGB-only (3-channel) and RGB+DEM (4-channel) runs.

**Implementation**:
- Grid adapts: 3 columns for RGB-only, 4 columns for RGB+DEM (GT overlay logged separately)
- Individual artifacts logged: rgb, dem (if present), gt_overlay, pred_heatmap, error_map
- DEM visualized with terrain colormap (blue→green→brown)

### Phase 3: Dataset Versioning ✅

**Status**: Complete

Implemented hash-based dataset identification to track which data was used for each run.

**Files Created**:
- `src/grave_border_detection/utils/dataset_hash.py`

**Implementation**:
- `compute_dataset_id()` generates SHA256 hash from file manifest (paths, sizes, mtimes)
- Returns 8-character hash prefix (e.g., "f94e5e59")
- Logged to MLflow as hyperparameter AND native Dataset

### Phase 4: Optuna HPO Integration ✅

**Status**: Complete

Added hyperparameter optimization with Optuna, including SQLite storage for resumability and pruning for efficiency.

**Files Created**:
- `src/grave_border_detection/hpo.py`
- `configs/hpo/default.yaml`
- `configs/hpo/pilot.yaml`
- `scripts/run_hpo.sh`
- `scripts/view_hpo.sh`

---

## MLflow Integration (Updated to Modern APIs)

Updated all MLflow logging to use current best practices per [MLflow docs](https://mlflow.org/docs/latest/python_api/mlflow.html):

### Image Logging
**Old approach** (removed):
```python
# Temp files + MlflowClient().log_artifact()
with tempfile.TemporaryDirectory() as tmpdir:
    img_path = tmpdir / "image.png"
    img.save(img_path)
    client.log_artifact(run_id, str(img_path), artifact_path="predictions")
```

**New approach** (implemented):
```python
# Direct mlflow.log_image() with numpy arrays
import mlflow
mlflow.log_image(img_numpy, key="predictions/combined", step=epoch)
```

### Dataset Logging
**Implementation** (native Dataset tracking):
```python
import mlflow
import mlflow.data

dataset_df = pd.DataFrame({"cemetery": cemeteries, "split": splits})
mlflow_dataset = mlflow.data.from_pandas(
    dataset_df,
    source=str(data_root),
    name=f"dataset_{dataset_id}",
)
mlflow.log_input(mlflow_dataset, context="training")
```

This populates the native "Dataset" column in MLflow UI.

### Files Updated for Modern MLflow:
- `src/grave_border_detection/train.py` - Added `mlflow.log_input()` for dataset tracking
- `src/grave_border_detection/hpo.py` - Added `mlflow.log_input()` for dataset tracking
- `src/grave_border_detection/callbacks/image_logger.py` - Uses `mlflow.log_image()` with `key` and `step`
- `src/grave_border_detection/callbacks/full_cemetery_viz.py` - Uses `mlflow.log_image()` for images, keeps `log_artifact()` for GeoPackages

---

## Phase 5: Pilot HPO Study (Pending)

### Overview
Run a small pilot study to validate setup and get initial insights on promising hyperparameter ranges.

### To Run:
```bash
# Quick test (2 trials, 2-3 epochs)
uv run python -m grave_border_detection.hpo +hpo=pilot data=rgb_only \
    hpo.n_trials=2 hpo.search_space.max_epochs.low=2 hpo.search_space.max_epochs.high=3 \
    hpo.study_name=test_run hpo.storage=sqlite:///test.db

# Full pilot study (20 trials)
bash scripts/run_hpo.sh +hpo=pilot

# View results
bash scripts/view_hpo.sh sqlite:///hpo_pilot.db
```

### Success Criteria:
- [ ] Run pilot study completes
- [ ] Analyze results in Optuna Dashboard
- [ ] Document insights about search space

---

## What We're NOT Doing

- Ray Tune / distributed HPO (future enhancement when scaling to cluster)
- DVC integration (using simpler hash-based versioning for now)
- Multi-objective optimization (single objective: val/dice)
- `mlflow.pytorch.autolog()` - Already have comprehensive logging via MLFlowLogger

---

## References

- MLflow Python API: https://mlflow.org/docs/latest/python_api/mlflow.html
- MLflow Dataset Tracking: https://mlflow.org/docs/latest/tracking/data-api.html
- MLflow log_image(): https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_image
- PyTorch Lightning MLFlowLogger: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html
- Optuna RDB docs: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
- Optuna Dashboard: https://github.com/optuna/optuna-dashboard
