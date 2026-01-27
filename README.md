# Grave Border Detection

Computer vision project for detecting and vectorizing cemetery grave borders from drone imagery using deep learning. Produces GIS-compatible results for integration with [HADES-X](https://hades-x.de/) cemetery management software.

## Overview

This project uses CNN-based semantic segmentation (U-Net architecture) to automatically detect grave borders in georeferenced drone orthophotos. The pipeline supports:

- **Multi-channel input**: RGB imagery with optional elevation (DEM) data
- **Tiled processing**: Handles large GeoTIFFs via windowed reading
- **GIS-compatible output**: Vectorized polygons in GeoJSON/GeoPackage format
- **Experiment tracking**: MLflow integration for logging and comparison
- **Hyperparameter optimization**: Optuna-based HPO with pruning and resumability

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd grave-border-detection

# Install dependencies with uv
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

## Quick Start

### Training a Model

```bash
# Train with default config (synthetic data)
uv run python -m grave_border_detection.train

# Train on real data
uv run python -m grave_border_detection.train data=real

# Quick debug run
uv run python -m grave_border_detection.train +experiment=debug

# Override parameters
uv run python -m grave_border_detection.train training.optimizer.lr=0.001 training.max_epochs=50
```

### Hyperparameter Optimization

```bash
# Run full HPO study
uv run python -m grave_border_detection.hpo

# Run pilot study (fewer trials)
uv run python -m grave_border_detection.hpo +hpo=pilot

# View results in dashboard
uv run optuna-dashboard sqlite:///hpo_studies.db
```

### Viewing Experiment Results

```bash
# Launch MLflow UI
uv run mlflow ui

# Open http://localhost:5000 in browser
```

## Project Structure

```
grave-border-detection/
├── src/grave_border_detection/   # Main package
│   ├── models/                   # Neural network architectures
│   ├── data/                     # Data loaders and datasets
│   ├── preprocessing/            # Data preparation pipelines
│   ├── postprocessing.py         # Vectorization and export
│   ├── callbacks/                # Training callbacks
│   ├── train.py                  # Training entrypoint
│   ├── hpo.py                    # Hyperparameter optimization
│   └── inference.py              # Model inference
├── configs/                      # Hydra configuration files
│   ├── config.yaml               # Main config
│   ├── data/                     # Data configs (synthetic, real)
│   ├── model/                    # Model configs
│   ├── training/                 # Training configs
│   ├── hpo/                      # HPO configs
│   └── experiment/               # Experiment presets
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
└── data/                         # Data directories
    ├── raw/                      # Original data (read-only)
    ├── external/                 # External datasets
    └── processed/                # Preprocessed data
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Configs are composed from multiple files:

```yaml
# configs/config.yaml
defaults:
  - data: synthetic      # or: real
  - model: unet_resnet34
  - training: default
  - augmentation: default
```

Override any parameter from the command line:

```bash
uv run python -m grave_border_detection.train \
  data=real \
  data.batch_size=8 \
  training.max_epochs=100 \
  model.encoder_name=resnet50
```

## Data

### Available Data

| Type | Count | Format | CRS |
|------|-------|--------|-----|
| Orthophotos | 10 | GeoTIFF (RGBA) | EPSG:3857 |
| DEMs | 10 | GeoPackage tiles | EPSG:3857 |
| Annotations | 1,465 polygons | GeoPackage | EPSG:3857 |

### Data Preparation

1. Place raw data in `data/external/`
2. Run preprocessing to generate tiles:
   ```bash
   uv run python -m grave_border_detection.preprocessing.prepare_real_data
   ```
3. Train on processed data:
   ```bash
   uv run python -m grave_border_detection.train data=real
   ```

## Development

### Code Quality

```bash
# Run linting, formatting, and type checking
uv run task lint

# Auto-fix linting issues
uv run task fix

# Run tests with coverage
uv run task test

# Run all checks (lint + test)
uv run task all
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit:

```bash
# Run manually on all files
uv run pre-commit run --all-files
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

## Model Architecture

The default model is U-Net with a ResNet34 encoder (pretrained on ImageNet):

- **Input**: 4 channels (RGB + DEM) or 3 channels (RGB only)
- **Output**: Binary segmentation mask (grave border / background)
- **Loss**: BCE + Dice loss combination
- **Metrics**: Dice coefficient, IoU

## License

MIT

## Author

Berenike Radeck
