# Grave Border Detection - Claude Guidelines

**CRITICAL: Follow Python rules in `docs/python-rules.md`**

## Project Overview (2026)

Computer vision project for detecting and vectorizing cemetery grave borders from drone imagery using deep learning. Produces GIS-compatible results for HADES-X cemetery management software.

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: uv (preferred) or pip
- **Deep Learning**: PyTorch with Lightning
- **Geospatial**: GDAL, Rasterio, GeoPandas
- **Computer Vision**: OpenCV, scikit-image, Albumentations
- **Code Quality**: Ruff (linting & formatting), MyPy (type checking)
- **Testing**: Pytest with coverage
- **Config**: Hydra

## Development Commands

```bash
# Install dependencies
uv sync --dev

# Code quality (run before every commit)
uv run task lint             # Linting, formatting, type checking
uv run task test             # Tests with coverage

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## Project Structure

```
grave-border-detection/
├── src/grave_border_detection/   # Main package
│   ├── models/                   # Neural network architectures
│   ├── data/                     # Data loaders and datasets
│   ├── preprocessing/            # Data preparation pipelines
│   ├── postprocessing/           # Vectorization and export
│   ├── utils/                    # Utility functions
│   └── visualization/            # Plotting and visualization
├── configs/                      # Hydra configuration files
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks (experimentation only)
├── docs/                         # Documentation
│   └── python-rules.md           # Python coding standards
└── data/                         # Data directories (never modify raw/)
```

## Core Principles

1. **Type Safety**: Type hints on all functions, never use `any`
2. **Modern Python 3.12+**: See `docs/python-rules.md`
3. **Fail Fast**: No silent fallbacks, no defensive "just in case" error handling. Raise exceptions explicitly.
4. **No Backwards Compatibility**: Delete old code, don't deprecate. No compatibility shims, re-exports, or `_old` prefixes.
5. **CRS Integrity**: Always preserve coordinate reference systems
6. **Memory Aware**: Windowed reading, tiling for large images

## Editing Guidelines

- **Edit inline**: When updating docs or code, modify existing content in place. Don't append new sections at the end while leaving outdated content above.
- **Delete, don't deprecate**: Remove obsolete code/docs entirely. No `# deprecated` comments.
- **No silent fallbacks**: Never add try/except that swallows errors or returns defaults. If something fails, let it fail loudly.

## Geospatial Rules

- Always read CRS from source files, never hardcode
- Validate CRS matches before combining datasets
- Use `rasterio` for GeoTIFFs, `geopandas` for vectors
- Use windowed reading for large rasters
- Never modify files in `data/raw/`

## Deep Learning Rules

- Use Lightning modules for training
- Store hyperparameters in Hydra configs
- Implement proper train/val/test splits
- Use Albumentations for augmentation
- Consider multi-channel input (RGB + height)

## Testing Requirements

- Minimum 80% test coverage
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Use pytest fixtures for test data
- Mock external dependencies (GDAL, file I/O)
- Test failure paths, not just happy paths

## Code Quality Checklist

Before completing any task:

- [ ] `uv run task lint` passes (ruff check, format, mypy)
- [ ] `uv run task test` passes with 80%+ coverage
- [ ] No unused imports/variables
- [ ] No commented-out code
- [ ] CRS preserved in geospatial operations

## Common Tasks

### Adding a New Model

1. Create file in `src/grave_border_detection/models/`
2. Inherit from `lightning.LightningModule`
3. Add config in `configs/model/`
4. Write tests in `tests/unit/models/`

### Processing New Data

1. Place raw data in `data/raw/`
2. Implement preprocessor in `preprocessing/`
3. Save to `data/processed/`
4. Never modify `data/raw/`

### Adding Export Format

1. Implement in `postprocessing/`
2. Preserve georeferencing
3. Add format validation tests

### Adding Dependencies

1. **Check the web** for current version and Python compatibility
2. Verify compatibility with existing stack (PyTorch, Lightning, GDAL)
3. Use `uv add <package>` to get latest compatible version
4. Update tests if new dependency requires mocking

## Reference Documentation

- **Python Rules**: `docs/python-rules.md`
- **Project Description**: `docs/Project_Description.md`
- **Data Inventory**: `docs/data-inventory.md`

## Current Data Status

- **10 cemeteries** with orthophotos (GeoTIFF, RGBA, ~4K)
- **10 DEMs** as GeoPackage tile pyramids
- **CRS**: EPSG:3857 (consistent across all files)
- **Annotations**: None yet (need to be created before training)
- **Location**: `data/external/goetiff-orthos-ki-training-data/4k/`
