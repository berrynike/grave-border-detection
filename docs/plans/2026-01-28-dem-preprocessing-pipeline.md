# DEM Preprocessing Pipeline Implementation Plan

## Overview

Implement a configurable DEM preprocessing pipeline that:
1. Extracts DEMs from GeoTIFF sources with minimal modification (one-time manual step)
2. Applies normalization at runtime during DataModule.setup() on full cemetery DEMs
3. Enables HPO to compare different preprocessing methods and parameters

## Current State Analysis

### Key Discoveries:
- **DEM normalization is hardcoded** as z-score in `dataset.py:169-170`
- **Normalization happens per-tile** which breaks methods like local_height that need full context
- **Source DEMs need resampling** - DEMs are 2-3x coarser than orthophotos
- **Current DEMs in `data/real/dems/` are incorrectly processed** - scaled to 0-255 instead of real elevation

### Files to Modify:
- `src/grave_border_detection/preprocessing/prepare_real_data.py` - DEM extraction (Phase 1)
- `src/grave_border_detection/preprocessing/dem_normalization.py` - new module (Phase 2)
- `src/grave_border_detection/data/datamodule.py` - runtime normalization (Phase 3)
- `src/grave_border_detection/data/dataset.py` - remove per-tile normalization (Phase 3)
- `configs/data/real.yaml` - DEM config options (Phase 3)

## Desired End State

1. **Raw DEMs stored** with real elevation values (meters), resampled to match orthophoto resolution
2. **Normalization computed at runtime** on full cemetery images during DataModule.setup()
3. **Normalized DEMs cached in memory**, tiling reads from cache
4. **HPO can search** over normalization methods and parameters

### Verification:
- `uv run task lint` passes
- `uv run task test` passes
- Training works with each preprocessing method
- HPO can vary `data.dem.normalization.method` and parameters

## What We're NOT Doing

- Extracting from GeoPackage (those are colormap visualizations, not real elevation)
- Pre-computing all normalization variants to disk (too inflexible for HPO)
- Per-tile normalization (breaks local_height context)

## Implementation Approach

```
Phase 1 (Manual, one-time):
  Source GeoTIFF DEMs → Extraction script → Raw resampled DEMs on disk

Phase 2-4 (Code changes):
  Training run starts
      ↓
  DataModule.setup() loads full raw DEMs
      ↓
  Applies normalization (method from config) to full images
      ↓
  Caches normalized DEMs in memory
      ↓
  Dataset tiles read from normalized cache
      ↓
  Training loop
```

---

## Phase 1: DEM Extraction Script (Manual, One-Time)

### Overview
Create/update script to extract and resample DEMs from GeoTIFF sources. Run manually when new DEM data arrives.

### Changes Required:

#### 1. Add DEM resampling function
**File**: `src/grave_border_detection/preprocessing/prepare_real_data.py`

Add new function to resample DEM to match orthophoto:

```python
def resample_dem_to_ortho(
    dem_path: Path,
    ortho_path: Path,
    output_path: Path,
) -> None:
    """Resample DEM to match orthophoto resolution and extent.

    Preserves real elevation values (meters). Only resamples spatially.

    Args:
        dem_path: Path to source DEM GeoTIFF.
        ortho_path: Path to reference orthophoto.
        output_path: Path to save resampled DEM.
    """
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ortho_path) as ortho:
        target_crs = ortho.crs
        target_transform = ortho.transform
        target_width = ortho.width
        target_height = ortho.height

    with rasterio.open(dem_path) as dem:
        dem_resampled = np.zeros((target_height, target_width), dtype=np.float32)

        reproject(
            source=rasterio.band(dem, 1),
            destination=dem_resampled,
            src_transform=dem.transform,
            src_crs=dem.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

        # Handle nodata (typically -32767)
        nodata_mask = dem_resampled < -1000
        if nodata_mask.any():
            valid_median = np.nanmedian(dem_resampled[~nodata_mask])
            dem_resampled[nodata_mask] = valid_median

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=target_height,
        width=target_width,
        count=1,
        dtype=np.float32,
        crs=target_crs,
        transform=target_transform,
        compress="lzw",
    ) as dst:
        dst.write(dem_resampled, 1)

    logger.info(
        f"Resampled DEM: {dem_path.name} -> {output_path.name} "
        f"({target_width}x{target_height}, elevation range: "
        f"[{dem_resampled.min():.1f}, {dem_resampled.max():.1f}]m)"
    )
```

#### 2. Update DEM handling in prepare_real_dataset
**File**: `src/grave_border_detection/preprocessing/prepare_real_data.py`

Replace the `shutil.copy2` call (around line 259) with `resample_dem_to_ortho`:

```python
# Instead of: shutil.copy2(dem_src, dem_dst)
resample_dem_to_ortho(
    dem_path=dem_src,
    ortho_path=ortho_out_path,
    output_path=dem_dst,
)
```

#### 3. Update main block to enable DEM processing
**File**: `src/grave_border_detection/preprocessing/prepare_real_data.py`

Change line 305 from `dem_dir=None` to actually process DEMs:

```python
split = prepare_real_dataset(
    ortho_dir=ortho_dir,
    annotation_dir=annotation_dir,
    dem_dir=ortho_dir,  # DEMs are in same directory as orthophotos
    output_dir=output_dir,
    convert_rgba_to_rgb=True,
)
```

### Success Criteria:

#### Automated Verification:
- [x] `uv run task lint` passes
- [x] `uv run task test` passes

#### Manual Verification:
- [ ] Run: `uv run python -m grave_border_detection.preprocessing.prepare_real_data`
- [ ] Check `data/real/dems/cemetery_01_dem.tif` exists
- [ ] Verify elevation values are real meters (e.g., 430-450m range, not 0-255)
- [ ] Verify DEM dimensions match orthophoto (e.g., 4096×3920 for cemetery_01)

**Note**: This phase is run manually once when new DEM data arrives. After running, the raw resampled DEMs are stored and used by all subsequent training runs.

**Status**: Code complete. Manual verification pending (requires proper DEM TIFFs for cemeteries 03-10).

---

## Phase 2: Create DEM Preprocessing Module

### Overview
Create a dedicated module with normalization methods. These will be applied at runtime to full cemetery DEMs.

### Changes Required:

#### 1. Create preprocessing module
**File**: `src/grave_border_detection/preprocessing/dem_normalization.py` (new file)

```python
"""DEM normalization methods for grave border detection.

These methods are designed to be applied to FULL cemetery DEMs before tiling,
not to individual tiles. This ensures methods like local_height have full
spatial context for accurate ground estimation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import percentile_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NormalizationMethod(str, Enum):
    """Available DEM normalization methods."""

    ZSCORE = "zscore"
    PERCENTILE_CLIP = "percentile_clip"
    LOCAL_HEIGHT = "local_height"
    SLOPE = "slope"
    ROBUST_ZSCORE = "robust_zscore"


@dataclass
class ZScoreConfig:
    """Config for z-score normalization."""

    epsilon: float = 1e-8


@dataclass
class PercentileClipConfig:
    """Config for percentile clipping."""

    lower: float = 2.0
    upper: float = 98.0


@dataclass
class LocalHeightConfig:
    """Config for local height above ground.

    Attributes:
        window_size: Size of window for ground estimation (pixels).
            At 0.02m resolution: 128px = 2.5m, 256px = 5m
        ground_percentile: Percentile for ground level (lower = more conservative).
        max_height: Maximum height above ground to preserve (meters).
            Heights above this are clipped. Typical grave heights: 0.1-0.5m
    """

    window_size: int = 128
    ground_percentile: float = 20.0
    max_height: float = 0.5


@dataclass
class SlopeConfig:
    """Config for slope calculation.

    Attributes:
        pixel_size: Size of each pixel in meters (for gradient calculation).
    """

    pixel_size: float = 0.02


@dataclass
class RobustZScoreConfig:
    """Config for robust z-score (median/IQR)."""

    epsilon: float = 1e-8


def normalize_zscore(
    dem: "NDArray[np.float32]",
    config: ZScoreConfig | None = None,
) -> "NDArray[np.float32]":
    """Z-score normalization: (x - mean) / std.

    Simple global normalization. Buildings will dominate the std,
    compressing grave-scale details.
    """
    config = config or ZScoreConfig()
    mean = dem.mean()
    std = dem.std()
    return (dem - mean) / (std + config.epsilon)


def normalize_percentile_clip(
    dem: "NDArray[np.float32]",
    config: PercentileClipConfig | None = None,
) -> "NDArray[np.float32]":
    """Clip to percentiles, scale to [0, 1].

    Removes extreme outliers (buildings) by clipping to percentile range.
    Does not remove terrain slope.
    """
    config = config or PercentileClipConfig()
    p_low = np.percentile(dem, config.lower)
    p_high = np.percentile(dem, config.upper)
    clipped = np.clip(dem, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low + 1e-8)


def normalize_local_height(
    dem: "NDArray[np.float32]",
    config: LocalHeightConfig | None = None,
) -> "NDArray[np.float32]":
    """Compute height above local ground level.

    Best method for grave detection:
    1. Estimates local ground using low percentile in neighborhood
    2. Subtracts ground to remove terrain slope
    3. Clips to max_height to remove buildings
    4. Normalizes to [0, 1]

    IMPORTANT: Must be applied to full cemetery DEM, not tiles,
    so that ground estimation has full spatial context.
    """
    config = config or LocalHeightConfig()

    # Estimate local ground level using percentile filter
    local_ground = percentile_filter(
        dem,
        percentile=config.ground_percentile,
        size=config.window_size,
    )

    # Compute height above ground
    height_above_ground = dem - local_ground

    # Clip negative values (below ground) and values above max_height
    clipped = np.clip(height_above_ground, 0, config.max_height)

    # Normalize to [0, 1]
    return clipped / config.max_height


def normalize_slope(
    dem: "NDArray[np.float32]",
    config: SlopeConfig | None = None,
) -> "NDArray[np.float32]":
    """Compute slope magnitude, normalized to [0, 1].

    Useful for edge detection - grave borders appear as slope discontinuities.
    Loses absolute height information.
    """
    config = config or SlopeConfig()

    # Compute gradients
    dy, dx = np.gradient(dem, config.pixel_size)

    # Compute slope magnitude
    slope = np.sqrt(dx**2 + dy**2)

    # Normalize using 99th percentile to handle outliers
    p99 = np.percentile(slope, 99)
    return np.clip(slope / (p99 + 1e-8), 0, 1).astype(np.float32)


def normalize_robust_zscore(
    dem: "NDArray[np.float32]",
    config: RobustZScoreConfig | None = None,
) -> "NDArray[np.float32]":
    """Robust z-score using median and IQR.

    More resistant to outliers (buildings) than standard z-score.
    """
    config = config or RobustZScoreConfig()
    median = np.median(dem)
    q75, q25 = np.percentile(dem, [75, 25])
    iqr = q75 - q25
    return (dem - median) / (iqr + config.epsilon)


def normalize_dem(
    dem: "NDArray[np.float32]",
    method: str | NormalizationMethod,
    **kwargs,
) -> "NDArray[np.float32]":
    """Apply normalization method to DEM.

    Args:
        dem: Input DEM array (H, W) with elevation in meters.
        method: Normalization method name.
        **kwargs: Method-specific parameters.

    Returns:
        Normalized DEM array (H, W).

    Example:
        >>> normalized = normalize_dem(
        ...     dem,
        ...     method="local_height",
        ...     window_size=128,
        ...     max_height=0.5,
        ... )
    """
    method = NormalizationMethod(method)

    if method == NormalizationMethod.ZSCORE:
        config = ZScoreConfig(**kwargs) if kwargs else None
        return normalize_zscore(dem, config)

    if method == NormalizationMethod.PERCENTILE_CLIP:
        config = PercentileClipConfig(**kwargs) if kwargs else None
        return normalize_percentile_clip(dem, config)

    if method == NormalizationMethod.LOCAL_HEIGHT:
        config = LocalHeightConfig(**kwargs) if kwargs else None
        return normalize_local_height(dem, config)

    if method == NormalizationMethod.SLOPE:
        config = SlopeConfig(**kwargs) if kwargs else None
        return normalize_slope(dem, config)

    if method == NormalizationMethod.ROBUST_ZSCORE:
        config = RobustZScoreConfig(**kwargs) if kwargs else None
        return normalize_robust_zscore(dem, config)

    msg = f"Unknown normalization method: {method}"
    raise ValueError(msg)
```

#### 2. Update preprocessing __init__.py
**File**: `src/grave_border_detection/preprocessing/__init__.py`

```python
"""Preprocessing modules."""

from grave_border_detection.preprocessing.dem_normalization import (
    NormalizationMethod,
    normalize_dem,
)

__all__ = ["NormalizationMethod", "normalize_dem"]
```

### Success Criteria:

#### Automated Verification:
- [x] `uv run task lint` passes
- [x] `uv run task test` passes

#### Manual Verification:
- [x] Import works: `from grave_border_detection.preprocessing import normalize_dem`
- [x] Test in Python REPL with sample array

**Status**: Complete.

---

## Phase 3: Runtime Normalization in DataModule

### Overview
Integrate normalization into DataModule.setup() so it runs on full DEMs before tiling. The normalized DEMs are cached in memory for the duration of training.

### Changes Required:

#### 1. Update data configs
**File**: `configs/data/real.yaml`

Add DEM normalization configuration:

```yaml
# Existing fields...
use_dem: true
input_channels: 4

# NEW: DEM normalization configuration
dem:
  normalization:
    method: local_height  # zscore, percentile_clip, local_height, slope, robust_zscore

    # Method-specific parameters (only the selected method's params are used)
    zscore:
      epsilon: 1.0e-8

    percentile_clip:
      lower: 2.0
      upper: 98.0

    local_height:
      window_size: 128      # pixels (~2.5m at 0.02m resolution)
      ground_percentile: 20.0
      max_height: 0.5       # meters (clip heights above this)

    slope:
      pixel_size: 0.02      # meters per pixel

    robust_zscore:
      epsilon: 1.0e-8
```

**File**: `configs/data/synthetic.yaml`

Add same structure (synthetic data may have different defaults).

#### 2. Update GraveDataModule to normalize at setup
**File**: `src/grave_border_detection/data/datamodule.py`

Add imports and new attributes:

```python
from grave_border_detection.preprocessing import normalize_dem

# In __init__, add new parameters:
def __init__(
    self,
    # ... existing params ...
    dem_normalization_method: str = "local_height",
    dem_normalization_params: dict | None = None,
) -> None:
    # ... existing code ...
    self.dem_normalization_method = dem_normalization_method
    self.dem_normalization_params = dem_normalization_params or {}

    # Cache for normalized full-cemetery DEMs
    self._normalized_dem_cache: dict[str, NDArray[np.float32]] = {}
```

Add method to load and normalize full DEM:

```python
def _load_and_normalize_dem(self, cemetery_id: str) -> NDArray[np.float32] | None:
    """Load full cemetery DEM and apply normalization.

    Results are cached in memory for the duration of training.
    """
    if cemetery_id in self._normalized_dem_cache:
        return self._normalized_dem_cache[cemetery_id]

    if self.dems_dir is None:
        return None

    dem_files = list(self.dems_dir.glob(f"{cemetery_id}*.tif"))
    if not dem_files:
        return None

    dem_path = dem_files[0]

    with rasterio.open(dem_path) as src:
        dem_raw = src.read(1).astype(np.float32)

    # Apply normalization to full DEM
    dem_normalized = normalize_dem(
        dem_raw,
        method=self.dem_normalization_method,
        **self.dem_normalization_params,
    )

    self._normalized_dem_cache[cemetery_id] = dem_normalized
    return dem_normalized
```

Update setup() to pre-load normalized DEMs:

```python
def setup(self, stage: str | None = None) -> None:
    """Set up datasets for each stage."""
    if stage in ("fit", None):
        # Pre-load and normalize DEMs for all cemeteries
        if self.use_dem:
            all_cemeteries = (
                self.train_cemeteries + self.val_cemeteries + self.test_cemeteries
            )
            for cemetery_id in all_cemeteries:
                self._load_and_normalize_dem(cemetery_id)

        # ... rest of existing setup code ...
```

#### 3. Update GraveDataset to use pre-normalized DEMs
**File**: `src/grave_border_detection/data/dataset.py`

Update constructor to accept normalized DEM cache:

```python
def __init__(
    self,
    tile_refs: list[TileReference],
    tile_size: int = 512,
    use_dem: bool = True,
    normalized_dem_cache: dict[str, NDArray[np.float32]] | None = None,
    transform: A.Compose | None = None,
    normalize_rgb: bool = True,
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> None:
    # ... existing code ...
    self.normalized_dem_cache = normalized_dem_cache or {}
```

Update `__getitem__` to read from cache instead of file + normalize:

```python
# Replace the current DEM loading block (lines 164-170) with:
dem: NDArray[np.float32] | None = None
if self.use_dem and ref.dem_path is not None:
    # Get pre-normalized DEM from cache
    normalized_full_dem = self.normalized_dem_cache.get(ref.cemetery_id)

    if normalized_full_dem is not None:
        # Extract tile from pre-normalized full DEM
        tile = ref.tile_info
        dem = normalized_full_dem[
            tile.y : tile.y + self.tile_size,
            tile.x : tile.x + self.tile_size,
        ].copy()

        # Pad if needed (for edge tiles)
        if dem.shape != (self.tile_size, self.tile_size):
            padded = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            padded[: dem.shape[0], : dem.shape[1]] = dem
            dem = padded
```

#### 4. Update DataModule._build_dataset to pass cache
**File**: `src/grave_border_detection/data/datamodule.py`

```python
def _build_dataset(
    self,
    cemetery_ids: list[str],
    transform: Any | None = None,
) -> GraveDataset:
    # ... existing tile_refs building code ...

    return GraveDataset(
        tile_refs=tile_refs,
        tile_size=self.tile_size,
        use_dem=self.use_dem,
        normalized_dem_cache=self._normalized_dem_cache,  # NEW
        transform=transform,
        normalize_rgb=self.normalize_rgb,
        rgb_mean=self.rgb_mean,
        rgb_std=self.rgb_std,
    )
```

#### 5. Update train.py to pass config
**File**: `src/grave_border_detection/train.py`

Update DataModule instantiation (around line 122):

```python
# Extract DEM normalization config
dem_norm_method = cfg.data.dem.normalization.method
dem_norm_params = dict(cfg.data.dem.normalization.get(dem_norm_method, {}))

data_module = GraveDataModule(
    data_root=cfg.data.root,
    train_cemeteries=list(cfg.data.train_cemeteries),
    val_cemeteries=list(cfg.data.val_cemeteries),
    test_cemeteries=list(cfg.data.get("test_cemeteries", [])),
    tile_size=cfg.data.tiling.tile_size,
    overlap=cfg.data.tiling.overlap,
    min_mask_coverage=cfg.data.tiling.min_mask_coverage,
    use_dem=cfg.data.use_dem,
    dem_normalization_method=dem_norm_method,
    dem_normalization_params=dem_norm_params,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
    normalize_rgb=cfg.data.get("normalize_rgb", True),
)
```

### Success Criteria:

#### Automated Verification:
- [x] `uv run task lint` passes
- [x] `uv run task test` passes
- [ ] Config override works: `uv run python -m grave_border_detection.train data.dem.normalization.method=percentile_clip`

#### Manual Verification:
- [ ] Training runs with default config (local_height)
- [ ] Training runs with `data.dem.normalization.method=zscore`
- [ ] Training runs with `data.dem.normalization.method=slope`
- [ ] Check that different methods produce different loss curves

**Status**: Code complete. Manual training verification pending.

---

## Phase 4: Tests and HPO Integration

### Overview
Add unit tests for preprocessing methods and enable HPO to search over normalization parameters.

### Changes Required:

#### 1. Create test directory and file
**File**: `tests/unit/preprocessing/__init__.py` (empty file)

**File**: `tests/unit/preprocessing/test_dem_normalization.py`

```python
"""Tests for DEM normalization methods."""

import numpy as np
import pytest

from grave_border_detection.preprocessing.dem_normalization import (
    LocalHeightConfig,
    NormalizationMethod,
    PercentileClipConfig,
    normalize_dem,
    normalize_local_height,
    normalize_percentile_clip,
    normalize_slope,
    normalize_zscore,
    normalize_robust_zscore,
)


@pytest.fixture
def sample_dem() -> np.ndarray:
    """Create sample DEM with building and grave features."""
    # Base elevation ~440m
    dem = np.full((256, 256), 440.0, dtype=np.float32)

    # Add terrain slope (1m across image)
    for i in range(256):
        dem[i, :] += i * (1.0 / 256)

    # Add "building" in corner (10m tall)
    dem[20:60, 20:60] = 450.0

    # Add subtle "grave" (30cm raised)
    dem[150:170, 150:180] = 440.3

    return dem


class TestZScore:
    def test_mean_zero(self, sample_dem: np.ndarray) -> None:
        result = normalize_zscore(sample_dem)
        assert result.mean() == pytest.approx(0.0, abs=1e-5)

    def test_std_one(self, sample_dem: np.ndarray) -> None:
        result = normalize_zscore(sample_dem)
        assert result.std() == pytest.approx(1.0, abs=1e-5)

    def test_shape_preserved(self, sample_dem: np.ndarray) -> None:
        result = normalize_zscore(sample_dem)
        assert result.shape == sample_dem.shape


class TestPercentileClip:
    def test_output_range(self, sample_dem: np.ndarray) -> None:
        result = normalize_percentile_clip(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_percentiles(self, sample_dem: np.ndarray) -> None:
        config = PercentileClipConfig(lower=5.0, upper=95.0)
        result = normalize_percentile_clip(sample_dem, config)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestLocalHeight:
    def test_output_range(self, sample_dem: np.ndarray) -> None:
        result = normalize_local_height(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_building_clipped(self, sample_dem: np.ndarray) -> None:
        config = LocalHeightConfig(max_height=1.0)
        result = normalize_local_height(sample_dem, config)
        # Building (10m above ground) should be clipped to 1.0
        assert result[40, 40] == pytest.approx(1.0, abs=0.1)

    def test_grave_preserved(self, sample_dem: np.ndarray) -> None:
        config = LocalHeightConfig(max_height=1.0, window_size=64)
        result = normalize_local_height(sample_dem, config)
        # Grave (0.3m) should be ~0.3 (normalized to max_height=1.0)
        assert 0.1 < result[160, 165] < 0.5

    def test_ground_near_zero(self, sample_dem: np.ndarray) -> None:
        config = LocalHeightConfig(window_size=64)
        result = normalize_local_height(sample_dem, config)
        # Flat ground away from features should be near 0
        assert result[200, 200] < 0.2


class TestSlope:
    def test_output_range(self, sample_dem: np.ndarray) -> None:
        result = normalize_slope(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_edges_detected(self, sample_dem: np.ndarray) -> None:
        result = normalize_slope(sample_dem)
        # Edge of building should have higher slope than interior
        edge_slope = result[20, 40]  # building edge
        interior_slope = result[40, 40]  # building interior
        assert edge_slope > interior_slope


class TestRobustZScore:
    def test_shape_preserved(self, sample_dem: np.ndarray) -> None:
        result = normalize_robust_zscore(sample_dem)
        assert result.shape == sample_dem.shape


class TestNormalizeDem:
    @pytest.mark.parametrize("method", list(NormalizationMethod))
    def test_all_methods_work(self, sample_dem: np.ndarray, method: NormalizationMethod) -> None:
        result = normalize_dem(sample_dem, method)
        assert result.shape == sample_dem.shape
        assert np.isfinite(result).all()

    def test_unknown_method_raises(self, sample_dem: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_dem(sample_dem, "invalid_method")

    def test_kwargs_passed(self, sample_dem: np.ndarray) -> None:
        # Should not raise with valid kwargs
        result = normalize_dem(
            sample_dem,
            "local_height",
            window_size=64,
            max_height=1.0,
        )
        assert result.shape == sample_dem.shape
```

#### 2. Update HPO config
**File**: `configs/hpo/default.yaml`

Add DEM normalization to search space:

```yaml
search_space:
  # ... existing params ...

  # DEM normalization method
  dem_normalization_method:
    type: categorical
    choices:
      - zscore
      - percentile_clip
      - local_height

  # Local height parameters (used when method=local_height)
  dem_local_height_window_size:
    type: categorical
    choices: [64, 128, 256]

  dem_local_height_max_height:
    type: float
    low: 0.3
    high: 1.0
    step: 0.1

  dem_local_height_ground_percentile:
    type: float
    low: 10.0
    high: 30.0
    step: 5.0

  # Percentile clip parameters (used when method=percentile_clip)
  dem_percentile_clip_upper:
    type: float
    low: 90.0
    high: 99.0
    step: 1.0
```

#### 3. Update HPO objective function
**File**: `src/grave_border_detection/hpo.py`

Update the suggest function to handle DEM normalization parameters and build appropriate config overrides:

```python
def suggest_config(trial: optuna.Trial, search_space: DictConfig) -> dict[str, Any]:
    """Suggest hyperparameters for a trial."""
    suggestions = {}

    # ... existing suggestion code ...

    # Handle DEM normalization
    if "dem_normalization_method" in search_space:
        method = trial.suggest_categorical(
            "dem_normalization_method",
            search_space.dem_normalization_method.choices,
        )
        suggestions["data.dem.normalization.method"] = method

        # Suggest method-specific params
        if method == "local_height":
            if "dem_local_height_window_size" in search_space:
                suggestions["data.dem.normalization.local_height.window_size"] = (
                    trial.suggest_categorical(
                        "dem_local_height_window_size",
                        search_space.dem_local_height_window_size.choices,
                    )
                )
            if "dem_local_height_max_height" in search_space:
                suggestions["data.dem.normalization.local_height.max_height"] = (
                    trial.suggest_float(
                        "dem_local_height_max_height",
                        search_space.dem_local_height_max_height.low,
                        search_space.dem_local_height_max_height.high,
                        step=search_space.dem_local_height_max_height.get("step"),
                    )
                )

        elif method == "percentile_clip":
            if "dem_percentile_clip_upper" in search_space:
                suggestions["data.dem.normalization.percentile_clip.upper"] = (
                    trial.suggest_float(
                        "dem_percentile_clip_upper",
                        search_space.dem_percentile_clip_upper.low,
                        search_space.dem_percentile_clip_upper.high,
                        step=search_space.dem_percentile_clip_upper.get("step"),
                    )
                )

    return suggestions
```

### Success Criteria:

#### Automated Verification:
- [x] `uv run task lint` passes
- [x] `uv run task test` passes
- [x] All new tests pass: `uv run pytest tests/unit/preprocessing/ -v` (28 tests)

#### Manual Verification:
- [ ] Run HPO: `uv run python -m grave_border_detection.hpo`
- [ ] Verify different normalization methods appear in trial logs
- [ ] Check Optuna dashboard shows dem_normalization_method as a parameter

**Status**: Unit tests complete. HPO config integration optional - can be added when needed for hyperparameter search.

---

## Summary

| Phase | Type | What | When to Run |
|-------|------|------|-------------|
| 1 | Manual script | Extract & resample raw DEMs | Once, when new data arrives |
| 2 | Code | Normalization module | Implement once |
| 3 | Code | Runtime normalization in DataModule | Implement once |
| 4 | Code | Tests + HPO integration | Implement once |

After implementation, the workflow is:
1. Get new DEM TIFFs → run extraction script (Phase 1)
2. Training/HPO runs apply normalization from config at setup time
3. HPO searches over methods and parameters automatically

## References

- Current DEM handling: `src/grave_border_detection/data/dataset.py:164-170`
- Config loading: `src/grave_border_detection/train.py:104-147`
- Analysis script: `scripts/analyze_dem_normalization.py`
- HPO implementation: `src/grave_border_detection/hpo.py`
