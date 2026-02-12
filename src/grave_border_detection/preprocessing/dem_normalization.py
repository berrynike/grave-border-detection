"""DEM normalization methods for grave border detection.

These methods are designed to be applied to FULL cemetery DEMs before tiling,
not to individual tiles. This ensures methods like local_height have full
spatial context for accurate ground estimation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import percentile_filter  # type: ignore[import-untyped]

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
    cfg = config or ZScoreConfig()
    mean = float(dem.mean())
    std = float(dem.std())
    result: NDArray[np.float32] = ((dem - mean) / (std + cfg.epsilon)).astype(np.float32)
    return result


def normalize_percentile_clip(
    dem: "NDArray[np.float32]",
    config: PercentileClipConfig | None = None,
) -> "NDArray[np.float32]":
    """Clip to percentiles, scale to [0, 1].

    Removes extreme outliers (buildings) by clipping to percentile range.
    Does not remove terrain slope.
    """
    cfg = config or PercentileClipConfig()
    p_low = float(np.percentile(dem, cfg.lower))
    p_high = float(np.percentile(dem, cfg.upper))
    clipped = np.clip(dem, p_low, p_high)
    result: NDArray[np.float32] = ((clipped - p_low) / (p_high - p_low + 1e-8)).astype(np.float32)
    return result


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
    cfg = config or LocalHeightConfig()

    # Estimate local ground level using percentile filter
    local_ground: NDArray[np.float32] = percentile_filter(
        dem,
        percentile=cfg.ground_percentile,
        size=cfg.window_size,
    )

    # Compute height above ground
    height_above_ground = dem - local_ground

    # Clip negative values (below ground) and values above max_height
    clipped = np.clip(height_above_ground, 0, cfg.max_height)

    # Normalize to [0, 1]
    result: NDArray[np.float32] = (clipped / cfg.max_height).astype(np.float32)
    return result


def normalize_slope(
    dem: "NDArray[np.float32]",
    config: SlopeConfig | None = None,
) -> "NDArray[np.float32]":
    """Compute slope magnitude, normalized to [0, 1].

    Useful for edge detection - grave borders appear as slope discontinuities.
    Loses absolute height information.
    """
    cfg = config or SlopeConfig()

    # Compute gradients
    dy, dx = np.gradient(dem, cfg.pixel_size)

    # Compute slope magnitude
    slope = np.sqrt(dx**2 + dy**2)

    # Normalize using 99th percentile to handle outliers
    p99 = float(np.percentile(slope, 99))
    result: NDArray[np.float32] = np.clip(slope / (p99 + 1e-8), 0, 1).astype(np.float32)
    return result


def normalize_robust_zscore(
    dem: "NDArray[np.float32]",
    config: RobustZScoreConfig | None = None,
) -> "NDArray[np.float32]":
    """Robust z-score using median and IQR.

    More resistant to outliers (buildings) than standard z-score.
    """
    cfg = config or RobustZScoreConfig()
    median = float(np.median(dem))
    q75, q25 = np.percentile(dem, [75, 25])
    iqr = float(q75 - q25)
    result: NDArray[np.float32] = ((dem - median) / (iqr + cfg.epsilon)).astype(np.float32)
    return result


def normalize_dem(
    dem: "NDArray[np.float32]",
    method: str | NormalizationMethod,
    **kwargs: float | int,
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
    method_enum = NormalizationMethod(method)

    if method_enum == NormalizationMethod.ZSCORE:
        zscore_config = ZScoreConfig(**kwargs) if kwargs else None
        return normalize_zscore(dem, zscore_config)

    if method_enum == NormalizationMethod.PERCENTILE_CLIP:
        pclip_config = PercentileClipConfig(**kwargs) if kwargs else None
        return normalize_percentile_clip(dem, pclip_config)

    if method_enum == NormalizationMethod.LOCAL_HEIGHT:
        # Build config with explicit type handling
        if kwargs:
            local_config = LocalHeightConfig(
                window_size=int(kwargs.get("window_size", 128)),
                ground_percentile=float(kwargs.get("ground_percentile", 20.0)),
                max_height=float(kwargs.get("max_height", 0.5)),
            )
        else:
            local_config = None
        return normalize_local_height(dem, local_config)

    if method_enum == NormalizationMethod.SLOPE:
        slope_config = SlopeConfig(**kwargs) if kwargs else None
        return normalize_slope(dem, slope_config)

    if method_enum == NormalizationMethod.ROBUST_ZSCORE:
        robust_config = RobustZScoreConfig(**kwargs) if kwargs else None
        return normalize_robust_zscore(dem, robust_config)

    msg = f"Unknown normalization method: {method}"
    raise ValueError(msg)
