#!/usr/bin/env python3
"""Analyze and compare different DEM normalization approaches for grave detection.

This script visualizes how different normalization methods handle buildings
vs. preserving subtle grave height variations.

Methods compared:
1. Raw DEM (original values)
2. Current z-score normalization (per-tile mean/std)
3. Percentile clipping (clip to 95th percentile)
4. Local percentile normalization (height above local ground)
5. Slope magnitude (gradient-based)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter, uniform_filter


def load_dem_tile(
    dem_path: Path,
    x_offset: int,
    y_offset: int,
    tile_size: int,
) -> NDArray[np.float32]:
    """Load a tile from a DEM file."""
    with rasterio.open(dem_path) as src:
        window = rasterio.windows.Window(x_offset, y_offset, tile_size, tile_size)
        dem = src.read(1, window=window).astype(np.float32)
    return dem


def load_ortho_tile(
    ortho_path: Path,
    x_offset: int,
    y_offset: int,
    tile_size: int,
) -> NDArray[np.uint8]:
    """Load a tile from an orthophoto file."""
    with rasterio.open(ortho_path) as src:
        window = rasterio.windows.Window(x_offset, y_offset, tile_size, tile_size)
        # Read RGB (first 3 bands)
        rgb = src.read([1, 2, 3], window=window)
        # Transpose to HWC
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def normalize_zscore(dem: NDArray[np.float32]) -> NDArray[np.float32]:
    """Current normalization: per-tile z-score."""
    return (dem - dem.mean()) / (dem.std() + 1e-8)


def normalize_percentile_clip(
    dem: NDArray[np.float32],
    lower: float = 2.0,
    upper: float = 95.0,
) -> NDArray[np.float32]:
    """Clip to percentiles, then normalize to 0-1."""
    p_low = np.percentile(dem, lower)
    p_high = np.percentile(dem, upper)
    clipped = np.clip(dem, p_low, p_high)
    # Normalize to 0-1
    return (clipped - p_low) / (p_high - p_low + 1e-8)


def normalize_local_percentile(
    dem: NDArray[np.float32],
    window_size: int = 128,
    ground_percentile: float = 20.0,
    max_height: float = 1.0,
) -> NDArray[np.float32]:
    """Compute height above local ground level.

    Uses a percentile filter to estimate local ground, then computes
    height above ground and clips to max_height.

    Args:
        dem: Input DEM array.
        window_size: Size of the local window in pixels.
        ground_percentile: Percentile to use for ground estimation (lower = more conservative).
        max_height: Maximum height above ground to preserve (meters).
    """
    # Estimate local ground level using percentile filter
    local_ground = percentile_filter(dem, percentile=ground_percentile, size=window_size)

    # Compute height above ground
    height_above_ground = dem - local_ground

    # Clip to max height and normalize to 0-1
    clipped = np.clip(height_above_ground, 0, max_height)
    return clipped / max_height


def compute_slope(dem: NDArray[np.float32], pixel_size: float = 0.05) -> NDArray[np.float32]:
    """Compute slope magnitude from DEM.

    Args:
        dem: Input DEM array.
        pixel_size: Size of each pixel in meters.
    """
    # Compute gradients
    dy, dx = np.gradient(dem, pixel_size)

    # Compute slope magnitude
    slope = np.sqrt(dx**2 + dy**2)

    # Normalize to 0-1 using percentile clipping
    p99 = np.percentile(slope, 99)
    slope_normalized = np.clip(slope / (p99 + 1e-8), 0, 1)

    return slope_normalized


def normalize_local_zscore(
    dem: NDArray[np.float32],
    window_size: int = 64,
) -> NDArray[np.float32]:
    """Local z-score normalization using a moving window.

    This normalizes based on local statistics, reducing the impact of
    large-scale terrain variation and buildings.
    """
    local_mean = uniform_filter(dem, size=window_size)
    local_sq_mean = uniform_filter(dem**2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0) + 1e-8)

    return (dem - local_mean) / local_std


def visualize_normalizations(
    ortho: NDArray[np.uint8],
    dem: NDArray[np.float32],
    output_path: Path | None = None,
    title_prefix: str = "",
) -> None:
    """Create a figure comparing all normalization approaches."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Original data and basic normalizations
    # Orthophoto
    axes[0, 0].imshow(ortho)
    axes[0, 0].set_title("Orthophoto (RGB)")
    axes[0, 0].axis("off")

    # Raw DEM
    im1 = axes[0, 1].imshow(dem, cmap="viridis")
    axes[0, 1].set_title(f"Raw DEM\nRange: [{dem.min():.1f}, {dem.max():.1f}]m")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Current z-score
    zscore = normalize_zscore(dem)
    im2 = axes[0, 2].imshow(zscore, cmap="viridis", vmin=-3, vmax=3)
    axes[0, 2].set_title("Current: Z-score\n(per-tile mean/std)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Percentile clip
    pclip = normalize_percentile_clip(dem, lower=2, upper=95)
    im3 = axes[0, 3].imshow(pclip, cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("Percentile Clip (2-95%)\n(buildings saturated)")
    axes[0, 3].axis("off")
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: Advanced normalizations
    # Local percentile (height above ground)
    local_pct = normalize_local_percentile(dem, window_size=128, max_height=1.0)
    im4 = axes[1, 0].imshow(local_pct, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("Local Height Above Ground\n(window=128px, max=1m)")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Local percentile with larger window
    local_pct2 = normalize_local_percentile(dem, window_size=256, max_height=0.5)
    im5 = axes[1, 1].imshow(local_pct2, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Local Height Above Ground\n(window=256px, max=0.5m)")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    # Slope
    slope = compute_slope(dem)
    im6 = axes[1, 2].imshow(slope, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("Slope Magnitude\n(gradient-based)")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Local z-score
    local_z = normalize_local_zscore(dem, window_size=64)
    im7 = axes[1, 3].imshow(local_z, cmap="viridis", vmin=-3, vmax=3)
    axes[1, 3].set_title("Local Z-score\n(window=64px)")
    axes[1, 3].axis("off")
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)

    plt.suptitle(f"{title_prefix}DEM Normalization Comparison", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close(fig)


def visualize_histograms(
    dem: NDArray[np.float32],
    output_path: Path | None = None,
    title_prefix: str = "",
) -> None:
    """Show histograms of different normalizations to understand value distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Raw DEM
    axes[0, 0].hist(dem.flatten(), bins=100, alpha=0.7)
    axes[0, 0].set_title("Raw DEM")
    axes[0, 0].set_xlabel("Height (m)")

    # Z-score
    zscore = normalize_zscore(dem)
    axes[0, 1].hist(zscore.flatten(), bins=100, alpha=0.7)
    axes[0, 1].set_title("Z-score")
    axes[0, 1].axvline(x=-3, color="r", linestyle="--", alpha=0.5)
    axes[0, 1].axvline(x=3, color="r", linestyle="--", alpha=0.5)

    # Percentile clip
    pclip = normalize_percentile_clip(dem, lower=2, upper=95)
    axes[0, 2].hist(pclip.flatten(), bins=100, alpha=0.7)
    axes[0, 2].set_title("Percentile Clip (2-95%)")

    # Local height above ground
    local_pct = normalize_local_percentile(dem, window_size=128, max_height=1.0)
    axes[1, 0].hist(local_pct.flatten(), bins=100, alpha=0.7)
    axes[1, 0].set_title("Local Height Above Ground")

    # Slope
    slope = compute_slope(dem)
    axes[1, 1].hist(slope.flatten(), bins=100, alpha=0.7)
    axes[1, 1].set_title("Slope Magnitude")

    # Local z-score
    local_z = normalize_local_zscore(dem, window_size=64)
    axes[1, 2].hist(local_z.flatten(), bins=100, alpha=0.7)
    axes[1, 2].set_title("Local Z-score")
    axes[1, 2].axvline(x=-3, color="r", linestyle="--", alpha=0.5)
    axes[1, 2].axvline(x=3, color="r", linestyle="--", alpha=0.5)

    plt.suptitle(f"{title_prefix}Value Distributions", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram to {output_path}")
        plt.close(fig)


def find_tile_with_building(
    dem_path: Path,
    tile_size: int = 512,
    height_threshold: float = 3.0,
) -> tuple[int, int] | None:
    """Find a tile that likely contains a building based on height variation."""
    with rasterio.open(dem_path) as src:
        width, height = src.width, src.height

    best_tile = None
    best_range = 0.0

    for y in range(0, height - tile_size, tile_size):
        for x in range(0, width - tile_size, tile_size):
            tile = load_dem_tile(dem_path, x, y, tile_size)
            height_range = tile.max() - tile.min()

            # Look for tiles with significant height variation (likely buildings)
            if height_range > height_threshold and height_range > best_range:
                best_range = height_range
                best_tile = (x, y)

    return best_tile


def load_dem_with_nodata_handling(
    dem_path: Path, tile_size: int, x: int, y: int
) -> NDArray[np.float32]:
    """Load DEM tile and handle nodata values."""
    dem = load_dem_tile(dem_path, x, y, tile_size)
    # Replace nodata values (typically -32767) with NaN, then fill with local median
    nodata_mask = dem < -1000  # Assume anything below -1000m is nodata
    if nodata_mask.any():
        valid_median = np.median(dem[~nodata_mask])
        dem[nodata_mask] = valid_median
    return dem


def main() -> None:
    """Run the DEM normalization analysis."""
    output_dir = Path("outputs/dem_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = 512

    # Use source DEM with real elevation values (cemetery 01)
    # The DEMs in data/real/dems/ are incorrectly scaled to 0-255
    source_dem = Path(
        "data/external/goetiff-orthos-ki-training-data/4k/01-251107v08epsg3857_res0.0046m-DEM.tif"
    )
    ortho_path = Path("data/real/orthophotos/cemetery_01_ortho.tif")

    if not source_dem.exists():
        print(f"Source DEM not found: {source_dem}")
        return

    print("Analyzing cemetery_01 with SOURCE DEM (real elevation values)...")
    print("Note: The DEMs in data/real/dems/ are incorrectly scaled to 0-255")

    # The source DEM is lower resolution than the ortho, so we need to find
    # a good region with building/height variation
    with rasterio.open(source_dem) as src:
        dem_width, dem_height = src.width, src.height
        print(f"  Source DEM size: {dem_width}x{dem_height}")

    # Use center tile of the DEM
    x_offset = max(0, (dem_width - tile_size) // 2)
    y_offset = max(0, (dem_height - tile_size) // 2)

    # Adjust tile size if DEM is smaller
    actual_tile = min(tile_size, dem_width, dem_height)

    # Load DEM with nodata handling
    dem = load_dem_with_nodata_handling(source_dem, actual_tile, x_offset, y_offset)

    print(f"  DEM tile range: [{dem.min():.2f}, {dem.max():.2f}] m")
    print(f"  DEM tile std: {dem.std():.2f} m")

    # Load corresponding ortho tile (approximate - different resolution)
    # Just use a placeholder for visualization since resolutions differ
    with rasterio.open(ortho_path) as src:
        # Scale coordinates to ortho resolution
        scale_x = src.width / dem_width
        scale_y = src.height / dem_height
        ortho_x = int(x_offset * scale_x)
        ortho_y = int(y_offset * scale_y)
        ortho = load_ortho_tile(ortho_path, ortho_x, ortho_y, actual_tile)

    # Create visualizations
    visualize_normalizations(
        ortho,
        dem,
        output_path=output_dir / "cemetery_01_source_dem_normalization.png",
        title_prefix="cemetery_01 (SOURCE DEM): ",
    )

    visualize_histograms(
        dem,
        output_path=output_dir / "cemetery_01_source_dem_histograms.png",
        title_prefix="cemetery_01 (SOURCE DEM): ",
    )

    # Also analyze different tiles to find one with buildings
    print("\nSearching for tiles with height variation (possible buildings)...")
    best_tile = find_tile_with_building(source_dem, actual_tile, height_threshold=2.0)

    if best_tile:
        x2, y2 = best_tile
        dem2 = load_dem_with_nodata_handling(source_dem, actual_tile, x2, y2)

        # Scale to ortho coords
        ortho_x2 = int(x2 * scale_x)
        ortho_y2 = int(y2 * scale_y)
        ortho2 = load_ortho_tile(ortho_path, ortho_x2, ortho_y2, actual_tile)

        print(f"  Found high-variation tile at ({x2}, {y2})")
        print(f"  DEM range: [{dem2.min():.2f}, {dem2.max():.2f}] m")

        visualize_normalizations(
            ortho2,
            dem2,
            output_path=output_dir / "cemetery_01_source_dem_building_tile.png",
            title_prefix="cemetery_01 (building tile): ",
        )

        visualize_histograms(
            dem2,
            output_path=output_dir / "cemetery_01_source_dem_building_histograms.png",
            title_prefix="cemetery_01 (building tile): ",
        )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
