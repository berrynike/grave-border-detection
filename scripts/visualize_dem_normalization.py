"""Visualize DEM normalization methods on real cemetery data.

Usage:
    uv run python scripts/visualize_dem_normalization.py
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from grave_border_detection.preprocessing.dem_normalization import (
    LocalHeightConfig,
    NormalizationMethod,
    normalize_dem,
)


def load_dem(path: Path) -> np.ndarray:
    """Load DEM from GeoTIFF file."""
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        # Handle nodata values
        nodata = src.nodata
        if nodata is not None:
            mask = dem == nodata
            if mask.any():
                dem[mask] = np.nanmedian(dem[~mask])
        # Also handle typical nodata value
        mask = dem < -1000
        if mask.any():
            dem[mask] = np.nanmedian(dem[~mask])
    return dem


def visualize_normalization_methods(
    dem: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Create visualization comparing all normalization methods."""
    methods = [
        ("Original DEM", None),
        ("Z-Score", NormalizationMethod.ZSCORE),
        ("Percentile Clip", NormalizationMethod.PERCENTILE_CLIP),
        ("Local Height", NormalizationMethod.LOCAL_HEIGHT),
        ("Slope", NormalizationMethod.SLOPE),
        ("Robust Z-Score", NormalizationMethod.ROBUST_ZSCORE),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, method) in enumerate(methods):
        print(f"  Computing {name}...")
        ax = axes[idx]

        if method is None:
            # Original DEM
            data = dem
            vmin, vmax = np.percentile(dem, [2, 98])
            cmap = "terrain"
            label = f"Elevation (m)\nRange: [{dem.min():.1f}, {dem.max():.1f}]"
        else:
            # Apply normalization
            if method == NormalizationMethod.LOCAL_HEIGHT:
                # Use parameters suitable for cemetery data
                data = normalize_dem(
                    dem,
                    method,
                    window_size=128,
                    ground_percentile=20.0,
                    max_height=0.5,
                )
            else:
                data = normalize_dem(dem, method)

            if method in (
                NormalizationMethod.ZSCORE,
                NormalizationMethod.ROBUST_ZSCORE,
            ):
                # Z-scores can be negative, use symmetric colormap
                vmax = np.percentile(np.abs(data), 98)
                vmin = -vmax
                cmap = "RdBu_r"
                label = f"Z-score\nRange: [{data.min():.2f}, {data.max():.2f}]"
            else:
                # [0, 1] normalized outputs
                vmin, vmax = 0, 1
                cmap = "viridis"
                label = f"Normalized [0,1]\nRange: [{data.min():.2f}, {data.max():.2f}]"

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_local_height_params(
    dem: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Visualize effect of different local_height parameters."""
    # Reduced set of configs for faster generation
    configs = [
        ("window=64, max=0.5m", LocalHeightConfig(window_size=64, max_height=0.5)),
        ("window=128, max=0.5m", LocalHeightConfig(window_size=128, max_height=0.5)),
        ("window=64, max=1.0m", LocalHeightConfig(window_size=64, max_height=1.0)),
        ("window=128, max=1.0m", LocalHeightConfig(window_size=128, max_height=1.0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (name, config) in enumerate(configs):
        print(f"  Computing local_height {name}...")
        ax = axes[idx]
        data = normalize_dem(
            dem,
            NormalizationMethod.LOCAL_HEIGHT,
            window_size=config.window_size,
            ground_percentile=config.ground_percentile,
            max_height=config.max_height,
        )

        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"{title}\nLocal Height Parameter Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Run visualization on cemetery DEMs."""
    # Paths to proper DEM files (with real elevation values)
    dem_files = {
        "Cemetery 01": Path(
            "data/external/goetiff-orthos-ki-training-data/4k/"
            "01-251107v08epsg3857_res0.0046m-DEM.tif"
        ),
        "Cemetery 02": Path(
            "data/external/goetiff-orthos-ki-training-data/4k/"
            "02-251107-epsg3857-res0.008m-2.1mio-pcld_res0.03m_cleaned_DEM.tif"
        ),
    }

    output_dir = Path("outputs/dem_normalization_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dem_path in dem_files.items():
        if not dem_path.exists():
            print(f"DEM not found: {dem_path}")
            continue

        print(f"\nProcessing {name}...")
        dem = load_dem(dem_path)
        print(f"  Shape: {dem.shape}")
        print(f"  Elevation range: [{dem.min():.2f}, {dem.max():.2f}] m")

        # Generate comparison of all methods
        safe_name = name.lower().replace(" ", "_")
        visualize_normalization_methods(
            dem,
            title=f"{name} - DEM Normalization Methods",
            output_path=output_dir / f"{safe_name}_methods.png",
        )

        # Generate local_height parameter comparison
        visualize_local_height_params(
            dem,
            title=name,
            output_path=output_dir / f"{safe_name}_local_height_params.png",
        )

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
