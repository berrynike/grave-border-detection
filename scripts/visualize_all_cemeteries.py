"""Visualize all cemeteries to compare train/val/test data quality."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Use processed data directory
DATA_DIR = Path("data/real")

# Split configuration
TRAIN_CEMETERIES = [
    "cemetery_01",
    "cemetery_02",
    "cemetery_03",
    "cemetery_04",
    "cemetery_05",
    "cemetery_06",
]
VAL_CEMETERIES = ["cemetery_07", "cemetery_08"]
TEST_CEMETERIES = ["cemetery_09", "cemetery_10"]
ALL_CEMETERIES = TRAIN_CEMETERIES + VAL_CEMETERIES + TEST_CEMETERIES


def get_split(cemetery_id: str) -> tuple[str, str]:
    """Get split name and color for cemetery."""
    if cemetery_id in TRAIN_CEMETERIES:
        return "TRAIN", "green"
    elif cemetery_id in VAL_CEMETERIES:
        return "VAL", "blue"
    else:
        return "TEST", "red"


def load_raster(path: Path) -> np.ndarray | None:
    """Load a raster file."""
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure: 10 rows x 5 cols (ortho, mask, local_height, slope, histogram)
    _fig, axes = plt.subplots(10, 5, figsize=(25, 40))

    all_stats = []

    for idx, cemetery_id in enumerate(ALL_CEMETERIES):
        print(f"Processing {cemetery_id}...")
        split, color = get_split(cemetery_id)

        # Paths
        ortho_path = DATA_DIR / "orthophotos" / f"{cemetery_id}_ortho.tif"
        mask_path = DATA_DIR / "masks" / f"{cemetery_id}_mask.tif"
        dem_lh_path = DATA_DIR / "dems" / f"{cemetery_id}_dem_local_height.tif"
        dem_slope_path = DATA_DIR / "dems" / f"{cemetery_id}_dem_slope.tif"

        stats = {"id": cemetery_id, "split": split}

        # Load orthophoto
        ax = axes[idx, 0]
        if ortho_path.exists():
            with rasterio.open(ortho_path) as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
                stats["shape"] = f"{rgb.shape[0]}x{rgb.shape[1]}"
            ax.imshow(rgb)
            ax.set_title(
                f"{cemetery_id} ({split})\nOrtho {stats['shape']}",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        else:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14, color="red")
            ax.set_title(f"{cemetery_id} ({split})\nOrtho MISSING", fontsize=9, color="red")
        ax.axis("off")

        # Load mask
        ax = axes[idx, 1]
        mask = load_raster(mask_path)
        if mask is not None:
            mask_coverage = 100 * (mask > 0).sum() / mask.size
            stats["mask_coverage"] = mask_coverage
            ax.imshow(mask, cmap="gray")
            ax.set_title(f"Mask\n{mask_coverage:.1f}% graves", fontsize=9)
        else:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14, color="red")
            ax.set_title("Mask MISSING", fontsize=9, color="red")
        ax.axis("off")

        # Load local_height DEM
        ax = axes[idx, 2]
        dem_lh = load_raster(dem_lh_path)
        if dem_lh is not None:
            stats["lh_min"] = float(dem_lh.min())
            stats["lh_max"] = float(dem_lh.max())
            stats["lh_mean"] = float(dem_lh.mean())
            stats["lh_std"] = float(dem_lh.std())
            im = ax.imshow(dem_lh, cmap="viridis", vmin=0, vmax=0.5)
            ax.set_title(
                f"Local Height\n[{stats['lh_min']:.2f}, {stats['lh_max']:.2f}] mean={stats['lh_mean']:.2f}",
                fontsize=9,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14, color="red")
            ax.set_title("Local Height MISSING", fontsize=9, color="red")
        ax.axis("off")

        # Load slope DEM
        ax = axes[idx, 3]
        dem_slope = load_raster(dem_slope_path)
        if dem_slope is not None:
            stats["slope_min"] = float(dem_slope.min())
            stats["slope_max"] = float(dem_slope.max())
            stats["slope_mean"] = float(dem_slope.mean())
            stats["slope_std"] = float(dem_slope.std())
            im = ax.imshow(dem_slope, cmap="magma", vmin=0, vmax=1)
            ax.set_title(
                f"Slope\n[{stats['slope_min']:.2f}, {stats['slope_max']:.2f}] mean={stats['slope_mean']:.2f}",
                fontsize=9,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14, color="red")
            ax.set_title("Slope MISSING", fontsize=9, color="red")
        ax.axis("off")

        # Histogram of DEM values
        ax = axes[idx, 4]
        if dem_lh is not None and dem_slope is not None:
            ax.hist(
                dem_lh.flatten(),
                bins=50,
                alpha=0.7,
                label="Local Height",
                color="green",
                density=True,
            )
            ax.hist(
                dem_slope.flatten(), bins=50, alpha=0.7, label="Slope", color="purple", density=True
            )
            ax.legend(fontsize=8)
            ax.set_title("DEM Distribution", fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title("DEM Distribution", fontsize=9)

        all_stats.append(stats)
        print(
            f"  {split}: mask={stats.get('mask_coverage', 'N/A'):.1f}%, lh_mean={stats.get('lh_mean', 'N/A'):.3f}, slope_mean={stats.get('slope_mean', 'N/A'):.3f}"
        )

    plt.suptitle("All Cemeteries - Train/Val/Test Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / "all_cemeteries_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY BY SPLIT")
    print("=" * 80)

    for split in ["TRAIN", "VAL", "TEST"]:
        split_stats = [s for s in all_stats if s["split"] == split]
        print(f"\n{split} ({len(split_stats)} cemeteries):")

        coverages = [s["mask_coverage"] for s in split_stats if "mask_coverage" in s]
        lh_means = [s["lh_mean"] for s in split_stats if "lh_mean" in s]
        lh_stds = [s["lh_std"] for s in split_stats if "lh_std" in s]
        slope_means = [s["slope_mean"] for s in split_stats if "slope_mean" in s]
        slope_stds = [s["slope_std"] for s in split_stats if "slope_std" in s]

        if coverages:
            print(
                f"  Mask coverage: {np.mean(coverages):.1f}% (range: {min(coverages):.1f}% - {max(coverages):.1f}%)"
            )
        if lh_means:
            print(
                f"  Local height: mean={np.mean(lh_means):.3f} (range: {min(lh_means):.3f} - {max(lh_means):.3f}), std={np.mean(lh_stds):.3f}"
            )
        if slope_means:
            print(
                f"  Slope: mean={np.mean(slope_means):.3f} (range: {min(slope_means):.3f} - {max(slope_means):.3f}), std={np.mean(slope_stds):.3f}"
            )


if __name__ == "__main__":
    main()
