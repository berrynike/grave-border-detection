"""Precompute normalized DEMs for faster training.

This script precomputes DEM normalization (local_height and slope) once,
so HPO and training can load them instantly instead of recomputing.

Usage:
    uv run python scripts/precompute_dem_normalization.py
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

from grave_border_detection.preprocessing import normalize_dem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed parameters for precomputation
METHODS = {
    "local_height": {
        "window_size": 128,
        "ground_percentile": 20.0,
        "max_height": 0.5,
    },
    "slope": {
        "pixel_size": 0.02,
    },
}


def process_single_dem(
    dem_path: Path, method_name: str, params: dict, output_path: Path
) -> tuple[str, str, float]:
    """Process a single DEM with a single method.

    Args:
        dem_path: Path to raw DEM file.
        method_name: Normalization method name.
        params: Method parameters.
        output_path: Path to save normalized DEM.

    Returns:
        Tuple of (cemetery_id, method_name, elapsed_time).
    """
    cemetery_id = dem_path.stem.replace("_dem", "")

    if output_path.exists():
        return (cemetery_id, method_name, 0.0)

    start_time = time.time()

    # Load raw DEM
    with rasterio.open(dem_path) as src:
        dem_raw = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Update profile for float32 output
    profile.update(dtype=np.float32, count=1)

    # Normalize
    dem_normalized = normalize_dem(dem_raw, method=method_name, **params)

    # Save
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dem_normalized, 1)

    elapsed = time.time() - start_time
    return (cemetery_id, method_name, elapsed)


def precompute_dem_normalization(data_root: Path, max_workers: int = 6) -> None:
    """Precompute normalized DEMs for all cemeteries in parallel.

    Args:
        data_root: Path to data directory containing dems/ subdirectory.
        max_workers: Number of parallel workers.
    """
    dems_dir = data_root / "dems"
    if not dems_dir.exists():
        raise FileNotFoundError(f"DEMs directory not found: {dems_dir}")

    # Find all DEM files
    dem_files = sorted(dems_dir.glob("cemetery_*_dem.tif"))
    # Filter out already-normalized files
    dem_files = [f for f in dem_files if "_dem_" not in f.stem or f.stem.endswith("_dem")]
    dem_files = [f for f in dem_files if f.stem.endswith("_dem")]

    logger.info(f"Found {len(dem_files)} DEM files, using {max_workers} workers")

    # Build list of jobs
    jobs = []
    for dem_path in dem_files:
        cemetery_id = dem_path.stem.replace("_dem", "")
        for method_name, params in METHODS.items():
            output_path = dems_dir / f"{cemetery_id}_dem_{method_name}.tif"
            if not output_path.exists():
                jobs.append((dem_path, method_name, params, output_path))
            else:
                logger.info(f"  {cemetery_id}/{method_name}: already exists, skipping")

    logger.info(f"Processing {len(jobs)} jobs in parallel...")
    start_total = time.time()

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_dem, *job): job for job in jobs}

        for future in as_completed(futures):
            cemetery_id, method_name, elapsed = future.result()
            if elapsed > 0:
                logger.info(f"  {cemetery_id}/{method_name}: done in {elapsed:.1f}s")

    total_elapsed = time.time() - start_total
    logger.info(f"\nPrecomputation complete in {total_elapsed:.1f}s!")
    logger.info("You can now use dem_normalization_method in configs and HPO.")


if __name__ == "__main__":
    data_root = Path("data/real")
    precompute_dem_normalization(data_root, max_workers=6)
