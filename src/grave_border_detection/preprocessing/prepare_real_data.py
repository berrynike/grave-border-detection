"""Prepare real cemetery data for training.

Rasterizes vector annotations to mask GeoTIFFs and organizes data
into the expected folder structure for the training pipeline.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import Resampling, reproject

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rasterio.crs import CRS
    from rasterio.transform import Affine

logger = logging.getLogger(__name__)


def extract_cemetery_id(filename: str) -> str | None:
    """Extract 2-digit cemetery ID from filename.

    Handles various naming patterns:
    - "01-251107Weilkirchen..." -> "01"
    - "02Kirchweg21Heldenstein-..." -> "02"
    - "09StNikolaus251128-..." -> "09"
    - "10-Westerbuchberg..." -> "10"

    Args:
        filename: Filename to extract ID from.

    Returns:
        2-digit ID string or None if not found.
    """
    import re

    # Match 2 digits at the start of the filename
    match = re.match(r"^(\d{2})", filename)
    return match.group(1) if match else None


def find_matching_files(
    ortho_dir: Path,
    annotation_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """Find matching orthophoto and annotation files by cemetery ID.

    Args:
        ortho_dir: Directory containing orthophoto GeoTIFFs.
        annotation_dir: Directory containing annotation GeoPackages.

    Returns:
        List of (cemetery_id, ortho_path, annotation_path) tuples.
    """
    matches: list[tuple[str, Path, Path]] = []

    # Get all annotation files by ID
    annotation_files: dict[str, Path] = {}
    for f in annotation_dir.glob("*.gpkg"):
        file_id = extract_cemetery_id(f.name)
        if file_id:
            annotation_files[file_id] = f

    # Get orthophotos (exclude DEM files)
    ortho_files: dict[str, Path] = {}
    for f in sorted(ortho_dir.glob("*.tif")):
        # Skip DEM files
        if "DEM" in f.name or "dem" in f.name:
            continue
        file_id = extract_cemetery_id(f.name)
        if file_id:
            ortho_files[file_id] = f

    # Match by ID
    for file_id in sorted(ortho_files.keys()):
        if file_id in annotation_files:
            cemetery_id = f"cemetery_{file_id}"
            ortho_path = ortho_files[file_id]
            annotation_path = annotation_files[file_id]
            matches.append((cemetery_id, ortho_path, annotation_path))
            logger.info(f"Matched {cemetery_id}: {ortho_path.name} <-> {annotation_path.name}")
        else:
            logger.warning(f"No annotation found for orthophoto: {ortho_files[file_id].name}")

    return matches


def rasterize_annotations(
    annotation_path: Path,
    reference_path: Path,
    output_path: Path,
) -> None:
    """Rasterize vector annotations to match a reference raster.

    Args:
        annotation_path: Path to vector annotation file (GeoPackage/Shapefile).
        reference_path: Path to reference raster (orthophoto) for extent/resolution.
        output_path: Path to save the output mask GeoTIFF.
    """
    # Read reference raster properties
    with rasterio.open(reference_path) as ref:
        transform: Affine = ref.transform
        crs: CRS = ref.crs
        height: int = ref.height
        width: int = ref.width

    # Read vector annotations
    gdf = gpd.read_file(annotation_path)

    # Ensure CRS matches
    if gdf.crs != crs:
        logger.info(f"Reprojecting annotations from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # Create mask array
    mask: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)

    # Rasterize polygons
    if len(gdf) > 0:
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        if shapes:
            features.rasterize(
                shapes=shapes,
                out=mask,
                transform=transform,
                dtype=np.uint8,
            )

    # Count pixels for logging
    grave_pixels = int(np.sum(mask > 0))
    total_pixels = height * width
    coverage = grave_pixels / total_pixels * 100
    logger.info(f"Mask stats: {grave_pixels:,} grave pixels ({coverage:.1f}% coverage)")

    # Write output mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)

    logger.info(f"Saved mask to {output_path}")


def copy_orthophoto(
    src_path: Path,
    dst_path: Path,
    convert_rgba_to_rgb: bool = True,
) -> None:
    """Copy orthophoto, optionally converting RGBA to RGB.

    Args:
        src_path: Source orthophoto path.
        dst_path: Destination path.
        convert_rgba_to_rgb: If True, convert 4-band RGBA to 3-band RGB.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        # Check if RGBA (4 bands)
        if convert_rgba_to_rgb and src.count == 4:
            logger.info(f"Converting RGBA to RGB: {src_path.name}")

            # Read RGB bands only (skip alpha)
            rgb: NDArray[np.uint8] = src.read([1, 2, 3])

            # Write RGB output
            profile = src.profile.copy()
            profile.update(count=3, compress="lzw")

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(rgb)
        else:
            # Just copy the file
            shutil.copy2(src_path, dst_path)

    logger.info(f"Saved orthophoto to {dst_path}")


def resample_dem_to_ortho(
    dem_path: Path,
    ortho_path: Path,
    output_path: Path,
    valid_mask_path: Path | None = None,
) -> None:
    """Resample DEM to match orthophoto resolution and extent.

    Preserves real elevation values (meters). Only resamples spatially
    using bilinear interpolation. Areas outside DEM coverage are marked
    as nodata (-9999).

    Args:
        dem_path: Path to source DEM GeoTIFF.
        ortho_path: Path to reference orthophoto (for target grid).
        output_path: Path to save resampled DEM.
        valid_mask_path: Optional path to save valid data mask (1=valid, 0=nodata).
    """
    # Nodata value for output
    NODATA_VALUE = -9999.0

    with rasterio.open(ortho_path) as ortho:
        target_crs = ortho.crs
        target_transform = ortho.transform
        target_width = ortho.width
        target_height = ortho.height

    with rasterio.open(dem_path) as dem:
        # Initialize with nodata
        dem_resampled: NDArray[np.float32] = np.full(
            (target_height, target_width), NODATA_VALUE, dtype=np.float32
        )

        reproject(
            source=rasterio.band(dem, 1),
            destination=dem_resampled,
            src_transform=dem.transform,
            src_crs=dem.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=dem.nodata,
            dst_nodata=NODATA_VALUE,
        )

    # Create valid data mask (where DEM has actual elevation data)
    # Nodata is either our NODATA_VALUE or values < -1000 (common nodata markers)
    valid_mask = (dem_resampled != NODATA_VALUE) & (dem_resampled > -1000)
    valid_count = int(valid_mask.sum())
    valid_pct = 100 * valid_count / (target_height * target_width)

    # Get stats for valid data only
    valid_elevations = dem_resampled[valid_mask]
    if len(valid_elevations) > 0:
        elev_min, elev_max = float(valid_elevations.min()), float(valid_elevations.max())
    else:
        elev_min, elev_max = 0.0, 0.0

    logger.info(
        f"  Resampled DEM: {target_width}x{target_height}, "
        f"valid: {valid_pct:.1f}%, elevation: [{elev_min:.1f}, {elev_max:.1f}]m"
    )

    # Save DEM with nodata value
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
        nodata=NODATA_VALUE,
    ) as dst:
        dst.write(dem_resampled, 1)

    # Save valid mask if path provided
    if valid_mask_path is not None:
        valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            valid_mask_path,
            "w",
            driver="GTiff",
            height=target_height,
            width=target_width,
            count=1,
            dtype=np.uint8,
            crs=target_crs,
            transform=target_transform,
            compress="lzw",
        ) as dst:
            dst.write(valid_mask.astype(np.uint8), 1)
        logger.info(f"  Saved valid mask: {valid_pct:.1f}% coverage")


def find_dem_for_cemetery(dem_dir: Path, cemetery_file_id: str) -> Path | None:
    """Find DEM file matching a cemetery ID.

    Args:
        dem_dir: Directory containing DEM GeoTIFFs.
        cemetery_file_id: The 2-digit cemetery ID (e.g., "01", "02").

    Returns:
        Path to DEM file or None if not found.
    """
    # Look for GeoTIFF DEMs (not GeoPackage which are colormap visualizations)
    for f in dem_dir.glob("*.tif"):
        if "DEM" not in f.name and "dem" not in f.name:
            continue
        file_id = extract_cemetery_id(f.name)
        if file_id == cemetery_file_id:
            return f
    return None


def prepare_real_dataset(
    ortho_dir: Path,
    annotation_dir: Path,
    dem_dir: Path | None,
    output_dir: Path,
    convert_rgba_to_rgb: bool = True,
) -> dict[str, list[str]]:
    """Prepare real cemetery data for training.

    Args:
        ortho_dir: Directory containing orthophoto GeoTIFFs.
        annotation_dir: Directory containing annotation GeoPackages.
        dem_dir: Optional directory containing DEM GeoTIFFs.
        output_dir: Output directory for prepared data.
        convert_rgba_to_rgb: Convert RGBA orthophotos to RGB.

    Returns:
        Dictionary with cemetery IDs organized by suggested split.
    """
    logger.info("Preparing real cemetery dataset...")

    # Find matching files
    matches = find_matching_files(ortho_dir, annotation_dir)

    if not matches:
        raise ValueError("No matching orthophoto/annotation pairs found!")

    logger.info(f"Found {len(matches)} cemetery pairs")

    # Create output directories
    orthophotos_out = output_dir / "orthophotos"
    masks_out = output_dir / "masks"
    dems_out = output_dir / "dems"
    valid_masks_out = output_dir / "valid_masks"

    for d in [orthophotos_out, masks_out, dems_out, valid_masks_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Process each cemetery
    cemetery_ids: list[str] = []

    for cemetery_id, ortho_path, annotation_path in matches:
        logger.info(f"\nProcessing {cemetery_id}...")

        # Rasterize annotations to mask
        mask_path = masks_out / f"{cemetery_id}_mask.tif"
        rasterize_annotations(annotation_path, ortho_path, mask_path)

        # Copy orthophoto
        ortho_out_path = orthophotos_out / f"{cemetery_id}_ortho.tif"
        copy_orthophoto(ortho_path, ortho_out_path, convert_rgba_to_rgb)

        # Resample DEM if available
        if dem_dir is not None:
            # Extract 2-digit ID from cemetery_id (e.g., "cemetery_01" -> "01")
            file_id = cemetery_id.split("_")[1]
            dem_src = find_dem_for_cemetery(dem_dir, file_id)

            if dem_src is not None:
                dem_dst = dems_out / f"{cemetery_id}_dem.tif"
                valid_mask_dst = valid_masks_out / f"{cemetery_id}_valid.tif"
                logger.info(f"Processing DEM: {dem_src.name}")
                resample_dem_to_ortho(
                    dem_path=dem_src,
                    ortho_path=ortho_out_path,
                    output_path=dem_dst,
                    valid_mask_path=valid_mask_dst,
                )
            else:
                logger.warning(f"No GeoTIFF DEM found for {cemetery_id}")

        cemetery_ids.append(cemetery_id)

    # Suggest train/val/test split (6/2/2)
    n = len(cemetery_ids)
    n_train = max(1, int(n * 0.6))
    n_val = max(1, int(n * 0.2))

    split = {
        "train": cemetery_ids[:n_train],
        "val": cemetery_ids[n_train : n_train + n_val],
        "test": cemetery_ids[n_train + n_val :],
    }

    logger.info("\nSuggested split:")
    logger.info(f"  Train: {split['train']}")
    logger.info(f"  Val: {split['val']}")
    logger.info(f"  Test: {split['test']}")

    logger.info(f"\nDataset prepared in {output_dir}")

    return split


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Default paths
    base_dir = Path("data/external")
    ortho_dir = base_dir / "goetiff-orthos-ki-training-data/4k"
    annotation_dir = base_dir / "shape"
    output_dir = Path("data/real")

    # Check for GeoTIFF DEMs (only 01 and 02 have them)
    dem_dir = ortho_dir  # DEMs are in same directory as orthophotos

    split = prepare_real_dataset(
        ortho_dir=ortho_dir,
        annotation_dir=annotation_dir,
        dem_dir=ortho_dir,  # DEMs are in same directory as orthophotos
        output_dir=output_dir,
        convert_rgba_to_rgb=True,
    )

    print("\n" + "=" * 60)
    print("Add this to your config or use CLI overrides:")
    print("=" * 60)
    print(f"train_cemeteries: {split['train']}")
    print(f"val_cemeteries: {split['val']}")
    print(f"test_cemeteries: {split['test']}")
