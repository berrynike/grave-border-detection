"""Synthetic dataset generator for testing the pipeline.

Generates fake cemetery orthophotos with rectangular "graves" and
corresponding annotations in proper geospatial formats.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box


def generate_grave_polygons(
    image_width: int,
    image_height: int,
    num_graves: int,
    grave_width_range: tuple[int, int] = (30, 60),
    grave_height_range: tuple[int, int] = (60, 120),
    min_spacing: int = 10,
    seed: int | None = None,
) -> list[Polygon]:
    """Generate random non-overlapping rectangular grave polygons.

    Args:
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        num_graves: Target number of graves to generate.
        grave_width_range: Min and max width of graves in pixels.
        grave_height_range: Min and max height of graves in pixels.
        min_spacing: Minimum spacing between graves in pixels.
        seed: Random seed for reproducibility.

    Returns:
        List of shapely Polygon objects representing graves.
    """
    rng = np.random.default_rng(seed)
    polygons: list[Polygon] = []
    attempts = 0
    max_attempts = num_graves * 50

    while len(polygons) < num_graves and attempts < max_attempts:
        attempts += 1

        # Random grave dimensions
        w = rng.integers(grave_width_range[0], grave_width_range[1])
        h = rng.integers(grave_height_range[0], grave_height_range[1])

        # Random position (with margin from edges)
        margin = min_spacing
        x = rng.integers(margin, image_width - w - margin)
        y = rng.integers(margin, image_height - h - margin)

        # Create candidate polygon
        candidate = box(x, y, x + w, y + h)

        # Check for overlap with existing polygons (including spacing)
        buffered_candidate = candidate.buffer(min_spacing)
        overlaps = False
        for existing in polygons:
            if buffered_candidate.intersects(existing):
                overlaps = True
                break

        if not overlaps:
            polygons.append(candidate)

    return polygons


def render_orthophoto(
    width: int,
    height: int,
    graves: list[Polygon],
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Render a synthetic orthophoto with graves.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        graves: List of grave polygons.
        seed: Random seed for reproducibility.

    Returns:
        RGBA image array of shape (4, height, width).
    """
    rng = np.random.default_rng(seed)

    # Create grass background with texture
    image = np.zeros((4, height, width), dtype=np.uint8)

    # Base grass color with noise
    grass_base = np.array([80, 120, 60], dtype=np.float32)
    noise = rng.normal(0, 15, (3, height, width)).astype(np.float32)

    for i in range(3):
        channel = grass_base[i] + noise[i]
        image[i] = np.clip(channel, 0, 255).astype(np.uint8)

    # Alpha channel (fully opaque)
    image[3] = 255

    # Draw graves as lighter rectangles (stone/gravel color)
    for grave in graves:
        minx, miny, maxx, maxy = [int(v) for v in grave.bounds]

        # Random grave color (gray/brown stone)
        grave_color = rng.integers(140, 200, size=3)
        grave_noise = rng.normal(0, 10, (3, maxy - miny, maxx - minx))

        for i in range(3):
            patch = grave_color[i] + grave_noise[i]
            image[i, miny:maxy, minx:maxx] = np.clip(patch, 0, 255).astype(np.uint8)

        # Add border (darker edge)
        border_width = 2
        border_color = (grave_color * 0.6).astype(np.uint8)
        # Top and bottom borders
        for c in range(3):
            image[c, miny : miny + border_width, minx:maxx] = border_color[c]
            image[c, maxy - border_width : maxy, minx:maxx] = border_color[c]
            # Left and right borders
            image[c, miny:maxy, minx : minx + border_width] = border_color[c]
            image[c, miny:maxy, maxx - border_width : maxx] = border_color[c]

    return image


def render_dem(
    width: int,
    height: int,
    graves: list[Polygon],
    base_elevation: float = 500.0,
    grave_elevation: float = 0.15,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Render a synthetic DEM with raised grave borders.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        graves: List of grave polygons.
        base_elevation: Base terrain elevation in meters.
        grave_elevation: Height of grave borders above base in meters.
        seed: Random seed for reproducibility.

    Returns:
        DEM array of shape (1, height, width) with float32 elevations.
    """
    rng = np.random.default_rng(seed)

    # Create base terrain with slight undulation
    dem = np.full((1, height, width), base_elevation, dtype=np.float32)

    # Add gentle terrain noise
    noise = rng.normal(0, 0.02, (height, width)).astype(np.float32)
    dem[0] += noise

    # Raise grave borders
    for grave in graves:
        minx, miny, maxx, maxy = [int(v) for v in grave.bounds]

        # Raise the entire grave area slightly
        dem[0, miny:maxy, minx:maxx] += grave_elevation * 0.5

        # Raise borders more
        border_width = 3
        dem[0, miny : miny + border_width, minx:maxx] += grave_elevation
        dem[0, maxy - border_width : maxy, minx:maxx] += grave_elevation
        dem[0, miny:maxy, minx : minx + border_width] += grave_elevation
        dem[0, miny:maxy, maxx - border_width : maxx] += grave_elevation

    return dem


def render_mask(
    width: int,
    height: int,
    graves: list[Polygon],
) -> NDArray[np.uint8]:
    """Render binary segmentation mask.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        graves: List of grave polygons.

    Returns:
        Binary mask array of shape (1, height, width).
    """
    mask = np.zeros((1, height, width), dtype=np.uint8)

    for grave in graves:
        minx, miny, maxx, maxy = [int(v) for v in grave.bounds]
        mask[0, miny:maxy, minx:maxx] = 1

    return mask


def generate_synthetic_dataset(
    output_dir: Path,
    num_cemeteries: int = 3,
    image_size: tuple[int, int] = (512, 512),
    graves_per_cemetery: int = 15,
    resolution: float = 0.02,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate a complete synthetic dataset for pipeline testing.

    Creates orthophotos, DEMs, masks, and vector annotations in proper
    geospatial formats with consistent CRS (EPSG:3857).

    Args:
        output_dir: Directory to save generated files.
        num_cemeteries: Number of synthetic cemeteries to generate.
        image_size: Size of generated images (width, height) in pixels.
        graves_per_cemetery: Target number of graves per cemetery.
        resolution: Pixel resolution in meters.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with paths to generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ortho_dir = output_dir / "orthophotos"
    dem_dir = output_dir / "dems"
    mask_dir = output_dir / "masks"
    vector_dir = output_dir / "annotations"

    for d in [ortho_dir, dem_dir, mask_dir, vector_dir]:
        d.mkdir(exist_ok=True)

    crs = "EPSG:3857"
    width, height = image_size

    # Base coordinates (somewhere in Bavaria)
    base_x = 1370000  # ~12.3° E in EPSG:3857
    base_y = 6120000  # ~48.2° N in EPSG:3857

    all_annotations = []
    generated_files: dict[str, list[Path]] = {
        "orthophotos": [],
        "dems": [],
        "masks": [],
        "annotations": [],
    }

    for i in range(num_cemeteries):
        cemetery_id = f"synthetic_{i + 1:02d}"
        cemetery_seed = seed + i

        # Generate graves
        graves = generate_grave_polygons(
            width,
            height,
            graves_per_cemetery,
            seed=cemetery_seed,
        )

        # Offset for this cemetery
        offset_x = base_x + i * (width * resolution + 100)
        offset_y = base_y

        # Bounds in CRS coordinates
        bounds = (
            offset_x,
            offset_y,
            offset_x + width * resolution,
            offset_y + height * resolution,
        )
        transform = from_bounds(*bounds, width, height)

        # Generate and save orthophoto
        ortho = render_orthophoto(width, height, graves, seed=cemetery_seed)
        ortho_path = ortho_dir / f"{cemetery_id}_ortho.tif"

        with rasterio.open(
            ortho_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(ortho)

        generated_files["orthophotos"].append(ortho_path)

        # Generate and save DEM
        dem = render_dem(width, height, graves, seed=cemetery_seed)
        dem_path = dem_dir / f"{cemetery_id}_dem.tif"

        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(dem)

        generated_files["dems"].append(dem_path)

        # Generate and save mask
        mask = render_mask(width, height, graves)
        mask_path = mask_dir / f"{cemetery_id}_mask.tif"

        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask)

        generated_files["masks"].append(mask_path)

        # Convert pixel polygons to georeferenced polygons
        for j, grave in enumerate(graves):
            minx, miny, maxx, maxy = grave.bounds
            # Transform pixel coordinates to CRS coordinates
            geo_minx = offset_x + minx * resolution
            geo_maxx = offset_x + maxx * resolution
            # Y is inverted in image coordinates
            geo_miny = offset_y + (height - maxy) * resolution
            geo_maxy = offset_y + (height - miny) * resolution

            geo_polygon = box(geo_minx, geo_miny, geo_maxx, geo_maxy)

            all_annotations.append(
                {
                    "cemetery_id": cemetery_id,
                    "grave_id": f"{cemetery_id}_grave_{j + 1:03d}",
                    "geometry": geo_polygon,
                }
            )

    # Save all annotations as single GeoPackage
    gdf = gpd.GeoDataFrame(all_annotations, crs=crs)
    annotations_path = vector_dir / "synthetic_annotations.gpkg"
    gdf.to_file(annotations_path, driver="GPKG")
    generated_files["annotations"].append(annotations_path)

    # Create summary
    summary = {
        "num_cemeteries": num_cemeteries,
        "total_graves": len(all_annotations),
        "image_size": image_size,
        "resolution_m": resolution,
        "crs": crs,
        "files": {k: [str(p) for p in v] for k, v in generated_files.items()},
    }

    import json

    summary_path = output_dir / "synthetic_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "orthophotos": ortho_dir,
        "dems": dem_dir,
        "masks": mask_dir,
        "annotations": annotations_path,
        "summary": summary_path,
    }


if __name__ == "__main__":
    # Generate test dataset when run directly
    output = generate_synthetic_dataset(
        output_dir=Path("data/synthetic"),
        num_cemeteries=3,
        image_size=(512, 512),
        graves_per_cemetery=20,
        seed=42,
    )
    print("Generated synthetic dataset:")
    for key, path in output.items():
        print(f"  {key}: {path}")
