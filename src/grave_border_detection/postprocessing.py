"""Postprocessing module for converting predictions to vector polygons.

TODO: Improve instance separation for touching graves. Current approach
(connected components + rectangle fitting) merges adjacent graves into
single polygons. Options to explore:
- Watershed algorithm using probability gradients
- Instance segmentation model (Mask R-CNN, YOLO)
- Boundary detection network (predict edges between graves)
- Distance transform + peak detection for seed points
"""

from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine  # type: ignore[import-untyped]
from shapely.geometry import shape
from shapely.validation import make_valid


def prediction_to_instances(
    prediction: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100,
    morphology_size: int = 5,
) -> tuple[np.ndarray, int]:
    """Convert prediction probabilities to instance labels.

    Uses morphological cleaning and connected components.

    Args:
        prediction: Prediction probabilities (H, W) in [0, 1].
        threshold: Threshold for binarization.
        min_area: Minimum area in pixels for a valid instance.
        morphology_size: Kernel size for morphological operations.

    Returns:
        Tuple of (labeled_mask, num_instances) where labeled_mask has
        unique integer labels for each instance (0 = background).
    """
    # Binarize
    binary = (prediction > threshold).astype(np.uint8)

    # Morphological cleaning: close small holes, then open to remove noise
    if morphology_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_size, morphology_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel).astype(np.uint8)

    # Connected components
    num_labels, labels = cv2.connectedComponents(binary)

    # Filter small instances
    filtered_labels = np.zeros_like(labels)
    new_label = 1

    for label_id in range(1, num_labels):
        mask = labels == label_id
        area = mask.sum()
        if area >= min_area:
            filtered_labels[mask] = new_label
            new_label += 1

    return filtered_labels, new_label - 1


def fit_rotated_rectangles(
    instance_labels: np.ndarray,
    transform: Affine | None = None,
) -> list[dict[str, Any]]:
    """Fit minimum area rotated rectangles to each instance.

    Args:
        instance_labels: Labeled mask (H, W) with unique integer per instance.
        transform: Affine transform for georeferencing.

    Returns:
        List of dicts with 'geometry' (Shapely Polygon) and 'instance_id'.
    """
    from shapely.geometry import Polygon

    polygons = []
    unique_labels = np.unique(instance_labels)

    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue

        # Get contours for this instance
        mask = (instance_labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 5:
            continue

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 4 corner points

        # Convert to proper coordinate order
        box = np.array(box)

        # Transform to CRS coordinates if needed
        if transform is not None:
            coords = []
            for px, py in box:
                x, y = transform * (px, py)
                coords.append((x, y))
            box = np.array(coords)

        # Create polygon (close the ring)
        poly = Polygon(box)

        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(
                {
                    "geometry": poly,
                    "instance_id": int(label_id),
                }
            )

    return polygons


def instances_to_polygons(
    instance_labels: np.ndarray,
    transform: Affine | None = None,
    simplify_tolerance: float = 1.0,
) -> list[dict[str, Any]]:
    """Convert instance labels to polygon geometries.

    Args:
        instance_labels: Labeled mask (H, W) with unique integer per instance.
        transform: Affine transform for georeferencing (pixel to CRS coords).
        simplify_tolerance: Tolerance for Douglas-Peucker simplification.
            In pixels if no transform, in CRS units if transform provided.

    Returns:
        List of dicts with 'geometry' (Shapely Polygon) and 'instance_id'.
    """
    polygons = []

    # Use rasterio to extract shapes with georeferencing
    if transform is not None:
        for geom, value in rasterio.features.shapes(
            instance_labels.astype(np.int32),
            transform=transform,
        ):
            if value == 0:  # Skip background
                continue

            poly = shape(geom)
            poly = make_valid(poly)

            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            if poly.is_valid and not poly.is_empty:
                polygons.append(
                    {
                        "geometry": poly,
                        "instance_id": int(value),
                    }
                )
    else:
        # No transform - use pixel coordinates
        for geom, value in rasterio.features.shapes(
            instance_labels.astype(np.int32),
        ):
            if value == 0:
                continue

            poly = shape(geom)
            poly = make_valid(poly)

            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            if poly.is_valid and not poly.is_empty:
                polygons.append(
                    {
                        "geometry": poly,
                        "instance_id": int(value),
                    }
                )

    return polygons


def prediction_to_polygons(
    prediction: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100,
    transform: Affine | None = None,
    simplify_tolerance: float = 1.0,
    fit_rectangles: bool = True,
    morphology_size: int = 5,
) -> list[dict[str, Any]]:
    """Full pipeline: prediction probabilities to polygons.

    Args:
        prediction: Prediction probabilities (H, W) in [0, 1].
        threshold: Threshold for binarization.
        min_area: Minimum area in pixels for a valid instance.
        transform: Affine transform for georeferencing.
        simplify_tolerance: Tolerance for polygon simplification.
        fit_rectangles: If True, fit minimum area rectangles (cleaner).
        morphology_size: Kernel size for morphological cleaning.

    Returns:
        List of dicts with 'geometry' and 'instance_id'.
    """
    instance_labels, num_instances = prediction_to_instances(
        prediction, threshold, min_area, morphology_size
    )

    if num_instances == 0:
        return []

    if fit_rectangles:
        return fit_rotated_rectangles(instance_labels, transform)
    else:
        return instances_to_polygons(instance_labels, transform, simplify_tolerance)


def polygons_to_geopackage(
    polygons: list[dict[str, Any]],
    output_path: Path,
    crs: str | None = None,
) -> None:
    """Export polygons to GeoPackage file.

    Args:
        polygons: List of dicts with 'geometry' and 'instance_id'.
        output_path: Path to output .gpkg file.
        crs: Coordinate reference system (e.g., 'EPSG:3857').
    """
    if not polygons:
        # Create empty GeoDataFrame
        gdf = gpd.GeoDataFrame(columns=["instance_id", "geometry"], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(polygons, crs=crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")


def draw_polygons_on_image(
    image: np.ndarray,
    polygons: list[dict[str, Any]],
    transform: Affine | None = None,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw polygon outlines on an image.

    Args:
        image: RGB image (H, W, 3) uint8.
        polygons: List of dicts with 'geometry' (Shapely Polygon).
        transform: Affine transform to convert CRS coords to pixels.
            If None, assumes polygons are already in pixel coordinates.
        color: BGR color for outlines.
        thickness: Line thickness in pixels.

    Returns:
        Image with polygon outlines drawn.
    """
    result = image.copy()

    for poly_dict in polygons:
        poly = poly_dict["geometry"]

        if poly.is_empty:
            continue

        # Get exterior coordinates
        if hasattr(poly, "exterior"):
            coords = np.array(poly.exterior.coords)
        else:
            continue

        # Convert from CRS to pixel coordinates if transform provided
        if transform is not None:
            # Inverse transform: CRS -> pixel
            inv_transform = ~transform
            pixel_coords = []
            for x, y in coords:
                px, py = inv_transform * (x, y)
                pixel_coords.append([int(px), int(py)])
            coords = np.array(pixel_coords)
        else:
            coords = coords.astype(int)

        # Draw polygon outline
        coords = coords.reshape((-1, 1, 2))
        cv2.polylines(result, [coords], isClosed=True, color=color, thickness=thickness)

    return result
