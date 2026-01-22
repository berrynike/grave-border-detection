"""Inference module for full cemetery prediction."""

from pathlib import Path

import numpy as np
import rasterio
import torch

from grave_border_detection.data.tiling import calculate_tile_grid
from grave_border_detection.postprocessing import (
    draw_polygons_on_image,
    prediction_to_polygons,
)


def predict_full_cemetery(
    model: torch.nn.Module,
    orthophoto_path: Path,
    mask_path: Path | None = None,
    dem_path: Path | None = None,
    tile_size: int = 512,
    overlap: float = 0.15,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run inference on a full cemetery and reconstruct the prediction.

    Args:
        model: Trained segmentation model.
        orthophoto_path: Path to orthophoto GeoTIFF.
        mask_path: Optional path to ground truth mask.
        dem_path: Optional path to DEM GeoTIFF.
        tile_size: Size of tiles for inference.
        overlap: Overlap between tiles (0.0 to 0.5).
        device: Device to run inference on.

    Returns:
        Dictionary with:
            - 'rgb': Full RGB image (H, W, 3) in [0, 255] uint8
            - 'prediction': Full prediction probabilities (H, W) in [0, 1]
            - 'mask': Ground truth mask (H, W) if mask_path provided
            - 'transform': Rasterio transform for georeferencing
            - 'crs': Coordinate reference system
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # Read orthophoto metadata
    with rasterio.open(orthophoto_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

        # Calculate tile grid
        tiles = calculate_tile_grid(width, height, tile_size, overlap)

        # Initialize output arrays
        prediction_sum = np.zeros((height, width), dtype=np.float32)
        prediction_count = np.zeros((height, width), dtype=np.float32)
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Read full RGB for visualization
        rgb_data = src.read([1, 2, 3])  # (3, H, W)
        rgb_image = np.transpose(rgb_data, (1, 2, 0))  # (H, W, 3)

    # Check if we need DEM
    use_dem = dem_path is not None and dem_path.exists()

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Process each tile
    with rasterio.open(orthophoto_path) as ortho_src:
        dem_src = rasterio.open(dem_path) if use_dem else None

        try:
            for tile in tiles:
                # Create window for this tile
                window = rasterio.windows.Window(
                    col_off=tile.x_offset,
                    row_off=tile.y_offset,
                    width=tile.width,
                    height=tile.height,
                )

                # Read RGB tile
                rgb_tile = ortho_src.read([1, 2, 3], window=window)  # (3, H, W)

                # Normalize RGB
                rgb_normalized = rgb_tile.astype(np.float32) / 255.0
                for c in range(3):
                    rgb_normalized[c] = (rgb_normalized[c] - mean[c]) / std[c]

                # Stack channels
                if use_dem and dem_src is not None:
                    dem_tile = dem_src.read(1, window=window)  # (H, W)
                    dem_tile = dem_tile[np.newaxis, :, :]  # (1, H, W)
                    # Z-score normalize DEM
                    dem_mean = dem_tile.mean()
                    dem_std = dem_tile.std() + 1e-6
                    dem_normalized = (dem_tile - dem_mean) / dem_std
                    input_tensor = np.concatenate(
                        [rgb_normalized, dem_normalized.astype(np.float32)], axis=0
                    )
                else:
                    input_tensor = rgb_normalized

                # Convert to torch and run inference
                input_batch = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)

                # Accumulate predictions with overlap handling
                y_start = tile.y_offset
                y_end = min(y_start + tile.height, height)
                x_start = tile.x_offset
                x_end = min(x_start + tile.width, width)

                # Handle edge tiles that might be smaller
                tile_h = y_end - y_start
                tile_w = x_end - x_start

                prediction_sum[y_start:y_end, x_start:x_end] += probs[:tile_h, :tile_w]
                prediction_count[y_start:y_end, x_start:x_end] += 1.0

        finally:
            if dem_src is not None:
                dem_src.close()

    # Average overlapping predictions
    prediction = prediction_sum / np.maximum(prediction_count, 1.0)

    result = {
        "rgb": rgb_image,
        "prediction": prediction,
        "transform": transform,
        "crs": crs,
    }

    # Read ground truth mask if provided
    if mask_path is not None and mask_path.exists():
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.float32)
            result["mask"] = mask

    return result


def create_full_visualization(
    rgb: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
    min_area: int = 100,
    simplify_tolerance: float = 1.0,
) -> np.ndarray:
    """Create a visualization of full cemetery prediction.

    Args:
        rgb: RGB image (H, W, 3) uint8.
        prediction: Prediction probabilities (H, W) in [0, 1].
        mask: Optional ground truth mask (H, W).
        threshold: Threshold for binary prediction.
        min_area: Minimum polygon area in pixels.
        simplify_tolerance: Tolerance for polygon simplification.

    Returns:
        Visualization image (H, W*5, 3) with mask or (H, W*4, 3) without.
        Columns: RGB | GT | Prob | Error | Polygons (if mask provided)
                 RGB | Prob | Pred | Polygons (if no mask)
    """
    h, w = rgb.shape[:2]
    rgb_float = rgb.astype(np.float32) / 255.0

    # Column 1: RGB
    col1 = rgb_float.copy()

    # Column: Probability heatmap (black -> red -> yellow)
    prob_heatmap = np.zeros((h, w, 3), dtype=np.float32)
    prob_heatmap[:, :, 0] = prediction  # Red
    prob_heatmap[:, :, 1] = prediction * prediction  # Green (slower ramp)

    # Extract polygons from prediction
    polygons = prediction_to_polygons(
        prediction,
        threshold=threshold,
        min_area=min_area,
        transform=None,  # Pixel coordinates
        simplify_tolerance=simplify_tolerance,
    )

    # Column: Polygon overlay (yellow outlines on RGB)
    polygon_overlay = draw_polygons_on_image(
        rgb,
        polygons,
        transform=None,
        color=(255, 255, 0),  # Yellow
        thickness=2,
    )
    polygon_overlay_float = polygon_overlay.astype(np.float32) / 255.0

    if mask is not None:
        # Column 2: GT overlay (green)
        gt_overlay = rgb_float.copy()
        gt_mask = mask > 0.5
        gt_overlay[gt_mask] = [0.0, 1.0, 0.0]

        # Column 4: Error map
        pred_binary = prediction > threshold
        error_map = rgb_float.copy()

        tp = pred_binary & gt_mask
        fp = pred_binary & ~gt_mask
        fn = ~pred_binary & gt_mask

        error_map[tp] = [0.0, 1.0, 0.0]  # Green - correct
        error_map[fp] = [1.0, 0.0, 0.0]  # Red - false positive
        error_map[fn] = [0.0, 0.0, 1.0]  # Blue - missed

        # Stack: RGB | GT | Prob | Error | Polygons
        viz = np.concatenate(
            [col1, gt_overlay, prob_heatmap, error_map, polygon_overlay_float], axis=1
        )
    else:
        # Pred overlay
        pred_overlay = rgb_float.copy()
        pred_mask = prediction > threshold
        pred_overlay[pred_mask] = [1.0, 0.3, 0.3]

        # Stack: RGB | Prob | Pred | Polygons
        viz = np.concatenate([col1, prob_heatmap, pred_overlay, polygon_overlay_float], axis=1)

    # Convert back to uint8
    viz = (viz * 255).clip(0, 255).astype(np.uint8)

    return viz
