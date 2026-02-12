"""Generate enhanced full cemetery visualization with DEM channels.

Creates a visualization showing:
- RGB orthophoto
- Local height DEM
- Slope DEM
- Ground truth mask
- Prediction probability
- Error map (TP/FP/FN)
- Polygon overlay
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from grave_border_detection.data.tiling import calculate_tile_grid
from grave_border_detection.models.segmentation import SegmentationModel
from grave_border_detection.postprocessing import (
    draw_polygons_on_image,
    prediction_to_polygons,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[SegmentationModel, int]:
    """Load model from checkpoint.

    Returns:
        Tuple of (model, in_channels) where in_channels indicates model configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    in_channels = hparams.get("in_channels", 4)  # Default to RGB + 1 DEM

    model = SegmentationModel(
        architecture=hparams.get("architecture", "Unet"),
        encoder_name=hparams.get("encoder_name", "resnet34"),
        encoder_weights=None,  # Don't load pretrained, we have trained weights
        in_channels=in_channels,
        classes=hparams.get("classes", 1),
        lr=hparams.get("lr", 1e-4),
    )

    # Load state dict
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, in_channels


def predict_full_cemetery(
    model: torch.nn.Module,
    orthophoto_path: Path,
    local_height_path: Path,
    slope_path: Path | None = None,
    in_channels: int = 4,
    tile_size: int = 512,
    overlap: float = 0.15,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run inference on full cemetery.

    Args:
        model: Trained model.
        orthophoto_path: Path to RGB orthophoto.
        local_height_path: Path to local_height DEM.
        slope_path: Path to slope DEM (only used if in_channels=5).
        in_channels: Number of input channels (3=RGB, 4=RGB+height, 5=RGB+height+slope).
        tile_size: Inference tile size.
        overlap: Tile overlap fraction.
        device: Torch device.

    Returns:
        Dictionary with rgb, local_height, slope, prediction, transform, crs.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()

    # Read metadata
    with rasterio.open(orthophoto_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        rgb_image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))

    # Read DEM channels (always load both for visualization, even if model uses fewer)
    with rasterio.open(local_height_path) as src:
        local_height = src.read(1).astype(np.float32)

    slope = None
    if slope_path and slope_path.exists():
        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype(np.float32)

    # Calculate tiles
    tiles = calculate_tile_grid(width, height, tile_size, overlap)

    # Initialize accumulators
    prediction_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    log.info(f"Processing {len(tiles)} tiles with {in_channels} input channels...")

    with rasterio.open(orthophoto_path) as ortho_src:
        for _i, tile in enumerate(tiles):
            window = rasterio.windows.Window(
                col_off=tile.x_offset,
                row_off=tile.y_offset,
                width=tile.width,
                height=tile.height,
            )

            # Read RGB tile
            rgb_tile = ortho_src.read([1, 2, 3], window=window).astype(np.float32)
            rgb_tile = rgb_tile / 255.0
            for c in range(3):
                rgb_tile[c] = (rgb_tile[c] - mean[c]) / std[c]

            # Read DEM tiles
            y_start, y_end = tile.y_offset, tile.y_offset + tile.height
            x_start, x_end = tile.x_offset, tile.x_offset + tile.width

            if in_channels == 3:
                # RGB only
                input_tensor = rgb_tile
            elif in_channels == 4:
                # RGB + local_height
                lh_tile = local_height[y_start:y_end, x_start:x_end][np.newaxis, ...]
                input_tensor = np.concatenate([rgb_tile, lh_tile], axis=0)
            elif in_channels == 5 and slope is not None:
                # RGB + local_height + slope
                lh_tile = local_height[y_start:y_end, x_start:x_end][np.newaxis, ...]
                sl_tile = slope[y_start:y_end, x_start:x_end][np.newaxis, ...]
                input_tensor = np.concatenate([rgb_tile, lh_tile, sl_tile], axis=0)
            else:
                raise ValueError(f"Unsupported in_channels={in_channels}")

            input_batch = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_batch)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # Accumulate
            tile_h = min(tile.height, height - y_start)
            tile_w = min(tile.width, width - x_start)
            prediction_sum[y_start : y_start + tile_h, x_start : x_start + tile_w] += probs[
                :tile_h, :tile_w
            ]
            prediction_count[y_start : y_start + tile_h, x_start : x_start + tile_w] += 1.0

    prediction = prediction_sum / np.maximum(prediction_count, 1.0)

    return {
        "rgb": rgb_image,
        "local_height": local_height,
        "slope": slope,
        "prediction": prediction,
        "transform": transform,
        "crs": crs,
    }


def create_enhanced_visualization(
    rgb: np.ndarray,
    local_height: np.ndarray,
    slope: np.ndarray | None,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
    output_path: Path | None = None,
    title: str = "",
) -> None:
    """Create enhanced visualization with all channels."""
    _h, _w = rgb.shape[:2]

    # Determine layout: 2 rows
    # Row 1: RGB, Local Height, Slope (or placeholder), Ground Truth (if mask)
    # Row 2: Prediction, Error Map (if mask), Polygons
    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 6, 12))
    axes = axes.flatten()

    # Row 1: Input channels
    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Orthophoto", fontsize=12)
    axes[0].axis("off")

    # Local Height
    im1 = axes[1].imshow(local_height, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Local Height (normalized)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Slope (or empty if not available)
    if slope is not None:
        im2 = axes[2].imshow(slope, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Slope (normalized)", fontsize=12)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "Slope\n(not used)", ha="center", va="center", fontsize=14)
        axes[2].set_title("Slope", fontsize=12)
    axes[2].axis("off")

    if mask is not None:
        # Ground truth
        gt_overlay = rgb.copy().astype(np.float32) / 255.0
        gt_mask = mask > 0.5
        gt_overlay[gt_mask] = [0.0, 1.0, 0.0]
        axes[3].imshow(gt_overlay)
        axes[3].set_title("Ground Truth", fontsize=12)
        axes[3].axis("off")

    # Row 2: Predictions
    row2_start = n_cols

    # Prediction heatmap
    im3 = axes[row2_start].imshow(prediction, cmap="hot", vmin=0, vmax=1)
    axes[row2_start].set_title("Prediction Probability", fontsize=12)
    axes[row2_start].axis("off")
    plt.colorbar(im3, ax=axes[row2_start], fraction=0.046, pad=0.04)

    if mask is not None:
        # Error map
        pred_binary = prediction > threshold
        gt_mask = mask > 0.5

        error_map = rgb.copy().astype(np.float32) / 255.0
        tp = pred_binary & gt_mask
        fp = pred_binary & ~gt_mask
        fn = ~pred_binary & gt_mask

        error_map[tp] = [0.0, 1.0, 0.0]  # Green - correct
        error_map[fp] = [1.0, 0.0, 0.0]  # Red - false positive
        error_map[fn] = [0.0, 0.0, 1.0]  # Blue - missed

        axes[row2_start + 1].imshow(error_map)
        axes[row2_start + 1].set_title("Error Map (G=TP, R=FP, B=FN)", fontsize=12)
        axes[row2_start + 1].axis("off")

        # Polygon overlay
        polygons = prediction_to_polygons(prediction, threshold=threshold, min_area=100)
        polygon_img = draw_polygons_on_image(rgb, polygons, color=(255, 255, 0), thickness=2)
        axes[row2_start + 2].imshow(polygon_img)
        axes[row2_start + 2].set_title("Detected Polygons", fontsize=12)
        axes[row2_start + 2].axis("off")

        # Metrics text in last cell
        dice = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)
        iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + 1e-8)
        axes[row2_start + 3].text(
            0.5,
            0.5,
            f"Dice: {dice:.3f}\nIoU: {iou:.3f}\n\nTP: {tp.sum():,}\nFP: {fp.sum():,}\nFN: {fn.sum():,}",
            ha="center",
            va="center",
            fontsize=14,
            family="monospace",
        )
        axes[row2_start + 3].set_title("Metrics", fontsize=12)
        axes[row2_start + 3].axis("off")
    else:
        # Prediction overlay
        pred_overlay = rgb.copy().astype(np.float32) / 255.0
        pred_mask = prediction > threshold
        pred_overlay[pred_mask] = [1.0, 0.3, 0.3]
        axes[row2_start + 1].imshow(pred_overlay)
        axes[row2_start + 1].set_title("Prediction Overlay", fontsize=12)
        axes[row2_start + 1].axis("off")

        # Polygons
        polygons = prediction_to_polygons(prediction, threshold=threshold, min_area=100)
        polygon_img = draw_polygons_on_image(rgb, polygons, color=(255, 255, 0), thickness=2)
        axes[row2_start + 2].imshow(polygon_img)
        axes[row2_start + 2].set_title("Detected Polygons", fontsize=12)
        axes[row2_start + 2].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved visualization to {output_path}")

    plt.close(fig)


def main() -> None:
    """Generate visualizations for test cemeteries."""
    # Paths
    data_root = Path("data/real")
    output_dir = Path("outputs/full_cemetery_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find latest checkpoint
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        log.error("No checkpoints found in outputs/checkpoints/")
        return

    # Use 'best.ckpt' if available, otherwise latest
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        checkpoint_path = best_ckpt
    else:
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    log.info(f"Using checkpoint: {checkpoint_path}")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load model
    model, in_channels = load_checkpoint(checkpoint_path, device)
    log.info(f"Model expects {in_channels} input channels")

    # Process test cemeteries
    test_cemeteries = ["cemetery_09", "cemetery_10"]

    for cemetery_id in test_cemeteries:
        log.info(f"\nProcessing {cemetery_id}...")

        ortho_path = data_root / "orthophotos" / f"{cemetery_id}_ortho.tif"
        mask_path = data_root / "masks" / f"{cemetery_id}_mask.tif"
        local_height_path = data_root / "dems" / f"{cemetery_id}_dem_local_height.tif"
        slope_path = data_root / "dems" / f"{cemetery_id}_dem_slope.tif"

        if not ortho_path.exists():
            log.warning(f"Orthophoto not found: {ortho_path}")
            continue

        if not local_height_path.exists():
            log.warning(f"Local height DEM not found for {cemetery_id}")
            continue

        # Run inference
        result = predict_full_cemetery(
            model=model,
            orthophoto_path=ortho_path,
            local_height_path=local_height_path,
            slope_path=slope_path if slope_path.exists() else None,
            in_channels=in_channels,
            tile_size=512,
            overlap=0.15,
            device=device,
        )

        # Load mask if available
        mask = None
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)

        # Create visualization
        output_path = output_dir / f"{cemetery_id}_enhanced.png"
        create_enhanced_visualization(
            rgb=result["rgb"],
            local_height=result["local_height"],
            slope=result["slope"],
            prediction=result["prediction"],
            mask=mask,
            threshold=0.5,
            output_path=output_path,
            title=f"{cemetery_id} - Full Cemetery Analysis",
        )

    log.info(f"\nDone! Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
