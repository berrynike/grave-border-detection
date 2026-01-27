"""Callback for logging prediction images to MLflow during training.

Logs sample predictions to `epochs/epoch_XXX/` artifact folder.
For full test set visualization, see FullCemeteryVisualizationCallback.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger

log = logging.getLogger(__name__)


class ImageLoggerCallback(L.Callback):
    """Log sample predictions as images to MLflow during validation.

    Logs to `epochs/epoch_XXX/` with:
    - combined.png: Grid with all samples (one row per sample)
    - sample_N/: Individual component images for each sample

    Combined columns per row:
    - RGB-only (3ch): [RGB | Pred heatmap | Error map]
    - RGB+DEM (4ch):  [RGB | DEM | Pred heatmap | Error map]
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 5,
    ) -> None:
        """Initialize callback.

        Args:
            num_samples: Number of samples to log per epoch.
            log_every_n_epochs: Log images every N epochs.
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self._val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _outputs: Any,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Store validation batches for logging."""
        # Only collect during epochs we'll log
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Only store first batch
        if batch_idx == 0:
            images, masks = batch
            self._val_batches = [(images.detach().cpu(), masks.detach().cpu())]

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log prediction images at end of validation."""
        # Only log every N epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        if not self._val_batches:
            return

        # Get MLflow logger
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        # Get stored batch
        images, masks = self._val_batches[0]
        num_samples = min(self.num_samples, images.shape[0])

        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            device = next(pl_module.parameters()).device
            logits = pl_module(images[:num_samples].to(device))
            preds = torch.sigmoid(logits).cpu()

        # Create combined and individual images
        combined, individual = self._create_prediction_images(
            images[:num_samples],
            masks[:num_samples],
            preds,
        )

        # Log to MLflow under epochs/ folder
        epoch_name = f"epoch_{trainer.current_epoch:03d}"
        self._log_images_to_mlflow(mlflow_logger, combined, individual, epoch_name)

        # Clear stored batches
        self._val_batches = []

    def _get_mlflow_logger(self, trainer: L.Trainer) -> MLFlowLogger | None:
        """Get MLflow logger from trainer."""
        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        return None

    def _create_prediction_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Create combined grid and individual images for logging.

        Args:
            images: Input images (N, C, H, W) - 3 channels (RGB) or 4 (RGB+DEM).
            masks: Ground truth masks (N, 1, H, W).
            preds: Predicted probabilities (N, 1, H, W).

        Returns:
            Tuple of:
            - Combined image as tensor (3, H*N, W*num_cols) - one row per sample
            - Dict mapping component names to lists of individual images
        """
        n_samples = images.shape[0]
        has_dem = images.shape[1] == 4

        # Extract RGB (first 3 channels) and denormalize
        rgb = images[:, :3, :, :].clone()
        rgb = self._denormalize_rgb(rgb)

        # Extract and visualize DEM if present
        dem_viz = None
        if has_dem:
            dem = images[:, 3:4, :, :]
            dem_viz = self._visualize_dem(dem)

        # Binarize predictions
        pred_binary = (preds > 0.5).float()
        gt_binary = (masks > 0.5).float()

        # Collect individual images and grid rows
        individual: dict[str, list[torch.Tensor]] = {
            "rgb": [],
            "gt_overlay": [],
            "pred_heatmap": [],
            "error_map": [],
        }
        if has_dem:
            individual["dem"] = []

        rows = []
        for i in range(n_samples):
            gt = gt_binary[i, 0]  # (H, W)
            pred = pred_binary[i, 0]  # (H, W)
            img = rgb[i]  # (3, H, W)

            # RGB
            individual["rgb"].append(img)

            # DEM visualization (if present)
            if has_dem and dem_viz is not None:
                individual["dem"].append(dem_viz[i])

            # GT overlay (green) on RGB
            gt_overlay = img.clone()
            gt_mask = gt > 0.5
            gt_overlay[0, gt_mask] = 0.0  # R
            gt_overlay[1, gt_mask] = 1.0  # G
            gt_overlay[2, gt_mask] = 0.0  # B
            individual["gt_overlay"].append(gt_overlay)

            # Probability heatmap (black=0, red=0.5, yellow=1)
            prob = preds[i, 0]  # Raw probability 0-1
            prob_heatmap = torch.zeros(3, prob.shape[0], prob.shape[1])
            prob_heatmap[0] = prob  # Red channel = probability
            prob_heatmap[1] = prob * prob  # Green ramps up slower (yellow at high prob)
            prob_heatmap[2] = 0  # No blue
            individual["pred_heatmap"].append(prob_heatmap)

            # Error map on RGB
            # Green = True Positive (correct)
            # Red = False Positive (predicted but wrong)
            # Blue = False Negative (missed)
            error_map = img.clone()

            tp = (pred > 0.5) & (gt > 0.5)  # True positive - green
            fp = (pred > 0.5) & (gt <= 0.5)  # False positive - red
            fn = (pred <= 0.5) & (gt > 0.5)  # False negative - blue

            error_map[0, tp] = 0.0
            error_map[1, tp] = 1.0
            error_map[2, tp] = 0.0

            error_map[0, fp] = 1.0
            error_map[1, fp] = 0.0
            error_map[2, fp] = 0.0

            error_map[0, fn] = 0.0
            error_map[1, fn] = 0.0
            error_map[2, fn] = 1.0

            individual["error_map"].append(error_map)

            # Build row for combined image (one row per sample)
            # RGB-only: [RGB | Pred heatmap | Error map]
            # RGB+DEM:  [RGB | DEM | Pred heatmap | Error map]
            if has_dem and dem_viz is not None:
                row = torch.cat([img, dem_viz[i], prob_heatmap, error_map], dim=2)
            else:
                row = torch.cat([img, prob_heatmap, error_map], dim=2)
            rows.append(row)

        # Stack rows vertically (one row per sample)
        combined = torch.cat(rows, dim=1)

        return combined, individual

    def _visualize_dem(self, dem: torch.Tensor) -> torch.Tensor:
        """Visualize DEM as terrain colormap.

        Uses a terrain-like colormap:
        - Blue: low elevation
        - Green: mid elevation
        - Brown/tan: high elevation

        Args:
            dem: DEM tensor (N, 1, H, W).

        Returns:
            Colored DEM (N, 3, H, W) in [0, 1] range.
        """
        # Normalize to [0, 1] per sample
        dem_min = dem.amin(dim=(2, 3), keepdim=True)
        dem_max = dem.amax(dim=(2, 3), keepdim=True)
        dem_norm = (dem - dem_min) / (dem_max - dem_min + 1e-6)

        # Terrain colormap: interpolate blue -> green -> brown
        n, _, h, w = dem.shape
        result = torch.zeros(n, 3, h, w)

        t = dem_norm.squeeze(1)  # (N, H, W)

        # R: 0.1 -> 0.2 -> 0.7 (blue to green to brown)
        result[:, 0] = torch.where(t < 0.5, 0.1 + t * 0.2, 0.2 + (t - 0.5) * 1.0)
        # G: 0.2 -> 0.6 -> 0.4 (ramp up then down)
        result[:, 1] = torch.where(t < 0.5, 0.2 + t * 0.8, 0.6 - (t - 0.5) * 0.4)
        # B: 0.5 -> 0.2 -> 0.1 (decrease)
        result[:, 2] = torch.where(t < 0.5, 0.5 - t * 0.6, 0.2 - (t - 0.5) * 0.2)

        return torch.clamp(result, 0, 1)

    def _denormalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Denormalize RGB using ImageNet stats.

        Args:
            rgb: Normalized RGB tensor (N, 3, H, W).

        Returns:
            Denormalized RGB in [0, 1] range.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        rgb = rgb * std + mean
        return torch.clamp(rgb, 0, 1)

    def _log_images_to_mlflow(
        self,
        logger: MLFlowLogger,
        combined: torch.Tensor,
        individual: dict[str, list[torch.Tensor]],
        epoch_name: str,
    ) -> None:
        """Log combined grid and individual components to MLflow.

        Artifact structure:
            epochs/
              epoch_000/
                combined.png          # Grid with all samples
                sample_0/
                  rgb.png
                  dem.png             # (if available)
                  gt_overlay.png
                  pred_heatmap.png
                  error_map.png
                sample_1/
                  ...

        Args:
            logger: MLflow logger instance.
            combined: Combined grid image (3, H, W).
            individual: Dict mapping component names to lists of images.
            epoch_name: Epoch identifier (e.g., "epoch_000").
        """
        from mlflow.tracking import MlflowClient

        run_id = logger.run_id
        if not run_id:
            log.warning("No MLflow run_id, skipping image logging")
            return

        client = MlflowClient()
        epoch_artifact_path = f"epochs/{epoch_name}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Log combined image (grid with all samples)
            combined_path = tmpdir_path / "combined.png"
            self._save_image(self._tensor_to_numpy(combined), combined_path)
            client.log_artifact(run_id, str(combined_path), artifact_path=epoch_artifact_path)

            # Log individual components organized by sample
            # Get number of samples from first component
            first_component = next(iter(individual.values()))
            num_samples = len(first_component)

            for sample_idx in range(num_samples):
                sample_artifact_path = f"{epoch_artifact_path}/sample_{sample_idx}"

                for component_name, images in individual.items():
                    img_tensor = images[sample_idx]
                    img_path = tmpdir_path / f"{component_name}.png"
                    self._save_image(self._tensor_to_numpy(img_tensor), img_path)
                    client.log_artifact(run_id, str(img_path), artifact_path=sample_artifact_path)

        log.info(f"Logged predictions for {epoch_name} to MLflow (epochs/)")

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
            tensor: Image tensor (3, H, W) in [0, 1] range.

        Returns:
            Numpy array (H, W, 3) in uint8 [0, 255] range.
        """
        img_np = tensor.permute(1, 2, 0).numpy()
        return (img_np * 255).astype(np.uint8)

    def _save_image(self, img_np: np.ndarray, path: Path) -> None:
        """Save numpy array as PNG image.

        Args:
            img_np: Image array (H, W, 3) in uint8 format.
            path: Output path.
        """
        from PIL import Image

        img_pil = Image.fromarray(img_np)
        img_pil.save(path)
