"""Callback for logging prediction images to MLflow."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger


class ImageLoggerCallback(L.Callback):
    """Log sample predictions as images to MLflow.

    Creates a grid showing input RGB, ground truth mask, and predicted mask
    for a few validation samples.
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

        # Create and log grid image
        grid = self._create_prediction_grid(
            images[:num_samples],
            masks[:num_samples],
            preds,
        )

        # Log to MLflow
        self._log_image_to_mlflow(
            mlflow_logger,
            grid,
            f"predictions_epoch_{trainer.current_epoch:03d}",
        )

        # Clear stored batches
        self._val_batches = []

    def _get_mlflow_logger(self, trainer: L.Trainer) -> MLFlowLogger | None:
        """Get MLflow logger from trainer."""
        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        return None

    def _create_prediction_grid(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
    ) -> torch.Tensor:
        """Create a grid showing RGB, GT overlay, Pred overlay, and error map.

        Args:
            images: Input images (N, C, H, W) - can be 3 or 4 channels.
            masks: Ground truth masks (N, 1, H, W).
            preds: Predicted masks (N, 1, H, W).

        Returns:
            Grid image as tensor (3, H*N, W*4).
        """
        n_samples = images.shape[0]

        # Extract RGB (first 3 channels) and denormalize
        rgb = images[:, :3, :, :].clone()
        rgb = self._denormalize_rgb(rgb)

        # Binarize predictions
        pred_binary = (preds > 0.5).float()
        gt_binary = (masks > 0.5).float()

        rows = []
        for i in range(n_samples):
            gt = gt_binary[i, 0]  # (H, W)
            pred = pred_binary[i, 0]  # (H, W)
            img = rgb[i]  # (3, H, W)

            # Column 2: GT overlay (green) on RGB
            gt_overlay = img.clone()
            gt_mask = gt > 0.5
            gt_overlay[0, gt_mask] = 0.0  # R
            gt_overlay[1, gt_mask] = 1.0  # G
            gt_overlay[2, gt_mask] = 0.0  # B

            # Column 3: Probability heatmap (black=0, red=0.5, yellow=1)
            prob = preds[i, 0]  # Raw probability 0-1
            prob_heatmap = torch.zeros(3, prob.shape[0], prob.shape[1])
            prob_heatmap[0] = prob  # Red channel = probability
            prob_heatmap[1] = prob * prob  # Green ramps up slower (yellow at high prob)
            prob_heatmap[2] = 0  # No blue

            # Column 4: Error map on RGB
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

            # Stack horizontally: [RGB | GT overlay | Prob heatmap | Error map]
            row = torch.cat([img, gt_overlay, prob_heatmap, error_map], dim=2)
            rows.append(row)

        # Stack vertically
        grid = torch.cat(rows, dim=1)

        return grid

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

    def _log_image_to_mlflow(
        self,
        logger: MLFlowLogger,
        image: torch.Tensor,
        name: str,
    ) -> None:
        """Log image tensor to MLflow.

        Args:
            logger: MLflow logger instance.
            image: Image tensor (3, H, W) in [0, 1] range.
            name: Artifact name.
        """
        try:
            import tempfile

            import mlflow
            import numpy as np
            from PIL import Image

            # Convert to numpy HWC format
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Save as PNG
            img_pil = Image.fromarray(img_np)

            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = Path(tmpdir) / f"{name}.png"
                img_pil.save(img_path)

                # Log to MLflow
                run_id = logger.run_id
                if run_id:
                    mlflow.log_artifact(str(img_path), artifact_path="predictions")

        except Exception as e:
            # Don't fail training if image logging fails
            import logging

            logging.getLogger(__name__).warning(f"Failed to log image: {e}")
