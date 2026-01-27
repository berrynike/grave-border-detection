"""Lightning Module for semantic segmentation."""

from typing import Any, cast

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted probabilities (B, 1, H, W).
            target: Ground truth masks (B, 1, H, W).

        Returns:
            Scalar Dice loss.
        """
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Raw logits (B, 1, H, W).
            target: Ground truth masks (B, 1, H, W).

        Returns:
            Scalar combined loss.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)

        combined = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return cast("torch.Tensor", combined)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute segmentation metrics.

    Args:
        pred: Predicted probabilities (B, 1, H, W).
        target: Ground truth masks (B, 1, H, W).
        threshold: Threshold for binarization.

    Returns:
        Dictionary with dice and iou metrics.
    """
    pred_binary = (pred > threshold).float()

    # Flatten for easier computation
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # Dice coefficient
    dice = (2.0 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    # IoU (Jaccard)
    iou = (intersection + 1e-6) / (union + 1e-6)

    return {"dice": dice, "iou": iou}


class SegmentationModel(L.LightningModule):
    """Lightning Module for grave border segmentation.

    Uses segmentation_models_pytorch for the architecture.
    """

    def __init__(
        self,
        architecture: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        max_epochs: int = 100,
    ) -> None:
        """Initialize segmentation model.

        Args:
            architecture: SMP architecture name (Unet, FPN, etc.).
            encoder_name: Encoder backbone name.
            encoder_weights: Pretrained weights (imagenet or None).
            in_channels: Number of input channels (3 for RGB, 4 with DEM).
            classes: Number of output classes (1 for binary).
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            bce_weight: Weight for BCE loss.
            dice_weight: Weight for Dice loss.
            max_epochs: Max epochs for LR scheduler.
        """
        super().__init__()
        self.save_hyperparameters()

        # Build model using SMP
        model_class = getattr(smp, architecture)
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # Loss function
        self.loss_fn = CombinedLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        # Hyperparameters for optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Raw logits (B, 1, H, W).
        """
        return cast("torch.Tensor", self.model(x))

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """Shared step for train/val/test.

        Args:
            batch: Tuple of (images, masks).
            stage: One of 'train', 'val', or 'test'.

        Returns:
            Loss value.
        """
        images, masks = batch

        # Forward pass
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute metrics
        probs = torch.sigmoid(logits)
        metrics = compute_metrics(probs, masks)

        # Log metrics
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/dice", metrics["dice"], prog_bar=True, on_epoch=True)
        self.log(f"{stage}/iou", metrics["iou"], on_epoch=True)

        return cast("torch.Tensor", loss)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test")

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        _batch_idx: int,
    ) -> torch.Tensor:
        """Prediction step.

        Args:
            batch: Input images (or tuple with masks).
            batch_idx: Batch index.

        Returns:
            Predicted probabilities.
        """
        if isinstance(batch, tuple):
            images, _ = batch
        else:
            images = batch

        logits = self(images)
        return torch.sigmoid(logits)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.lr / 100,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract encoder features for visualization.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            List of feature maps from encoder stages.
        """
        return cast("list[torch.Tensor]", self.model.encoder(x))
