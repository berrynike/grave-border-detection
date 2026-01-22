"""Lightning DataModule for grave border segmentation."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from grave_border_detection.data.dataset import (
    GraveDataset,
    TileReference,
    build_tile_index,
    get_train_transforms,
    get_val_transforms,
)


class GraveDataModule(L.LightningDataModule):
    """Lightning DataModule for grave border segmentation.

    Handles data loading, train/val/test splits, and creates DataLoaders.
    Split is done by cemetery to prevent data leakage.
    """

    def __init__(
        self,
        data_root: str | Path,
        train_cemeteries: list[str],
        val_cemeteries: list[str],
        test_cemeteries: list[str] | None = None,
        tile_size: int = 512,
        overlap: float = 0.15,
        min_mask_coverage: float = 0.0,
        use_dem: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize_rgb: bool = True,
        rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize DataModule.

        Args:
            data_root: Root directory containing orthophotos/, masks/, dems/ subdirs.
            train_cemeteries: List of cemetery IDs for training.
            val_cemeteries: List of cemetery IDs for validation.
            test_cemeteries: List of cemetery IDs for testing (optional).
            tile_size: Size of tiles in pixels.
            overlap: Overlap fraction between tiles.
            min_mask_coverage: Minimum mask coverage to include tile.
            use_dem: Whether to include DEM as 4th channel.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            pin_memory: Whether to pin memory for DataLoaders.
            normalize_rgb: Whether to normalize RGB channels.
            rgb_mean: Mean for RGB normalization.
            rgb_std: Std for RGB normalization.
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.train_cemeteries = train_cemeteries
        self.val_cemeteries = val_cemeteries
        self.test_cemeteries = test_cemeteries or []

        self.tile_size = tile_size
        self.overlap = overlap
        self.min_mask_coverage = min_mask_coverage
        self.use_dem = use_dem

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.normalize_rgb = normalize_rgb
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        # Will be set in setup()
        self.train_dataset: GraveDataset | None = None
        self.val_dataset: GraveDataset | None = None
        self.test_dataset: GraveDataset | None = None

    @property
    def orthophotos_dir(self) -> Path:
        """Return path to orthophotos directory."""
        return self.data_root / "orthophotos"

    @property
    def masks_dir(self) -> Path:
        """Return path to masks directory."""
        return self.data_root / "masks"

    @property
    def dems_dir(self) -> Path | None:
        """Return path to DEMs directory if use_dem is True."""
        if self.use_dem:
            return self.data_root / "dems"
        return None

    def _build_dataset(
        self,
        cemetery_ids: list[str],
        transform: Any | None = None,
    ) -> GraveDataset:
        """Build a dataset for given cemeteries.

        Args:
            cemetery_ids: List of cemetery IDs to include.
            transform: Albumentations transform to apply.

        Returns:
            GraveDataset instance.
        """
        tile_refs = build_tile_index(
            cemetery_ids=cemetery_ids,
            orthophotos_dir=self.orthophotos_dir,
            masks_dir=self.masks_dir,
            dems_dir=self.dems_dir,
            tile_size=self.tile_size,
            overlap=self.overlap,
            min_mask_coverage=self.min_mask_coverage,
        )

        return GraveDataset(
            tile_refs=tile_refs,
            tile_size=self.tile_size,
            use_dem=self.use_dem,
            transform=transform,
            normalize_rgb=self.normalize_rgb,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(
                self.train_cemeteries,
                transform=get_train_transforms(),
            )
            self.val_dataset = self._build_dataset(
                self.val_cemeteries,
                transform=get_val_transforms(),
            )

        if stage in ("test", None) and self.test_cemeteries:
            self.test_dataset = self._build_dataset(
                self.test_cemeteries,
                transform=get_val_transforms(),
            )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return training DataLoader."""
        if self.train_dataset is None:
            msg = "Train dataset not set up. Call setup('fit') first."
            raise RuntimeError(msg)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            msg = "Validation dataset not set up. Call setup('fit') first."
            raise RuntimeError(msg)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return test DataLoader."""
        if self.test_dataset is None:
            msg = "Test dataset not set up. Call setup('test') first."
            raise RuntimeError(msg)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_tile_refs(self, split: str) -> list[TileReference]:
        """Get tile references for a specific split.

        Useful for inference or visualization.

        Args:
            split: One of 'train', 'val', or 'test'.

        Returns:
            List of TileReference objects.
        """
        datasets = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        dataset = datasets.get(split)
        if dataset is None:
            msg = f"Dataset for '{split}' not set up."
            raise RuntimeError(msg)

        return dataset.tile_refs

    @property
    def num_train_samples(self) -> int:
        """Return number of training samples."""
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset)

    @property
    def num_val_samples(self) -> int:
        """Return number of validation samples."""
        if self.val_dataset is None:
            return 0
        return len(self.val_dataset)

    @property
    def num_test_samples(self) -> int:
        """Return number of test samples."""
        if self.test_dataset is None:
            return 0
        return len(self.test_dataset)

    @property
    def input_channels(self) -> int:
        """Return number of input channels (3 for RGB, 4 with DEM)."""
        return 4 if self.use_dem else 3
