"""Lightning DataModule for grave border segmentation."""

import logging
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import rasterio
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from grave_border_detection.data.dataset import (
    GraveDataset,
    TileReference,
    build_tile_index,
    get_train_transforms,
    get_val_transforms,
)
from grave_border_detection.preprocessing import normalize_dem

logger = logging.getLogger(__name__)


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
        dem_normalization_method: str = "zscore",
        dem_normalization_params: dict[str, float | int] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool | None = None,
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
            dem_normalization_method: DEM normalization method (zscore, percentile_clip,
                local_height, slope, robust_zscore).
            dem_normalization_params: Method-specific parameters for DEM normalization.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            pin_memory: Whether to pin memory for DataLoaders.
                If None (default), uses a sensible device-aware default:
                - False on MPS (pinning not supported, avoids warnings)
                - True on CUDA/CPU.
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
        self.dem_normalization_method = dem_normalization_method
        self.dem_normalization_params = dem_normalization_params or {}

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Make pin_memory device-aware:
        # - On MPS, pinning is not supported and generates warnings.
        # - On CUDA/CPU, pinning is typically beneficial.
        if pin_memory is None:
            self.pin_memory = not torch.backends.mps.is_available()
        else:
            self.pin_memory = pin_memory

        self.normalize_rgb = normalize_rgb
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        # Will be set in setup()
        self.train_dataset: GraveDataset | None = None
        self.val_dataset: GraveDataset | None = None
        self.test_dataset: GraveDataset | None = None

        # Cache for normalized full-cemetery DEMs (populated in setup)
        self._normalized_dem_cache: dict[str, NDArray[np.float32]] = {}

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

    @property
    def valid_masks_dir(self) -> Path | None:
        """Return path to valid masks directory if use_dem is True."""
        if self.use_dem:
            valid_dir = self.data_root / "valid_masks"
            if valid_dir.exists():
                return valid_dir
        return None

    def _load_and_normalize_dem(self, cemetery_id: str) -> NDArray[np.float32] | None:
        """Load full cemetery DEM and apply normalization.

        First checks for precomputed normalized DEM files (e.g., cemetery_01_dem_slope.tif).
        Falls back to runtime normalization if precomputed file not found.

        For method="local_height_slope", loads both local_height and slope and stacks them
        as a 2-channel array (shape: 2, H, W).

        Results are cached in memory for the duration of training.

        Args:
            cemetery_id: Cemetery ID (e.g., "cemetery_01").

        Returns:
            Normalized DEM array or None if DEM not found.
            Shape is (H, W) for single methods, (2, H, W) for combined.
        """
        if cemetery_id in self._normalized_dem_cache:
            return self._normalized_dem_cache[cemetery_id]

        if self.dems_dir is None:
            return None

        # Handle dual-channel DEM modes
        dual_channel_modes = ("local_height_slope", "zscore_slope", "zscore_local_height")
        if self.dem_normalization_method in dual_channel_modes:
            # Determine which two channels to load
            if self.dem_normalization_method == "local_height_slope":
                channel1_name, channel2_name = "local_height", "slope"
            elif self.dem_normalization_method == "zscore_slope":
                channel1_name, channel2_name = "zscore", "slope"
            elif self.dem_normalization_method == "zscore_local_height":
                channel1_name, channel2_name = "zscore", "local_height"
            else:
                raise ValueError(f"Unknown dual-channel mode: {self.dem_normalization_method}")

            # Load first channel
            if channel1_name == "zscore":
                # Zscore needs runtime computation from raw DEM
                raw_dem_path = self.dems_dir / f"{cemetery_id}_dem.tif"
                if not raw_dem_path.exists():
                    logger.warning(f"Raw DEM not found for {cemetery_id}: {raw_dem_path}")
                    return None
                with rasterio.open(raw_dem_path) as src:
                    raw_dem = src.read(1).astype(np.float32)
                channel1 = (raw_dem - raw_dem.mean()) / (raw_dem.std() + 1e-8)
            else:
                channel1_path = self.dems_dir / f"{cemetery_id}_dem_{channel1_name}.tif"
                if not channel1_path.exists():
                    logger.warning(f"Precomputed {channel1_name} DEM not found: {channel1_path}")
                    return None
                with rasterio.open(channel1_path) as src:
                    channel1 = src.read(1).astype(np.float32)

            # Load second channel
            if channel2_name == "zscore":
                raw_dem_path = self.dems_dir / f"{cemetery_id}_dem.tif"
                if not raw_dem_path.exists():
                    logger.warning(f"Raw DEM not found for {cemetery_id}: {raw_dem_path}")
                    return None
                with rasterio.open(raw_dem_path) as src:
                    raw_dem = src.read(1).astype(np.float32)
                channel2 = (raw_dem - raw_dem.mean()) / (raw_dem.std() + 1e-8)
            else:
                channel2_path = self.dems_dir / f"{cemetery_id}_dem_{channel2_name}.tif"
                if not channel2_path.exists():
                    logger.warning(f"Precomputed {channel2_name} DEM not found: {channel2_path}")
                    return None
                with rasterio.open(channel2_path) as src:
                    channel2 = src.read(1).astype(np.float32)

            logger.info(f"Loading {self.dem_normalization_method} DEM for {cemetery_id}")
            dem_stacked = np.stack([channel1, channel2], axis=0)
            self._normalized_dem_cache[cemetery_id] = dem_stacked
            return dem_stacked

        # Try precomputed normalized DEM first (much faster)
        precomputed_path = self.dems_dir / f"{cemetery_id}_dem_{self.dem_normalization_method}.tif"
        if precomputed_path.exists():
            logger.info(
                f"Loading precomputed {self.dem_normalization_method} DEM for {cemetery_id}"
            )
            with rasterio.open(precomputed_path) as src:
                dem_normalized: NDArray[np.float32] = src.read(1).astype(np.float32)
            self._normalized_dem_cache[cemetery_id] = dem_normalized
            return dem_normalized

        # Fall back to runtime normalization
        dem_files = list(self.dems_dir.glob(f"{cemetery_id}_dem.tif"))
        if not dem_files:
            logger.warning(f"No DEM found for {cemetery_id}")
            return None

        dem_path = dem_files[0]

        with rasterio.open(dem_path) as src:
            dem_raw = src.read(1).astype(np.float32)

        # Apply normalization to full DEM
        logger.info(
            f"Normalizing DEM for {cemetery_id} with method={self.dem_normalization_method} (runtime)"
        )
        dem_normalized = normalize_dem(
            dem_raw,
            method=self.dem_normalization_method,
            **self.dem_normalization_params,
        )

        self._normalized_dem_cache[cemetery_id] = dem_normalized
        return dem_normalized

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
            valid_masks_dir=self.valid_masks_dir,
            tile_size=self.tile_size,
            overlap=self.overlap,
            min_mask_coverage=self.min_mask_coverage,
        )

        return GraveDataset(
            tile_refs=tile_refs,
            tile_size=self.tile_size,
            use_dem=self.use_dem,
            normalized_dem_cache=self._normalized_dem_cache,
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
        # Pre-load and normalize DEMs for all cemeteries that will be used
        # This happens BEFORE building datasets so the cache is populated
        if self.use_dem:
            all_cemeteries: list[str] = []
            if stage in ("fit", None):
                all_cemeteries.extend(self.train_cemeteries)
                all_cemeteries.extend(self.val_cemeteries)
            if stage in ("test", None) and self.test_cemeteries:
                all_cemeteries.extend(self.test_cemeteries)

            # Remove duplicates while preserving order
            seen: set[str] = set()
            unique_cemeteries = [c for c in all_cemeteries if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

            for cemetery_id in unique_cemeteries:
                self._load_and_normalize_dem(cemetery_id)

            # Validate all cemeteries have DEMs loaded (especially important for local_height_slope mode)
            missing = [c for c in unique_cemeteries if c not in self._normalized_dem_cache]
            if missing:
                raise RuntimeError(
                    f"Failed to load DEMs for cemeteries: {missing}. "
                    f"For method='{self.dem_normalization_method}', ensure precomputed DEM files exist."
                )

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
        """Return number of input channels (3 for RGB, 4 with single DEM, 5 with dual DEM)."""
        if not self.use_dem:
            return 3
        dual_channel_modes = ("local_height_slope", "zscore_slope", "zscore_local_height")
        if self.dem_normalization_method in dual_channel_modes:
            return 5  # RGB + 2 DEM channels
        return 4  # RGB + single DEM channel
