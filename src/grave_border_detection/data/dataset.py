"""PyTorch Dataset for grave border segmentation."""

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from grave_border_detection.data.tiling import (
    TileInfo,
    calculate_tile_grid,
    filter_tiles_by_mask_coverage,
    filter_tiles_by_valid_dem,
    read_tile,
)


@dataclass
class TileReference:
    """Reference to a tile in a specific cemetery."""

    cemetery_id: str
    ortho_path: Path
    mask_path: Path
    dem_path: Path | None
    valid_mask_path: Path | None
    tile_info: TileInfo


def build_tile_index(
    cemetery_ids: list[str],
    orthophotos_dir: Path,
    masks_dir: Path,
    dems_dir: Path | None,
    valid_masks_dir: Path | None,
    tile_size: int,
    overlap: float,
    min_mask_coverage: float = 0.0,
    min_valid_dem_coverage: float = 0.95,
) -> list[TileReference]:
    """Build index of all tiles across multiple cemeteries.

    Args:
        cemetery_ids: List of cemetery IDs to include.
        orthophotos_dir: Directory containing orthophoto GeoTIFFs.
        masks_dir: Directory containing mask GeoTIFFs.
        dems_dir: Directory containing DEM GeoTIFFs (or None).
        valid_masks_dir: Directory containing valid DEM mask GeoTIFFs (or None).
        tile_size: Size of tiles in pixels.
        overlap: Overlap fraction between tiles.
        min_mask_coverage: Minimum mask coverage to include tile.
        min_valid_dem_coverage: Minimum valid DEM coverage to include tile (0.95 = 95%).

    Returns:
        List of TileReference objects.
    """
    tile_refs: list[TileReference] = []

    for cemetery_id in cemetery_ids:
        # Find files for this cemetery
        ortho_files = list(orthophotos_dir.glob(f"{cemetery_id}*.tif"))
        if not ortho_files:
            continue
        ortho_path = ortho_files[0]

        mask_files = list(masks_dir.glob(f"{cemetery_id}*.tif"))
        if not mask_files:
            continue
        mask_path = mask_files[0]

        dem_path = None
        if dems_dir is not None:
            dem_files = list(dems_dir.glob(f"{cemetery_id}*.tif"))
            if dem_files:
                dem_path = dem_files[0]

        valid_mask_path = None
        if valid_masks_dir is not None:
            valid_mask_files = list(valid_masks_dir.glob(f"{cemetery_id}*.tif"))
            if valid_mask_files:
                valid_mask_path = valid_mask_files[0]

        # Calculate tile grid based on orthophoto size
        with rasterio.open(ortho_path) as src:
            tiles = calculate_tile_grid(
                image_width=src.width,
                image_height=src.height,
                tile_size=tile_size,
                overlap=overlap,
            )

        # Filter tiles by mask coverage if needed
        if min_mask_coverage > 0:
            tiles = filter_tiles_by_mask_coverage(
                tiles,
                mask_path,
                min_coverage=min_mask_coverage,
                tile_size=tile_size,
            )

        # Filter tiles by valid DEM coverage if mask available
        if valid_mask_path is not None and min_valid_dem_coverage > 0:
            tiles = filter_tiles_by_valid_dem(
                tiles,
                valid_mask_path,
                min_valid_coverage=min_valid_dem_coverage,
                tile_size=tile_size,
            )

        # Create tile references
        for tile in tiles:
            tile_refs.append(
                TileReference(
                    cemetery_id=cemetery_id,
                    ortho_path=ortho_path,
                    mask_path=mask_path,
                    dem_path=dem_path,
                    valid_mask_path=valid_mask_path,
                    tile_info=tile,
                )
            )

    return tile_refs


class GraveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for grave border segmentation.

    Loads tiles from orthophotos and masks, optionally including DEM data.
    Applies augmentations using Albumentations.
    """

    def __init__(
        self,
        tile_refs: list[TileReference],
        tile_size: int = 512,
        use_dem: bool = True,
        normalized_dem_cache: dict[str, NDArray[np.float32]] | None = None,
        transform: A.Compose | None = None,
        normalize_rgb: bool = True,
        rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize dataset.

        Args:
            tile_refs: List of tile references to load.
            tile_size: Size of tiles (for padding).
            use_dem: Whether to include DEM as 4th channel.
            normalized_dem_cache: Pre-normalized full DEMs keyed by cemetery_id.
                If provided, tiles are extracted from these cached arrays instead
                of loading and normalizing per-tile.
            transform: Albumentations transform to apply.
            normalize_rgb: Whether to normalize RGB channels.
            rgb_mean: Mean for RGB normalization.
            rgb_std: Std for RGB normalization.
        """
        self.tile_refs = tile_refs
        self.tile_size = tile_size
        self.use_dem = use_dem
        self.normalized_dem_cache = normalized_dem_cache or {}
        self.transform = transform
        self.normalize_rgb = normalize_rgb
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.tile_refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.tile_refs[idx]

        # Load orthophoto (RGBA -> RGB)
        with rasterio.open(ref.ortho_path) as src:
            ortho = read_tile(src, ref.tile_info, pad_to_size=self.tile_size)
        rgb = ortho[:3]  # Take RGB, ignore alpha

        # Load mask
        with rasterio.open(ref.mask_path) as src:
            mask = read_tile(src, ref.tile_info, pad_to_size=self.tile_size)
        mask = mask[0]  # Remove channel dimension

        # Load DEM if requested
        # dem shape: (H, W) for single channel, (C, H, W) for combined
        dem: NDArray[np.float32] | None = None
        if self.use_dem and ref.dem_path is not None:
            # Check if we have pre-normalized DEM in cache
            if ref.cemetery_id in self.normalized_dem_cache:
                # Extract tile from pre-normalized full DEM
                full_dem = self.normalized_dem_cache[ref.cemetery_id]
                tile = ref.tile_info

                # Handle both single (H, W) and multi-channel (C, H, W) DEMs
                if full_dem.ndim == 2:
                    # Single channel DEM
                    dem = full_dem[
                        tile.y_offset : tile.y_offset + self.tile_size,
                        tile.x_offset : tile.x_offset + self.tile_size,
                    ].copy()

                    # Pad if needed (for edge tiles)
                    if dem.shape != (self.tile_size, self.tile_size):
                        padded: NDArray[np.float32] = np.zeros(
                            (self.tile_size, self.tile_size), dtype=np.float32
                        )
                        padded[: dem.shape[0], : dem.shape[1]] = dem
                        dem = padded
                else:
                    # Multi-channel DEM (e.g., combined local_height + slope)
                    dem = full_dem[
                        :,
                        tile.y_offset : tile.y_offset + self.tile_size,
                        tile.x_offset : tile.x_offset + self.tile_size,
                    ].copy()

                    # Pad if needed (for edge tiles)
                    if dem.shape[1:] != (self.tile_size, self.tile_size):
                        padded_multi: NDArray[np.float32] = np.zeros(
                            (full_dem.shape[0], self.tile_size, self.tile_size),
                            dtype=np.float32,
                        )
                        padded_multi[:, : dem.shape[1], : dem.shape[2]] = dem
                        dem = padded_multi
            else:
                # Fallback: load from file and apply per-tile z-score (legacy behavior)
                with rasterio.open(ref.dem_path) as src:
                    dem_raw = read_tile(src, ref.tile_info, pad_to_size=self.tile_size)
                dem_channel = dem_raw[0]  # Remove channel dimension
                dem = (dem_channel - dem_channel.mean()) / (dem_channel.std() + 1e-8)

        # Convert to HWC for albumentations
        rgb_hwc = rgb.transpose(1, 2, 0)  # CHW -> HWC

        # Normalize RGB to 0-1 range
        rgb_hwc = rgb_hwc / 255.0

        # Apply augmentations
        if self.transform is not None:
            if dem is not None:
                # For multi-channel DEM, we need to handle augmentation differently
                if dem.ndim == 3:
                    # Multi-channel DEM: augment each channel separately with same transform
                    # First channel (local_height)
                    transformed = self.transform(
                        image=rgb_hwc,
                        mask=mask,
                        dem=dem[0],
                        dem2=dem[1],
                    )
                    rgb_hwc = transformed["image"]
                    mask = transformed["mask"]
                    dem = np.stack([transformed["dem"], transformed["dem2"]], axis=0)
                else:
                    # Single channel DEM
                    transformed = self.transform(
                        image=rgb_hwc,
                        mask=mask,
                        dem=dem,
                    )
                    rgb_hwc = transformed["image"]
                    mask = transformed["mask"]
                    dem = transformed["dem"]
            else:
                transformed = self.transform(image=rgb_hwc, mask=mask)
                rgb_hwc = transformed["image"]
                mask = transformed["mask"]

        # Apply ImageNet normalization to RGB
        if self.normalize_rgb:
            rgb_hwc = (rgb_hwc - self.rgb_mean) / self.rgb_std

        # Convert back to CHW
        rgb_chw = rgb_hwc.transpose(2, 0, 1)  # HWC -> CHW

        # Combine RGB and DEM if using DEM
        if self.use_dem and dem is not None:
            if dem.ndim == 3:
                # Multi-channel DEM already in (C, H, W) format
                image = np.concatenate([rgb_chw, dem], axis=0)
            else:
                # Single channel DEM needs new axis
                image = np.concatenate([rgb_chw, dem[np.newaxis]], axis=0)
        else:
            image = rgb_chw

        # Convert to tensors
        image_tensor = torch.from_numpy(image.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)

        return image_tensor, mask_tensor


def get_train_transforms() -> A.Compose:
    """Get training augmentations.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3,
            ),
        ],
        additional_targets={"dem": "image", "dem2": "image"},
    )


def get_val_transforms() -> A.Compose:
    """Get validation augmentations (none).

    Returns:
        Albumentations Compose object (identity transform).
    """
    return A.Compose(
        [],
        additional_targets={"dem": "image", "dem2": "image"},
    )
