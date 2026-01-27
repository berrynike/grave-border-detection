"""PyTorch Dataset for grave border segmentation."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray

from grave_border_detection.data.tiling import (
    TileInfo,
    calculate_tile_grid,
    filter_tiles_by_mask_coverage,
    read_tile,
)


@dataclass
class TileReference:
    """Reference to a tile in a specific cemetery."""

    cemetery_id: str
    ortho_path: Path
    mask_path: Path
    dem_path: Path | None
    tile_info: TileInfo


def build_tile_index(
    cemetery_ids: list[str],
    orthophotos_dir: Path,
    masks_dir: Path,
    dems_dir: Path | None,
    tile_size: int,
    overlap: float,
    min_mask_coverage: float = 0.0,
) -> list[TileReference]:
    """Build index of all tiles across multiple cemeteries.

    Args:
        cemetery_ids: List of cemetery IDs to include.
        orthophotos_dir: Directory containing orthophoto GeoTIFFs.
        masks_dir: Directory containing mask GeoTIFFs.
        dems_dir: Directory containing DEM GeoTIFFs (or None).
        tile_size: Size of tiles in pixels.
        overlap: Overlap fraction between tiles.
        min_mask_coverage: Minimum mask coverage to include tile.

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

        # Create tile references
        for tile in tiles:
            tile_refs.append(
                TileReference(
                    cemetery_id=cemetery_id,
                    ortho_path=ortho_path,
                    mask_path=mask_path,
                    dem_path=dem_path,
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
            transform: Albumentations transform to apply.
            normalize_rgb: Whether to normalize RGB channels.
            rgb_mean: Mean for RGB normalization.
            rgb_std: Std for RGB normalization.
        """
        self.tile_refs = tile_refs
        self.tile_size = tile_size
        self.use_dem = use_dem
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
        dem: NDArray[np.float32] | None = None
        if self.use_dem and ref.dem_path is not None:
            with rasterio.open(ref.dem_path) as src:
                dem_raw = read_tile(src, ref.tile_info, pad_to_size=self.tile_size)
            dem_channel = dem_raw[0]  # Remove channel dimension
            # Normalize DEM (z-score)
            dem = (dem_channel - dem_channel.mean()) / (dem_channel.std() + 1e-8)

        # Convert to HWC for albumentations
        rgb_hwc = rgb.transpose(1, 2, 0)  # CHW -> HWC

        # Normalize RGB to 0-1 range
        rgb_hwc = rgb_hwc / 255.0

        # Apply augmentations
        if self.transform is not None:
            if dem is not None:
                # Include DEM in augmentation for geometric transforms
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
        additional_targets={"dem": "image"},
    )


def get_val_transforms() -> A.Compose:
    """Get validation augmentations (none).

    Returns:
        Albumentations Compose object (identity transform).
    """
    return A.Compose(
        [],
        additional_targets={"dem": "image"},
    )
