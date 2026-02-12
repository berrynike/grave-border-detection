"""Tiling utilities for extracting patches from large GeoTIFFs.

Supports:
- Extracting tiles with configurable overlap
- Windowed reading for memory efficiency
- Preserving georeferencing
- Reconstructing full images from tiles (for inference)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window


@dataclass(frozen=True)
class TileInfo:
    """Metadata for a single tile."""

    row: int  # Tile row index
    col: int  # Tile column index
    window: Window  # Rasterio window for reading
    x_offset: int  # Pixel offset from image origin
    y_offset: int  # Pixel offset from image origin
    width: int  # Tile width in pixels
    height: int  # Tile height in pixels


def calculate_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: float = 0.0,
) -> list[TileInfo]:
    """Calculate tile positions for covering an image.

    Args:
        image_width: Width of source image in pixels.
        image_height: Height of source image in pixels.
        tile_size: Size of each tile (square tiles).
        overlap: Overlap fraction between adjacent tiles (0.0 to 0.5).

    Returns:
        List of TileInfo objects describing each tile.
    """
    if not 0.0 <= overlap < 0.5:
        raise ValueError(f"Overlap must be in [0, 0.5), got {overlap}")

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got {stride}")

    tiles: list[TileInfo] = []

    # Calculate number of tiles needed
    n_cols = max(1, int(np.ceil((image_width - tile_size) / stride)) + 1)
    n_rows = max(1, int(np.ceil((image_height - tile_size) / stride)) + 1)

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate tile position
            x_offset = min(col * stride, image_width - tile_size)
            y_offset = min(row * stride, image_height - tile_size)

            # Clamp to image bounds
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

            # Calculate actual tile size (may be smaller at edges)
            width = min(tile_size, image_width - x_offset)
            height = min(tile_size, image_height - y_offset)

            window = Window(
                col_off=x_offset,
                row_off=y_offset,
                width=width,
                height=height,
            )

            tiles.append(
                TileInfo(
                    row=row,
                    col=col,
                    window=window,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=width,
                    height=height,
                )
            )

    return tiles


def read_tile(
    src: rasterio.DatasetReader,
    tile: TileInfo,
    pad_to_size: int | None = None,
) -> NDArray[np.float32]:
    """Read a single tile from an open rasterio dataset.

    Args:
        src: Open rasterio dataset.
        tile: TileInfo describing the tile to read.
        pad_to_size: If provided, pad tile to this size (for edge tiles).

    Returns:
        Array of shape (C, H, W) with tile data.
    """
    data = src.read(window=tile.window)

    # Pad if needed (for edge tiles that are smaller than tile_size)
    if pad_to_size is not None and (data.shape[1] < pad_to_size or data.shape[2] < pad_to_size):
        padded = np.zeros(
            (data.shape[0], pad_to_size, pad_to_size),
            dtype=data.dtype,
        )
        padded[:, : data.shape[1], : data.shape[2]] = data
        data = padded

    result: NDArray[np.float32] = data.astype(np.float32)
    return result


def extract_tiles_from_geotiff(
    path: Path,
    tile_size: int,
    overlap: float = 0.0,
    pad_to_size: bool = True,
) -> tuple[list[NDArray[np.float32]], list[TileInfo]]:
    """Extract all tiles from a GeoTIFF file.

    Args:
        path: Path to GeoTIFF file.
        tile_size: Size of each tile (square tiles).
        overlap: Overlap fraction between adjacent tiles.
        pad_to_size: Whether to pad edge tiles to tile_size.

    Returns:
        Tuple of (list of tile arrays, list of TileInfo).
    """
    with rasterio.open(path) as src:
        tiles_info = calculate_tile_grid(
            image_width=src.width,
            image_height=src.height,
            tile_size=tile_size,
            overlap=overlap,
        )

        tiles_data = []
        for tile in tiles_info:
            data = read_tile(
                src,
                tile,
                pad_to_size=tile_size if pad_to_size else None,
            )
            tiles_data.append(data)

    return tiles_data, tiles_info


def reconstruct_from_tiles(
    tiles: list[NDArray[np.float32]],
    tiles_info: list[TileInfo],
    output_shape: tuple[int, int, int],
    blend_mode: str = "mean",
) -> NDArray[np.float32]:
    """Reconstruct full image from tiles.

    Handles overlapping regions by blending.

    Args:
        tiles: List of tile arrays (C, H, W).
        tiles_info: List of TileInfo matching tiles.
        output_shape: Shape of output image (C, H, W).
        blend_mode: How to blend overlapping regions ("mean" or "max").

    Returns:
        Reconstructed image array.
    """
    if blend_mode not in ("mean", "max"):
        raise ValueError(f"blend_mode must be 'mean' or 'max', got {blend_mode}")

    c, h, w = output_shape
    output = np.zeros((c, h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for tile_data, tile_info in zip(tiles, tiles_info, strict=True):
        y1 = tile_info.y_offset
        y2 = min(y1 + tile_info.height, h)
        x1 = tile_info.x_offset
        x2 = min(x1 + tile_info.width, w)

        # Handle padded tiles
        tile_h = y2 - y1
        tile_w = x2 - x1

        if blend_mode == "mean":
            output[:, y1:y2, x1:x2] += tile_data[:, :tile_h, :tile_w]
            weights[y1:y2, x1:x2] += 1.0
        else:  # max
            output[:, y1:y2, x1:x2] = np.maximum(
                output[:, y1:y2, x1:x2],
                tile_data[:, :tile_h, :tile_w],
            )
            weights[y1:y2, x1:x2] = 1.0  # Just mark as covered

    # Normalize by weights for mean blending
    if blend_mode == "mean":
        weights = np.maximum(weights, 1e-8)  # Avoid division by zero
        output = output / weights[np.newaxis, :, :]

    return output


def filter_tiles_by_mask_coverage(
    tiles_info: list[TileInfo],
    mask_path: Path,
    min_coverage: float = 0.01,
    tile_size: int = 512,
) -> list[TileInfo]:
    """Filter tiles to keep only those with sufficient mask coverage.

    Useful for training to skip empty tiles.

    Args:
        tiles_info: List of TileInfo to filter.
        mask_path: Path to mask GeoTIFF.
        min_coverage: Minimum fraction of tile that must be foreground.
        tile_size: Tile size for padding calculation.

    Returns:
        Filtered list of TileInfo.
    """
    filtered = []

    with rasterio.open(mask_path) as src:
        for tile in tiles_info:
            data = read_tile(src, tile, pad_to_size=tile_size)
            coverage = (data > 0).mean()
            if coverage >= min_coverage:
                filtered.append(tile)

    return filtered


def filter_tiles_by_valid_dem(
    tiles_info: list[TileInfo],
    valid_mask_path: Path,
    min_valid_coverage: float = 0.95,
    tile_size: int = 512,
) -> list[TileInfo]:
    """Filter tiles to keep only those with sufficient valid DEM coverage.

    Use this to exclude tiles that fall outside the DEM coverage area.

    Args:
        tiles_info: List of TileInfo to filter.
        valid_mask_path: Path to valid mask GeoTIFF (1=valid DEM, 0=nodata).
        min_valid_coverage: Minimum fraction of tile that must have valid DEM data.
        tile_size: Tile size for padding calculation.

    Returns:
        Filtered list of TileInfo.
    """
    filtered = []

    with rasterio.open(valid_mask_path) as src:
        for tile in tiles_info:
            data = read_tile(src, tile, pad_to_size=tile_size)
            valid_coverage = float((data > 0).mean())
            if valid_coverage >= min_valid_coverage:
                filtered.append(tile)

    return filtered


def get_tile_transform(
    src_transform: rasterio.Affine,
    tile: TileInfo,
) -> rasterio.Affine:
    """Get the geotransform for a specific tile.

    Args:
        src_transform: Transform of the source image.
        tile: TileInfo for the tile.

    Returns:
        Affine transform for the tile.
    """
    return src_transform * rasterio.Affine.translation(
        tile.x_offset,
        tile.y_offset,
    )
