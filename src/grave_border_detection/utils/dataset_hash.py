"""Dataset versioning utilities.

Computes deterministic hash-based dataset IDs to track which data was used
for each training run. This enables reproducibility and change detection.
"""

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def compute_dataset_id(
    data_root: Path,
    train_cemeteries: list[str],
    val_cemeteries: list[str],
    test_cemeteries: list[str] | None = None,
) -> str:
    """Compute SHA256 hash of dataset file manifest.

    Hashes file paths, sizes, and modification times for all data files
    used in the current split. This provides a unique identifier that
    changes when data is added, removed, or modified.

    Args:
        data_root: Root directory containing orthophotos/, masks/, dems/.
        train_cemeteries: List of training cemetery IDs.
        val_cemeteries: List of validation cemetery IDs.
        test_cemeteries: Optional list of test cemetery IDs.

    Returns:
        8-character hash prefix (e.g., "a1b2c3d4").
    """
    if test_cemeteries is None:
        test_cemeteries = []

    manifest: list[dict[str, str | int]] = []
    all_cemeteries = set(train_cemeteries + val_cemeteries + test_cemeteries)

    # Check each data subdirectory
    for subdir in ["orthophotos", "masks", "dems"]:
        dir_path = data_root / subdir
        if not dir_path.exists():
            log.debug(f"Directory {dir_path} does not exist, skipping")
            continue

        # Find all files in this directory
        for file in sorted(dir_path.glob("*")):
            # Skip hidden files and directories
            if file.name.startswith(".") or not file.is_file():
                continue

            # Check if file belongs to any cemetery in our split
            # Match if cemetery ID appears in filename (e.g., "cemetery_1_ortho.tif")
            file_stem = file.stem.lower()
            belongs_to_split = any(
                cem.lower() in file_stem or file_stem.startswith(cem.lower())
                for cem in all_cemeteries
            )

            if belongs_to_split:
                stat = file.stat()
                manifest.append(
                    {
                        "path": str(file.relative_to(data_root)),
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                    }
                )

    # Create deterministic hash
    manifest_json = json.dumps(manifest, sort_keys=True)
    hash_full = hashlib.sha256(manifest_json.encode()).hexdigest()

    log.info(f"Computed dataset_id from {len(manifest)} files")
    return hash_full[:8]


def get_dataset_summary(data_root: Path) -> dict[str, int]:
    """Get summary statistics of dataset for logging.

    Args:
        data_root: Root directory containing orthophotos/, masks/, dems/.

    Returns:
        Dictionary with counts of each file type.
    """
    summary: dict[str, int] = {}

    for subdir, key in [
        ("orthophotos", "orthophoto_count"),
        ("masks", "mask_count"),
        ("dems", "dem_count"),
    ]:
        dir_path = data_root / subdir
        if dir_path.exists():
            # Count GeoTIFF files
            count = len(list(dir_path.glob("*.tif"))) + len(list(dir_path.glob("*.tiff")))
            summary[key] = count
        else:
            summary[key] = 0

    return summary
