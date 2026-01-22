#!/usr/bin/env python3
"""Quick inventory of the dataset without GDAL dependencies."""

import json
from pathlib import Path


def inventory_data():
    """Create a simple inventory of the data files."""
    data_dir = Path("data/external/goetiff-orthos-ki-training-data/4k")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"📁 Data Directory: {data_dir}")
    print("=" * 60)

    # Find different file types
    tif_files = list(data_dir.glob("*.tif"))
    gpkg_files = list(data_dir.glob("*.gpkg"))
    kml_files = list(data_dir.glob("*.kml"))

    # Separate orthophotos and DEMs
    orthophotos = [f for f in tif_files if "ortho" in f.name.lower() and "DEM" not in f.name]
    dems = [f for f in tif_files if "DEM" in f.name]

    print("\n📊 File Summary:")
    print(f"  • Orthophotos: {len(orthophotos)}")
    print(f"  • DEMs: {len(dems)}")
    print(f"  • GeoPackages: {len(gpkg_files)}")
    print(f"  • KML files: {len(kml_files)}")

    # Group by cemetery ID
    cemeteries = {}
    for f in orthophotos:
        cemetery_id = f.name[:2]
        if cemetery_id not in cemeteries:
            cemeteries[cemetery_id] = {"id": cemetery_id, "files": {}}
        cemeteries[cemetery_id]["files"]["orthophoto"] = f.name
        cemeteries[cemetery_id]["size_mb"] = round(f.stat().st_size / (1024 * 1024), 1)

    for f in dems:
        cemetery_id = f.name[:2]
        if cemetery_id not in cemeteries:
            cemeteries[cemetery_id] = {"id": cemetery_id, "files": {}}
        cemeteries[cemetery_id]["files"]["dem"] = f.name

    for f in gpkg_files:
        cemetery_id = f.name[:2]
        if cemetery_id not in cemeteries:
            cemeteries[cemetery_id] = {"id": cemetery_id, "files": {}}
        cemeteries[cemetery_id]["files"]["annotations"] = f.name

    print(f"\n🏛️ Cemetery Sites: {len(cemeteries)}")
    print("-" * 60)

    for cid, info in sorted(cemeteries.items()):
        files = info.get("files", {})
        print(f"\n{cid}: ")
        if "orthophoto" in files:
            print(f"  📷 Ortho: {files['orthophoto'][:40]}... ({info.get('size_mb', 0)} MB)")
        if "dem" in files:
            print(f"  ⛰️  DEM: {files['dem'][:40]}...")
        if "annotations" in files:
            print(f"  📍 Anno: {files['annotations'][:40]}...")

    # Check data completeness
    complete = sum(
        1
        for c in cemeteries.values()
        if all(k in c.get("files", {}) for k in ["orthophoto", "dem", "annotations"])
    )
    partial = len(cemeteries) - complete

    print("\n✅ Data Completeness:")
    print(f"  • Complete (all 3 files): {complete}/{len(cemeteries)}")
    print(f"  • Partial: {partial}/{len(cemeteries)}")

    # Save inventory
    inventory = {
        "total_cemeteries": len(cemeteries),
        "complete_datasets": complete,
        "cemeteries": cemeteries,
    }

    inventory_path = Path("data/quick_inventory.json")
    with inventory_path.open("w") as f:
        json.dump(inventory, f, indent=2, default=str)

    print(f"\n💾 Inventory saved to: {inventory_path}")


if __name__ == "__main__":
    inventory_data()
