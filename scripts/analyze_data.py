#!/usr/bin/env python3
"""Analyze and inventory the cemetery dataset."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_geotiff(filepath: Path) -> dict:
    """Analyze a GeoTIFF file and extract metadata."""
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            metadata = {
                "filename": filepath.name,
                "path": str(filepath),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs),
                "resolution": (src.res[0], src.res[1]),
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "size_mb": filepath.stat().st_size / (1024 * 1024),
            }

            # Check if it's a DEM or orthophoto
            if "DEM" in filepath.name or src.count == 1:
                metadata["type"] = "DEM"
                # Sample height statistics
                data = src.read(1)
                metadata["height_stats"] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                }
            else:
                metadata["type"] = "Orthophoto"

            return metadata
    except Exception as e:
        console.print(f"[red]Error reading {filepath.name}: {e}[/red]")
        return {"filename": filepath.name, "error": str(e)}


def analyze_geopackage(filepath: Path) -> dict:
    """Analyze a GeoPackage file and extract metadata."""
    try:
        gdf = gpd.read_file(filepath)
        metadata = {
            "filename": filepath.name,
            "path": str(filepath),
            "type": "Vector",
            "features": len(gdf),
            "crs": str(gdf.crs),
            "geometry_type": gdf.geometry.geom_type.unique().tolist(),
            "bounds": {
                "left": gdf.total_bounds[0],
                "bottom": gdf.total_bounds[1],
                "right": gdf.total_bounds[2],
                "top": gdf.total_bounds[3],
            },
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "columns": list(gdf.columns),
        }
        return metadata
    except Exception as e:
        console.print(f"[red]Error reading {filepath.name}: {e}[/red]")
        return {"filename": filepath.name, "error": str(e)}


def find_matching_files(data_dir: Path) -> dict[str, dict]:
    """Find and group related files (ortho, DEM, annotations)."""
    orthophotos = list(data_dir.glob("*ortho*.tif"))
    dems = list(data_dir.glob("*DEM*.tif"))
    annotations = list(data_dir.glob("*.gpkg"))

    # Group by cemetery ID (first 2 digits)
    groups = {}

    for ortho in orthophotos:
        cemetery_id = ortho.name[:2]
        if cemetery_id not in groups:
            groups[cemetery_id] = {"orthophoto": None, "dem": None, "annotations": None}
        groups[cemetery_id]["orthophoto"] = ortho

    for dem in dems:
        cemetery_id = dem.name[:2]
        if cemetery_id not in groups:
            groups[cemetery_id] = {"orthophoto": None, "dem": None, "annotations": None}
        groups[cemetery_id]["dem"] = dem

    for annot in annotations:
        cemetery_id = annot.name[:2]
        if cemetery_id not in groups:
            groups[cemetery_id] = {"orthophoto": None, "dem": None, "annotations": None}
        groups[cemetery_id]["annotations"] = annot

    return groups


def main():
    """Main analysis function."""
    data_dir = Path("data/external/goetiff-orthos-ki-training-data/4k")

    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        return

    console.print(f"[green]Analyzing data in: {data_dir}[/green]\n")

    # Find and group files
    groups = find_matching_files(data_dir)

    # Create summary table
    table = Table(title="Cemetery Dataset Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Cemetery", style="white")
    table.add_column("Orthophoto", style="green")
    table.add_column("DEM", style="blue")
    table.add_column("Annotations", style="yellow")
    table.add_column("CRS", style="magenta")

    inventory = {"cemeteries": {}}

    for cemetery_id, files in sorted(groups.items()):
        ortho_info = {}
        dem_info = {}
        annot_info = {}

        if files["orthophoto"]:
            ortho_info = analyze_geotiff(files["orthophoto"])
            ortho_status = f"✓ {ortho_info.get('width', '?')}x{ortho_info.get('height', '?')}"
        else:
            ortho_status = "✗"

        if files["dem"]:
            dem_info = analyze_geotiff(files["dem"])
            dem_status = f"✓ {dem_info.get('width', '?')}x{dem_info.get('height', '?')}"
        else:
            dem_status = "✗"

        if files["annotations"]:
            annot_info = analyze_geopackage(files["annotations"])
            annot_status = f"✓ {annot_info.get('features', '?')} features"
        else:
            annot_status = "✗"

        # Extract cemetery name from filename
        cemetery_name = "Unknown"
        if files["orthophoto"]:
            name_parts = files["orthophoto"].name.split("-")
            if len(name_parts) > 1:
                cemetery_name = (
                    name_parts[1].split("_")[0] if "_" in name_parts[1] else name_parts[1]
                )

        # Get CRS (should be consistent)
        crs = ortho_info.get("crs", dem_info.get("crs", annot_info.get("crs", "Unknown")))

        table.add_row(
            cemetery_id,
            cemetery_name[:20],
            ortho_status,
            dem_status,
            annot_status,
            crs.split(":")[-1] if ":" in crs else crs[:15],
        )

        # Store in inventory
        inventory["cemeteries"][cemetery_id] = {
            "name": cemetery_name,
            "orthophoto": ortho_info if ortho_info else None,
            "dem": dem_info if dem_info else None,
            "annotations": annot_info if annot_info else None,
        }

    console.print(table)

    # Print statistics
    console.print("\n[bold]Dataset Statistics:[/bold]")
    console.print(f"Total cemeteries: {len(groups)}")
    console.print(f"Orthophotos: {sum(1 for g in groups.values() if g['orthophoto'])}")
    console.print(f"DEMs: {sum(1 for g in groups.values() if g['dem'])}")
    console.print(f"Annotations: {sum(1 for g in groups.values() if g['annotations'])}")

    # Check CRS consistency
    all_crs = set()
    for cemetery in inventory["cemeteries"].values():
        if cemetery["orthophoto"] and "crs" in cemetery["orthophoto"]:
            all_crs.add(cemetery["orthophoto"]["crs"])
        if cemetery["dem"] and "crs" in cemetery["dem"]:
            all_crs.add(cemetery["dem"]["crs"])
        if cemetery["annotations"] and "crs" in cemetery["annotations"]:
            all_crs.add(cemetery["annotations"]["crs"])

    console.print(f"\nUnique CRS found: {all_crs}")

    if len(all_crs) > 1:
        console.print(
            "[yellow]⚠ Warning: Multiple CRS detected. Reprojection may be needed.[/yellow]"
        )

    # Save inventory to JSON
    inventory_path = Path("data/inventory.json")
    with inventory_path.open("w") as f:
        json.dump(inventory, f, indent=2, default=str)
    console.print(f"\n[green]✓ Inventory saved to: {inventory_path}[/green]")


if __name__ == "__main__":
    main()
