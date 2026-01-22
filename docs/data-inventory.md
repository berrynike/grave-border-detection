# Data Inventory

Last updated: 2026-01-21

## Summary

| Metric | Value |
|--------|-------|
| Cemeteries | 10 |
| Orthophotos | 10 |
| DEMs (GeoTIFF) | 2 |
| DEMs (GeoPackage tiles) | 10 |
| CRS | EPSG:3857 (Web Mercator) |
| CRS Consistent | Yes |
| Vector Annotations | **None yet** |

## Data Location

```
data/external/goetiff-orthos-ki-training-data/4k/
```

## Orthophotos

All orthophotos are 4-band RGBA GeoTIFFs (uint8) at approximately 4K resolution.

| ID | Cemetery | Size | Resolution | File Size |
|----|----------|------|------------|-----------|
| 01 | Weilkirchen, Zangberg | 4096×3920 | 0.019m | 63 MB |
| 02 | Kirchweg, Heldenstein | 4096×3828 | 0.027m | 45 MB |
| 03 | Lauterbach, Heldenstein | 4096×3925 | 0.022m | 45 MB |
| 04 | Kirchstr, Heldenstein | 4096×3624 | 0.026m | 58 MB |
| 05 | Salmannskirchen | 3134×4096 | 0.026m | 46 MB |
| 06 | Palmberg, Zangberg | 3736×4096 | 0.035m | 51 MB |
| 07 | Palmberg, Zangberg | 4096×3050 | 0.034m | 56 MB |
| 08 | Feldmochinger, München | 4096×2566 | 0.037m | 44 MB |
| 09 | St. Nikolaus | 4096×3445 | 0.045m | 41 MB |
| 10 | Westerbuchberg | 4049×4096 | 0.020m | 54 MB |

**Total orthophoto size: ~503 MB**

## DEMs (Digital Elevation Models)

### GeoTIFF DEMs (2 cemeteries)

| ID | Size | Resolution | File Size |
|----|------|------------|-----------|
| 01 | 1422×1361 | 0.055m | 6 MB |
| 02 | 1604×2043 | 0.049m | 9 MB |

### GeoPackage Tile DEMs (all 10 cemeteries)

DEMs are stored as GeoPackage tile pyramids with 2-3 zoom levels.

| ID | Tiles | Zoom Levels | File Size |
|----|-------|-------------|-----------|
| 01 | 5 | 2 | 2 MB |
| 02 | 5 | 2 | 4 MB |
| 03 | 5 | 2 | 3 MB |
| 04 | 5 | 2 | 5 MB |
| 05 | 5 | 2 | 4 MB |
| 06 | 14 | 3 | 8 MB |
| 07 | 14 | 3 | 9 MB |
| 08 | 9 | 3 | 8 MB |
| 09 | 15 | 3 | 10 MB |
| 10 | 6 | 2 | 4 MB |

## CRS Information

All data uses **EPSG:3857** (Web Mercator / Pseudo-Mercator).

- Consistent across all orthophotos and DEMs
- No reprojection needed for combining datasets
- Note: EPSG:3857 has metric distortion at high latitudes (Bavaria ~48°N has ~1.5x distortion)

## Annotations Status

**No vector annotations exist yet.**

The `.gpkg` files in the data directory are **raster DEM tiles**, not vector annotations. Grave border annotations need to be created before model training can begin.

### Annotation Requirements

- Format: GeoPackage or GeoJSON (vector polygons)
- CRS: EPSG:3857 (match source data)
- Geometry: Polygons representing grave border outlines
- Attributes: `grave_id`, `cemetery_id`, optional metadata

## Data Quality Notes

1. **Resolution varies**: 0.019m to 0.045m per pixel
2. **Image sizes vary**: ~3K to 4K pixels per dimension
3. **All imagery is georeferenced** with consistent CRS
4. **DEM coverage**: All cemeteries have elevation data as tile pyramids
5. **Naming convention**: `{ID}-{date}{location}-epsg3857-{params}.{ext}`

## Preprocessing Needs

Before training:

1. [ ] Create grave border annotations (manual labeling)
2. [ ] Generate training tiles (256×256 or 512×512)
3. [ ] Create train/val/test splits by cemetery
4. [ ] Normalize elevation data if using DEM channel
5. [ ] Verify spatial alignment between ortho and DEM
