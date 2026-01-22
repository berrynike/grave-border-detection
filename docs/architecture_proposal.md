# Architecture Proposal: Grave Border Detection POC

## 0) Ziel

**Output:** Pro Friedhof ein Polygon pro Grab (GeoPackage/GeoJSON, EPSG:3857)

**Scope:** POC zur Validierung der technischen Machbarkeit

---

## 1) Projektstruktur & Pipeline

### 1.1 Config-Driven Pipeline (Hydra + Lightning)

Alles wird über YAML-Configs gesteuert — ein Command führt die gesamte Pipeline aus:

```bash
# Training starten
uv run python -m grave_border_detection.train experiment=baseline

# Hyperparameter-Sweep
uv run python -m grave_border_detection.train --multirun model.lr=1e-3,1e-4,1e-5

# Inference
uv run python -m grave_border_detection.predict experiment=baseline checkpoint=best
```

### 1.2 Config-Struktur

```
configs/
├── config.yaml                 # Hauptconfig (komponiert andere)
├── data/
│   └── default.yaml            # DataModule: paths, tile_size, batch_size
├── model/
│   ├── unet_resnet34.yaml      # U-Net mit ResNet34
│   └── unet_efficientnet.yaml  # U-Net mit EfficientNet
├── training/
│   └── default.yaml            # Trainer: epochs, callbacks, scheduler
├── augmentation/
│   └── default.yaml            # Albumentations pipeline
└── experiment/
    ├── baseline_rgb.yaml       # Nur RGB (Vergleich)
    ├── baseline_sam.yaml       # SAM zero-shot (Baseline)
    └── full.yaml               # RGB + DEM
```

### 1.3 Code-Struktur

```
src/grave_border_detection/
├── data/
│   ├── datamodule.py           # LightningDataModule
│   ├── dataset.py              # PyTorch Dataset
│   └── tiling.py               # Tile extraction
├── models/
│   ├── segmentation.py         # LightningModule wrapper
│   └── baseline_sam.py         # SAM zero-shot baseline
├── preprocessing/
│   ├── align_dem.py            # DEM auf Ortho-Grid resampling
│   └── rasterize_labels.py     # Polygone → Masken
├── postprocessing/
│   ├── instance_separation.py  # Connected components, watershed
│   └── vectorize.py            # Masken → Polygone
├── train.py                    # Hydra entrypoint
├── predict.py                  # Inference entrypoint
└── evaluate.py                 # Evaluation entrypoint
```

---

## 2) Daten

### 2.1 Verfügbare Daten

| Typ | Format | Anzahl | CRS |
|-----|--------|--------|-----|
| Orthophotos | GeoTIFF RGBA | 10 | EPSG:3857 |
| DEMs | GeoTIFF / GeoPackage tiles | 10 | EPSG:3857 |
| Annotationen | Shapefile (ausstehend) | 10 | EPSG:3857 |

### 2.2 Train/Val/Test Split

**Friedhofweise** (kein Data Leakage):

| Split | Friedhöfe | Anzahl |
|-------|-----------|--------|
| Train | 01, 02, 03, 04, 05, 06 | 6 |
| Val | 07, 08 | 2 |
| Test | 09, 10 | 2 |

```yaml
# configs/data/default.yaml
split:
  train: ["01", "02", "03", "04", "05", "06"]
  val: ["07", "08"]
  test: ["09", "10"]
```

### 2.3 Datenaufbereitung

**2.3.1 Orthophoto laden**
- GeoTIFF RGBA (uint8)
- RGB nutzen; Alpha nur wenn informativ
- Tools: `rasterio`

**2.3.2 DEM alignment**
- DEM auf exakt gleiches Raster wie Ortho resampling
- Gleiche: Auflösung, Bounds, CRS, Pixel-Grid
- Tools: `rasterio`, `gdalwarp`
- Output: `dem_aligned.tif` (float32)

**2.3.3 Ground Truth rasterisieren**
- Shapefile Polygone → Binärmaske (0/1)
- Auf Ortho-Grid
- Tools: `geopandas`, `shapely`, `rasterio.features.rasterize`

### 2.4 Tiling

```yaml
# configs/data/default.yaml
tiling:
  tile_size: 512
  overlap: 0.15  # 15%
  min_mask_coverage: 0.01  # Mindestens 1% Grab im Tile
```

- Angeschnittene Gräber sind OK
- Tools: `rasterio.windows`

---

## 3) Preprocessing & Augmentation

### 3.1 Normalisierung

```yaml
# configs/augmentation/default.yaml
normalize:
  rgb:
    method: "imagenet"  # oder "minmax", "dataset_stats"
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  dem:
    method: "zscore_per_cemetery"  # oder "global_minmax"
```

### 3.2 Augmentations (Albumentations)

```yaml
# configs/augmentation/default.yaml
augmentations:
  train:
    - RandomCrop:
        height: 512
        width: 512
    - HorizontalFlip:
        p: 0.5
    - VerticalFlip:
        p: 0.5
    - RandomRotate90:
        p: 0.5
    - RandomBrightnessContrast:
        brightness_limit: 0.1
        contrast_limit: 0.1
        p: 0.3
  val:
    - CenterCrop:
        height: 512
        width: 512
```

Tools: `albumentations` (synchronisiert Bild + Maske automatisch)

---

## 4) Baseline: SAM Zero-Shot

**Vor dem Training** → SAM als Baseline ohne Training:

```yaml
# configs/experiment/baseline_sam.yaml
baseline:
  type: "sam"
  model: "vit_b"
  checkpoint: "sam_vit_b_01ec64.pth"
  points_per_side: 32
  pred_iou_thresh: 0.88
```

**Zweck:**
- Zeigt was ohne Training erreichbar ist
- Vergleichspunkt für trainiertes Modell
- SAM wird über-segmentieren (findet alle Objekte, nicht nur Gräber)

Tools: `segment-anything`

---

## 5) Modell: U-Net mit Pretrained Encoder

### 5.1 Architektur

```yaml
# configs/model/unet_resnet34.yaml
model:
  architecture: "Unet"
  encoder_name: "resnet34"
  encoder_weights: "imagenet"
  in_channels: 4  # RGB + DEM (oder 3 für RGB-only)
  classes: 1      # Binäre Segmentierung
```

**Encoder-Optionen:**
| Encoder | Params | Empfehlung |
|---------|--------|------------|
| ResNet34 | 21M | Start (schnell) |
| EfficientNet-B3 | 12M | Upgrade |
| ConvNeXt-Tiny | 28M | Wenn mehr Power nötig |

### 5.2 RGB + DEM Integration

4 Input-Kanäle: `[R, G, B, H]`

Pretrained Encoder erwartet 3 Kanäle → Lösung:
- Erste Conv-Layer auf 4 Kanäle erweitern
- RGB-Gewichte aus Pretraining übernehmen
- DEM-Kanal: Random init oder Mean der RGB-Gewichte

`segmentation_models_pytorch` macht das automatisch mit `in_channels=4`.

### 5.3 Experimente

| Experiment | Input | Zweck |
|------------|-------|-------|
| `baseline_sam` | RGB | Zero-shot Baseline |
| `baseline_rgb` | RGB (3ch) | Trainiertes Modell ohne DEM |
| `full` | RGB+DEM (4ch) | Mehrwert von Höheninfo messen |

---

## 6) Training

### 6.1 Hyperparameter

```yaml
# configs/training/default.yaml
optimizer:
  type: "adamw"
  lr: 1e-4
  weight_decay: 1e-4

scheduler:
  type: "cosine_annealing"
  T_max: 100  # epochs
  eta_min: 1e-6

training:
  max_epochs: 100
  batch_size: 8
  accumulate_grad_batches: 2  # effektiv batch_size=16
  precision: "16-mixed"       # Mixed precision für M3

early_stopping:
  monitor: "val/dice"
  patience: 15
  mode: "max"

checkpointing:
  monitor: "val/dice"
  save_top_k: 3
  mode: "max"
```

### 6.2 Loss Function

```yaml
# configs/training/default.yaml
loss:
  type: "bce_dice"
  bce_weight: 0.5
  dice_weight: 0.5
  # Bei starkem Class-Imbalance:
  # type: "focal_dice"
  # focal_alpha: 0.25
  # focal_gamma: 2.0
```

### 6.3 Metrics

```yaml
# configs/training/default.yaml
metrics:
  - dice
  - iou
  - precision
  - recall
```

Nach Postprocessing zusätzlich objektbasiert:
- Anzahl erkannter Gräber vs. GT
- Merge/Split-Fehler

Tools: `torchmetrics`

### 6.4 Tracking (MLflow)

```yaml
# configs/training/default.yaml
logging:
  project: "grave-detection"
  log_every_n_steps: 10
  log_images_every_n_epochs: 5  # Overlay-Bilder
```

Was wird geloggt:
- Alle Hyperparameter (automatisch via Hydra)
- Train/Val Metrics pro Epoch
- Overlay-Bilder (Ortho + GT + Prediction)
- Model Checkpoints
- Confusion Matrix

---

## 7) Hardware

### 7.1 Lokale Entwicklung (M3 MacBook Air)

| Komponente | Spec | Ausreichend |
|------------|------|-------------|
| Chip | M3 (8 cores) | ✅ |
| RAM | 24GB unified | ✅ |
| GPU | MPS (Metal) | ✅ |

```python
# Automatische Device-Auswahl
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

**Erwartete Trainingszeit:** 2-4 Stunden pro Experiment

### 7.2 Cloud (falls nötig)

| Service | GPU | Kosten |
|---------|-----|--------|
| Google Colab | T4 | Gratis (limitiert) |
| Kaggle | P100 | Gratis (30h/Woche) |
| Lightning.ai | T4 | Gratis Tier |
| Vast.ai | Variabel | ~$0.30/h |

**Empfehlung:** Lokal starten, Cloud nur bei Bedarf.

---

## 8) Inference

### 8.1 Sliding Window

```yaml
# configs/predict.yaml
inference:
  tile_size: 512
  overlap: 0.25       # Mehr Overlap als Training für bessere Blending
  batch_size: 16
  tta: true           # Test-Time Augmentation (Flips)
  blend_mode: "mean"  # oder "max"
```

### 8.2 Output

- `prob_map.tif` (float32, georeferenziert) — Wahrscheinlichkeiten
- `mask_bin.tif` (uint8) — Binärmaske nach Threshold

---

## 9) Postprocessing: Instanz-Trennung

### 9.1 Pipeline

```yaml
# configs/postprocessing/default.yaml
postprocessing:
  threshold: 0.5
  morphology:
    closing_kernel: 5
    opening_kernel: 3
    hole_filling: true
  instance_separation:
    method: "connected_components"  # oder "watershed"
    min_area_m2: 0.5
    max_area_m2: 50.0
```

**Schritte:**
1. Threshold → Binärmaske
2. Morphologie (Closing, Hole Filling)
3. Connected Components → Label pro Grab
4. (Optional) Watershed falls Gräber zusammenkleben

Tools: `scipy.ndimage`, `skimage`

Output: `instance_labels.tif` (int32, 0..K)

---

## 10) Vektorisierung & Export

```yaml
# configs/postprocessing/default.yaml
vectorization:
  simplify_tolerance: 0.1  # Meter
  min_area_m2: 0.5
  output_format: "gpkg"    # oder "geojson"
```

**Schritte:**
1. `rasterio.features.shapes()` auf Instance Labels
2. `shapely` Cleanup (validity, simplify)
3. `geopandas` Export

**Output:** `graves_pred.gpkg`

| Attribut | Beschreibung |
|----------|--------------|
| `grave_id` | Eindeutige ID |
| `cemetery_id` | Friedhof-ID |
| `area_m2` | Fläche in m² |
| `confidence_mean` | Mittlere Konfidenz |
| `geometry` | Polygon (EPSG:3857) |

---

## 11) Evaluation & QC

### 11.1 Metriken

**Pixelweise:**
- Dice / IoU
- Precision / Recall

**Objektbasiert:**
- Anzahl Gräber (Predicted vs. GT)
- True Positives (IoU > 0.5 mit GT)
- False Positives / False Negatives
- Merge-Fehler (1 Pred = 2+ GT)
- Split-Fehler (2+ Pred = 1 GT)

### 11.2 Erfolgs-Kriterien

**POC-Ziel:** Machbarkeit validieren, nicht perfekte Performance.

| Metrik | Minimum | Gut | Sehr gut |
|--------|---------|-----|----------|
| Dice | > 0.6 | > 0.75 | > 0.85 |
| IoU | > 0.5 | > 0.65 | > 0.75 |

**Wichtiger:** Qualitative Inspektion der Ergebnisse.

### 11.3 Visualisierung

```yaml
# configs/evaluation/default.yaml
visualization:
  overlay_alpha: 0.4
  colors:
    gt: [0, 255, 0]       # Grün
    pred: [255, 0, 0]     # Rot
    overlap: [255, 255, 0] # Gelb
```

Tools: QGIS (Overlay GT + Prediction + Ortho)

---

## 12) Zusammenfassung: POC-Ablauf

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Daten vorbereiten                                            │
│    - DEM auf Ortho-Grid alignen                                 │
│    - Annotationen (Shapefile) → Raster-Masken                   │
│    - Tiling (512×512, 15% Overlap)                              │
├─────────────────────────────────────────────────────────────────┤
│ 2. Baseline: SAM Zero-Shot                                      │
│    - Keine Training nötig                                       │
│    - Etabliert Vergleichspunkt                                  │
├─────────────────────────────────────────────────────────────────┤
│ 3. Training: U-Net                                              │
│    - Experiment A: RGB-only                                     │
│    - Experiment B: RGB + DEM                                    │
│    - MLflow Tracking                                            │
├─────────────────────────────────────────────────────────────────┤
│ 4. Evaluation                                                   │
│    - Vergleich: SAM vs. RGB-only vs. RGB+DEM                    │
│    - Quantitativ (Dice/IoU) + Qualitativ (QGIS)                 │
├─────────────────────────────────────────────────────────────────┤
│ 5. Postprocessing & Export                                      │
│    - Instance Separation                                        │
│    - Vektorisierung → GeoPackage                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13) Abhängigkeiten

```toml
# pyproject.toml (zusätzlich zu bestehenden)
dependencies = [
    # Deep Learning
    "segmentation-models-pytorch>=0.3",
    "torchmetrics>=1.0",

    # Baseline
    "segment-anything>=1.0",

    # Tracking
    "mlflow>=2.10",

    # Config
    "hydra-core>=1.3",

    # Augmentation (bereits vorhanden)
    "albumentations>=1.3",
]
```

---

## 14) Offene Fragen / Risiken

| Risiko | Mitigation |
|--------|------------|
| Annotationen noch nicht da | Warten auf Shapefile von Vater |
| DEM niedriger aufgelöst als Ortho | Resampling; RGB-only Baseline als Fallback |
| Class Imbalance (wenig Grab-Pixel) | Focal Loss, Oversampling von Grab-Tiles |
| Gräber kleben zusammen | Watershed Postprocessing |
| M3 zu langsam | Cloud-Fallback (Colab/Kaggle) |
