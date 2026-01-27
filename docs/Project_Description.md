# PRD

# Product Requirements Document (PRD) - Friedhof Grabumrandungen Erkennung POC

## Projektzusammenfassung

Dieses Projekt zielt darauf ab, Drohnen-Luftbilder und Vermessungsdaten von Friedhöfen zu nutzen, um automatisch Grabumrandungen zu erkennen. Als erster Schritt entwickeln wir einen Proof of Concept (POC), der mittels Machine Learning die vorhandenen aufbereiteten Daten für ein Modelltraining verwendet.

## Projektziel

Automatisierte, reproduzierbare Erkennung und Vektorisierung von **Grabumrandungen** aus Drohnendaten, um manuelle Kartierung zu reduzieren und konsistente, GIS-kompatible Ergebnisse zu erzeugen. Die erkannten Daten sollen später in die Zielsoftware [HADES-X](https://hades-x.de/) integriert werden können.

## Datenbasis (Input)

### Primäre Eingabedaten

- **Georeferenzierte RGB-Orthofotos** (PNG oder GeoTIFF)
- **Höheninformation**
    - entweder als georeferenziertes Raster (z. B. DSM/DTM als GeoTIFF)
    - oder als Höhenkodierung pro Pixel (separates Höhenbild)

### Ground Truth / Annotationen

- **Vektorbasierte Auszeichnungen der Grabumrandungen**
    - bevorzugt: **GeoJSON** oder **GeoPackage**
    - alternativ: **Shapefile (SHP)**
- Gleicher **Koordinatenbezug** wie die Rasterdaten (z. B. WGS84 oder lokales UTM)

## Datenaufbereitung

1. **Koordinatenharmonisierung**
    - Alle Raster- und Vektordaten im selben CRS
2. **Rasterisierung der Vektordaten**
    - Umwandlung der Grabumrandungen in pixelgenaue Masken
3. **Feature-Zusammenführung**
    - RGB-Kanäle + Höhenkanal → Multi-Channel-Input
4. **Patch-Erstellung**
    - Kachelung der Bilder für Modelltraining

## Modell-PoC

- **Modelltyp**: CNN-basierte Segmentierung (z. B. U-Net-ähnlich)
- **Input**:
    - RGB (+ optional Höhenkanal)
- **Output**:
    - Pixelweise Segmentierung der Grabumrandungen
- **Training**:
    - Supervised Learning mit den erzeugten Masken
- **Optional im PoC**:
    - Vergleich **RGB-only vs. RGB + Höhe**, um Mehrwert der Höheninformation zu evaluieren

## Postprocessing & Output

1. **Binäre Segmentierung → Vektorisierung**
    - Umwandlung der Modellmaske in Polygone
2. **GIS-konformer Export**
    - GeoJSON / GeoPackage / SHP
3. **Qualitätsprüfung**
    - Overlay mit Original-Orthofoto
    - Stichprobenhafte manuelle Validierung

## Ziel des PoC

- Technische Machbarkeit validieren
- Einfluss der Höheninformation quantifizieren
- Abschätzen, ob das Verfahren robust genug für weitere Friedhöfe ist

## POC Scope

### Im Scope

- Training eines U-Net-ähnlichen CNN-Modells zur Segmentierung von Grabumrandungen
- Evaluierung mit und ohne Höheninformation
- Vektorisierung der Segmentierungsergebnisse
- Export als GIS-kompatible Formate
- Validierung der Erkennungsgenauigkeit anhand von Ground Truth Daten
- Dokumentation der Ergebnisse und Erkenntnisse

### Nicht im Scope (für späteren Ausbau)

- Vollständige Bestandserfassung aller Friedhofselemente
- Integration mit HADES-X Software
- Produktionsreife Automatisierung
- Erkennung von Grabsteinen, Inschriften oder anderen Details

## Verfügbare Datenquellen

- Aufbereitete georeferenzierte RGB-Orthofotos
- Höheninformationen (DSM/DTM)
- Vektorbasierte Ground Truth Annotationen der Grabumrandungen
- Bayern Atlas Luftbilder mit Parzellkarten (Referenz)

## Erfolgskriterien

- Segmentierungsgenauigkeit (IoU) von mindestens 85% auf Validierungsdaten
- Funktionierender Prototyp mit vollständiger Pipeline (Input → Segmentierung → Vektorisierung → Export)
- Quantifizierter Mehrwert der Höheninformation gegenüber RGB-only
- GIS-kompatible Outputs, die in QGIS/ArcGIS verifiziert werden können
- Dokumentierte Lessons Learned für Skalierung auf weitere Friedhöfe

## Technologie-Stack

- **Deep Learning Framework**: PyTorch oder TensorFlow
- **Geospatial Libraries**: GDAL, Rasterio, GeoPandas, Shapely
- **Computer Vision**: OpenCV, scikit-image
- **Modellarchitektur**: U-Net oder ähnliche Segmentierungsarchitektur

## Nächste Schritte

1. Setup der Entwicklungsumgebung und Technologie-Stack
2. Validierung der vorhandenen Datenbasis (Vollständigkeit, Qualität)
3. Implementierung der Daten-Pipeline
4. Start Modelltraining

## Risiken und Abhängigkeiten

- **Datenqualität**: Qualität und Konsistenz der Ground Truth Annotationen
- **Variabilität**: Unterschiedliche Friedhofslayouts, Grabtypen und Bildqualitäten
- **Computing-Ressourcen**: GPU-Verfügbarkeit für Modelltraining
- **Vektorisierungsqualität**: Genauigkeit der Polygon-Extraktion aus Segmentierungsmasken
- **Generalisierbarkeit**: Übertragbarkeit auf neue, ungesehene Friedhöfe
