# Vorverarbeitungsschritte (Preprocessing Steps)

## Übersicht

Dieses Dokument beschreibt die Vorverarbeitungsschritte, die im Rahmen der 3D-Zahnsegmentierung mit nnU-Net v2 durchgeführt wurden. Die Vorverarbeitung erfolgt in zwei Phasen:

1. **Standard nnU-Net Vorverarbeitung** (automatisch durch `nnUNetv2_preprocess`)
2. **Hyperparameter-Optimierung (HPO) Pipeline** (trial-spezifische Anpassungen)

---

## 1. Standard nnU-Net Vorverarbeitung

Die Vorverarbeitung wird durch `nnUNetv2_preprocess` durchgeführt und umfasst folgende Schritte:

### 1.1 Resampling (Neuabtastung)
- **Ziel**: Vereinheitlichung der Voxelspacing auf ein einheitliches Ziel-Spacing
- **Methode**: 
  - **Daten (CT-Images)**: B-spline Interpolation 3. Ordnung (`order: 3`) für isotrope Resampling
  - **Labels (Segmentation)**: Nearest-Neighbor Interpolation (`order: 1`) zur Erhaltung der Label-Integrität
  - **Z-Achse**: Separates Handling möglich (`order_z: 0` für Labels, `order_z: 0` für Daten)
- **Aktuelle HPO-Einstellungen**: 
  - Spacing: 0.075 mm oder 0.080 mm (je nach Trial)
  - Original-Spacing: ~0.04 mm (µCT-Daten)

### 1.2 Normalisierung
- **Schema**: `CTNormalization` (Z-Score Normalisierung basierend auf CT-Intensitätsstatistiken)
- **Mask-basierte Normalisierung**: `use_mask_for_norm: false` (deaktiviert, da bei Zähnen nicht vorteilhaft)
- **Berechnung**: 
  - Mean und Standardabweichung werden pro Bild berechnet
  - Normalisierung: `(voxel_value - mean) / std`

### 1.3 Cropping (Zuschneiden)
- **Ziel**: Entfernung von Hintergrundvoxeln zur Reduktion der Speicheranforderungen
- **Methode**: Automatisches Cropping basierend auf Label-Masken
- **Ergebnis**: Reduzierte Bildgröße bei Erhaltung aller relevanten Strukturen

### 1.4 Patch-basierte Verarbeitung
- **Patch-Größe**: Wird durch HPO optimiert
  - Aktuell: 192×256×128 (Spacing 0.075 mm) oder 224×256×128 (Spacing 0.080 mm)
- **Zweck**: Ermöglicht Training auf großen Volumina trotz begrenztem GPU-Speicher

### 1.5 Datenformat-Konvertierung
- **Input**: NIfTI-Format (`.nii.gz`) aus `nnUNet_raw/DatasetXXX/imagesTr` und `labelsTr`
- **Output**: Vorverarbeitete Daten in `nnUNet_preprocessed/DatasetXXX/`
- **Format**: Numpy-Arrays (`.npy`) für effiziente Speicherung und schnelles Laden

---

## 2. Hyperparameter-Optimierung (HPO) Pipeline

Die HPO-Pipeline (`hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`) erweitert die Standard-Vorverarbeitung um trial-spezifische Anpassungen:

### 2.1 Optuna-basierte Parameter-Suche
- **Framework**: Optuna für systematische Hyperparameter-Optimierung
- **Suchraum**: 
  - **Spacing**: 0.075 mm, 0.080 mm
  - **Patch-Größe**: (192, 256, 128), (224, 256, 128)
  - **Batch-Größe**: 1 (VRAM-optimiert)
  - **Features Base**: 64, 72 (Modellkapazität)
  - **Batch-Dice**: True/False (per Trial variiert)

### 2.2 Dynamische Plan-Generierung
- **Template**: `hpo/config/nnUNetPlans_template.json`
- **Anpassungen pro Trial**:
  - `spacing`: Ziel-Voxelspacing
  - `patch_size`: Patch-Dimensionen
  - `batch_size`: Batch-Größe
  - `features_per_stage`: Dynamisch basierend auf `features_base` skaliert
  - `n_conv_per_stage`: Anzahl Convolutions pro Stage
  - `batch_dice`: Batch-Dice Loss Flag
  - `use_mask_for_norm`: Normalisierungs-Mask (fest auf `false`)

### 2.3 Trial-spezifische Ausgabe
- **Pfad-Struktur**: `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X/`
- **Inhalt pro Trial**:
  - `nnUNetPlans.json`: Trial-spezifische Konfiguration
  - `dataset_fingerprint.json`: Dataset-Fingerprint
  - Vorverarbeitete Daten (`.npy` Dateien)
  - `params.json`: Gespeicherte Hyperparameter für Nachverfolgung
  - `preprocess.log`: Log-Datei der Vorverarbeitung
  - `error.log`: Fehlerprotokoll (falls vorhanden)

### 2.4 Skalierung der Modellarchitektur
- **Features per Stage**: Dynamisch berechnet basierend auf `features_base`
  - Beispiel für `features_base=64`: [64, 128, 256, 512, 640, 640]
  - Beispiel für `features_base=72`: [72, 144, 288, 576, 720, 720]
- **Zweck**: Maximale Modellkapazität bei begrenztem VRAM

---

## 3. Unterschiede zur Standard-Pipeline

| Aspekt | Standard nnU-Net | HPO-Pipeline |
|--------|------------------|--------------|
| **Spacing** | Automatisch gewählt (z.B. 0.2 mm) | Manuell optimiert (0.075-0.080 mm) |
| **Patch-Größe** | Konservativ (z.B. 128³) | Aggressiv (192×256×128, 224×256×128) |
| **Features Base** | Standard (32) | Hoch (64-72) |
| **Batch-Größe** | 2-4 | 1 (VRAM-optimiert) |
| **Trial-Management** | Einzelner Durchlauf | Multiple Trials mit Optuna |

---

## 4. Integration in die Gesamt-Pipeline

Die Vorverarbeitung ist Teil der folgenden Pipeline:

```
1. HPO Preprocessing (Optuna + nnUNetv2_preprocess)
   ↓
2. Training (nnUNetv2_train)
   ↓
3. Inference + TTA (nnUNetv2_predict)
   ↓
4. Postprocessing (nnUNetv2_find_best_configuration + nnUNetv2_apply_postprocessing)
   ↓
5. Evaluation (nnUNetv2_evaluate_folder)
```

**Wichtig**: Jeder Trial hat seine eigene vorverarbeitete Datenversion, die während des Trainings und der Inferenz verwendet wird.

---

## 5. Technische Details

### 5.1 Resampling-Parameter
```json
{
  "resampling_fn_data": "resample_data_or_seg_to_shape",
  "resampling_fn_data_kwargs": {
    "is_seg": false,
    "order": 3,        // B-spline 3. Ordnung für CT-Daten
    "order_z": 0,
    "force_separate_z": null
  },
  "resampling_fn_seg": "resample_data_or_seg_to_shape",
  "resampling_fn_seg_kwargs": {
    "is_seg": true,
    "order": 1,        // Nearest-Neighbor für Labels
    "order_z": 0,
    "force_separate_z": null
  }
}
```

### 5.2 Normalisierung
```json
{
  "normalization_schemes": ["CTNormalization"],
  "use_mask_for_norm": [false]
}
```

### 5.3 Beispiel-Trial-Konfiguration (trial_43)
- **Spacing**: 0.08 mm
- **Patch**: (224, 256, 128)
- **Batch**: 1
- **Features Base**: 72
- **Batch-Dice**: false

---

## 6. Empfehlung für Dokumentation

**Ja, die HPO-Pipeline sollte erwähnt werden**, da:

1. **Methodische Relevanz**: Die HPO-Pipeline ist ein zentraler Bestandteil der Arbeit und unterscheidet sich deutlich von der Standard-nnU-Net-Vorverarbeitung.
2. **Reproduzierbarkeit**: Ohne Erwähnung der HPO-Pipeline ist die Methodik nicht vollständig nachvollziehbar.
3. **Wissenschaftlicher Beitrag**: Die systematische Optimierung von Spacing, Patch-Größe und Modellkapazität ist ein wichtiger methodischer Beitrag.

**Vorschlag für Dokumentationsstruktur**:

```
## Methodik

### 2.1 Datensatz
[... Beschreibung des µCT-Datensatzes ...]

### 2.2 Vorverarbeitung
#### 2.2.1 Standard nnU-Net Vorverarbeitung
- Resampling auf einheitliches Voxelspacing
- CT-Normalisierung
- Cropping
- Patch-basierte Verarbeitung

#### 2.2.2 Hyperparameter-Optimierung (HPO)
- Optuna-basierte Suche
- Optimierte Parameter: Spacing, Patch-Größe, Modellkapazität
- Trial-spezifische Vorverarbeitung

### 2.3 Training
[...]
```

---

## 7. Referenzen

- **nnU-Net v2 Dokumentation**: [nnunet.readthedocs.io](https://nnunet.readthedocs.io/)
- **Optuna Dokumentation**: [optuna.org](https://optuna.org/)
- **Projekt-Skripte**:
  - `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`
  - `scripts/00_plan.sh` (Standard Planning)
  - `scripts/01_preprocess.sh` (Standard Preprocessing)

---

*Erstellt: 2024-12-27*
*Letzte Aktualisierung: 2024-12-27*
