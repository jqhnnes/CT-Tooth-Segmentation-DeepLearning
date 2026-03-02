# Analysis-Skripte

Alle Befehle von **Projektroot** aus ausführen (`/ssd/geiger/CT-Tooth-Segmentation-DeepLearning` oder wo das Repo liegt).  
Ausgaben landen in `analysis_results/` (siehe `analysis_results/README.md`).

---

## Grauwert / Intensität

### Grauwert-Statistiken + Histogramme (empfohlen)

```bash
python scripts/analysis/analyze_grayscale_statistics.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/grayscale/tooth_grayscale_distribution.txt`, `.png`
- Optionen: `--bins 70|100`, `--no_plot`, `--no_suppress_zero`, `--max_files N`, `--no_save`

### Grauwert-Histogramm (alternativ)

```bash
python scripts/analysis/create_grauwert_histogram.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/grayscale/grauwert_histogramm_<Dataset>_<timestamp>.png`

---

## Datensätze

### Dataset-Metadaten

```bash
python scripts/analysis/analyze_dataset_metadata.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/datasets/dataset_metadata_<Dataset>_<timestamp>.json`, `.txt`
- Optionen: `--output <pfad>`, `--nnunet_raw data/nnUNet_raw`, `--no_save`

### Label-Histogramm

```bash
python scripts/analysis/create_label_histogram.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/datasets/label_histogram_<Dataset>_<timestamp>.png`
- Optionen: `--log_scale`, `--output <pfad>`

---

## HPO / Trials

### Trial-Parameter (Spacing, Patch, Features, Batch)

```bash
python scripts/analysis/analyze_trial_parameters.py
```

- **Ausgabe:** `analysis_results/trials/trial_parameters_<timestamp>.json`, `trial_parameters_evolution_<timestamp>.png`
- Optionen:
  - `--config_dir hpo/config` (Standard)
  - `--output_dir analysis_results/trials`
  - `--individual_plots` → zusätzlich `trial_parameters_plots/`
  - `--no_save`

---

## Training / Ensemble

### Training- und Ensemble-Analyse

```bash
python scripts/analysis/analyze_training_and_ensemble.py --dataset Dataset001_GroundTruth
```

- **Ausgabe:** `analysis_results/training/training_analysis_<Dataset>_<timestamp>.csv`, `.json`
- Optionen: `--config 3d_fullres`, `--trainer`, `--plans`, `--output-dir <pfad>`

---

## Übersicht

| Skript | Kurzbeschreibung |
|--------|------------------|
| `analyze_grayscale_statistics.py` | Grauwert-Statistiken + Balken-Histogramme (anatomisch/pathologisch) |
| `create_grauwert_histogram.py` | Grauwert-Histogramm (Linien/Füllung) |
| `create_label_histogram.py` | Label-Verteilung pro Dataset |
| `analyze_dataset_metadata.py` | Spacing, Dimensionen, Label-Statistiken |
| `analyze_trial_parameters.py` | HPO-Parameter-Verlauf über Trials |
| `analyze_training_and_ensemble.py` | Folds, Konvergenz, Ensemble-Metriken |

**Voraussetzungen:** Aus Projektroot starten; für Grauwert/Dataset-Skripte: `nibabel`; für Plots: `matplotlib`.  
Umgebungsvariablen für nnU-Net (z. B. `nnUNet_raw`) ggf. vorher mit `source scripts/nnunet_env.sh` setzen.
