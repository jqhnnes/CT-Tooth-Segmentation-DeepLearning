# Analysis-Skripte

Alle Befehle von **Projektroot** aus ausführen.  
Ausgaben landen in `analysis_results/` (siehe `analysis_results/README.md`).  
Umgebungsvariablen vorher laden: `source scripts/nnunet_env.sh`

```
scripts/analysis/
├── dataset/       # Datensatz-Analyse (Metadaten, Grauwerte, Label-Histogramme)
├── training/      # Training- und HPO-Analyse (Folds, Trials, Ensemble)
└── evaluation/    # Modell-Evaluation (Dice, IoU per Label und Fold)
```

---

## `dataset/` – Datensatz-Analyse

### Grauwert-Statistiken + Histogramme

```bash
python scripts/analysis/dataset/analyze_grayscale_statistics.py --dataset Dataset002_Karies
```
- **Ausgabe:** `analysis_results/grayscale/tooth_grayscale_distribution.txt`, `.png`
- Optionen: `--bins 70|100`, `--no_plot`, `--no_suppress_zero`, `--max_files N`, `--no_save`

### Grauwert-Histogramm (alternativ)

```bash
python scripts/analysis/dataset/create_grauwert_histogram.py --dataset Dataset002_Karies
```
- **Ausgabe:** `analysis_results/grayscale/grauwert_histogramm_<Dataset>_<timestamp>.png`

### Dataset-Metadaten

```bash
python scripts/analysis/dataset/analyze_dataset_metadata.py --dataset Dataset002_Karies
```
- **Ausgabe:** `analysis_results/datasets/dataset_metadata_<Dataset>_<timestamp>.json`, `.txt`
- Optionen: `--output <pfad>`, `--nnunet_raw data/nnUNet_raw`, `--no_save`

### Label-Histogramm

```bash
python scripts/analysis/dataset/create_label_histogram.py --dataset Dataset002_Karies
```
- **Ausgabe:** `analysis_results/datasets/label_histogram_<Dataset>_<timestamp>.png`
- Optionen: `--log_scale`, `--output <pfad>`

---

## `training/` – Training- und HPO-Analyse

### Trial-Parameter (Spacing, Patch, Features, Batch)

```bash
python scripts/analysis/training/analyze_trial_parameters.py
```
- **Ausgabe:** `analysis_results/trials/trial_parameters.json`, `trial_parameters_evolution.png`
- Optionen: `--config_dir hpo/config`, `--output_dir analysis_results/trials`, `--individual_plots`, `--no_save`

### Training- und Ensemble-Analyse

```bash
python scripts/analysis/training/analyze_training_and_ensemble.py --dataset Dataset001_GroundTruth
```
- **Ausgabe:** `analysis_results/training/training_analysis_<Dataset>_<timestamp>.csv`, `.json`
- Optionen: `--config 3d_fullres`, `--trainer`, `--plans`, `--output-dir <pfad>`

---

## `evaluation/` – Modell-Evaluation

### Fold-Validation auswerten

```bash
# Alle Folds (Validation-Metriken aus summary.json)
python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset001_GroundTruth

# Für Karies
python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/evaluation/<Dataset>/fold_comparison_<Dataset>.png`, `per_case_dice_fold<N>_<Dataset>.png`
- Optionen: `--folds 0 1 2`, `--config 3d_fullres`, `--no_save`

### Ensemble-Evaluation (Test-Set)

```bash
# Ensemble-Predictions vs. labelsTs auswerten
python scripts/analysis/evaluation/evaluate_ensemble.py \
  --dataset Dataset001_GroundTruth \
  --predictions ensemble_predictions/Dataset001_GroundTruth_3d_fullres \
  --labels data/nnUNet_raw/Dataset001_GroundTruth/labelsTs

# Für Karies
python scripts/analysis/evaluation/evaluate_ensemble.py --dataset Dataset002_Karies
```

- **Ausgabe:** `analysis_results/evaluation/<Dataset>/ensemble/`

---

## Übersicht

| Skript | Unterordner | Kurzbeschreibung |
|--------|-------------|------------------|
| `analyze_grayscale_statistics.py` | `dataset/` | Grauwert-Statistiken + Histogramme |
| `create_grauwert_histogram.py` | `dataset/` | Grauwert-Histogramm (Linien/Füllung) |
| `create_label_histogram.py` | `dataset/` | Label-Verteilung pro Dataset |
| `analyze_dataset_metadata.py` | `dataset/` | Spacing, Dimensionen, Label-Statistiken |
| `analyze_trial_parameters.py` | `training/` | HPO-Parameter-Verlauf über Trials |
| `analyze_training_and_ensemble.py` | `training/` | Folds, Konvergenz, Ensemble-Metriken |
| `evaluate_folds.py` | `evaluation/` | Dice/IoU pro Label und Fold (Validation) |
| `evaluate_ensemble.py` | `evaluation/` | Ensemble-Evaluation via `nnUNetv2_evaluate_folder` |

**Voraussetzungen:** `nibabel` für Grauwert/Dataset-Skripte; `matplotlib` für Plots.
