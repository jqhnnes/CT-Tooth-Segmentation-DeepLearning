# Analysis Scripts

Run all commands from the **project root**.
Outputs are written to `analysis_results/`.
Load environment variables first: `source scripts/nnunet_env.sh`

## Structure

```
scripts/analysis/
├── dataset/       # Dataset analysis (metadata, grayscale, label histograms)
├── training/      # Training and HPO analysis (folds, trials, ensemble)
└── evaluation/    # Model evaluation (Dice, IoU per label and fold)
```

---

## dataset/ — Dataset Analysis

### Grayscale Statistics and Histograms

```bash
python scripts/analysis/dataset/analyze_grayscale_statistics.py --dataset Dataset002_Karies
```

Output: `analysis_results/grayscale/tooth_grayscale_distribution.txt`, `.png`  
Options: `--bins 70|100`, `--no_plot`, `--no_suppress_zero`, `--max_files N`, `--no_save`

### Grayscale Histogram (alternative)

```bash
python scripts/analysis/dataset/create_grayscale_histogram.py --dataset Dataset002_Karies
```

Output: `analysis_results/grayscale/grayscale_histogram_<Dataset>_<timestamp>.png`

### Dataset Metadata

```bash
python scripts/analysis/dataset/analyze_dataset_metadata.py --dataset Dataset002_Karies
```

Output: `analysis_results/datasets/dataset_metadata_<Dataset>_<timestamp>.json`, `.txt`  
Options: `--output <path>`, `--nnunet_raw data/nnUNet_raw`, `--no_save`

### Label Histogram

```bash
python scripts/analysis/dataset/create_label_histogram.py --dataset Dataset002_Karies
```

Output: `analysis_results/datasets/label_histogram_<Dataset>_<timestamp>.png`  
Options: `--log_scale`, `--output <path>`

---

## training/ — Training and HPO Analysis

### Trial Parameters (Spacing, Patch, Features, Batch)

```bash
python scripts/analysis/training/analyze_trial_parameters.py
```

Output: `analysis_results/trials/trial_parameters.json`, `trial_parameters_evolution.png`  
Options: `--config_dir hpo/config`, `--output_dir analysis_results/trials`, `--individual_plots`, `--no_save`

### Training and Ensemble Analysis

```bash
python scripts/analysis/training/analyze_training_and_ensemble.py --dataset Dataset001_GroundTruth
```

Output: `analysis_results/training/training_analysis_<Dataset>_<timestamp>.csv`, `.json`  
Options: `--config 3d_fullres`, `--trainer`, `--plans`, `--output-dir <path>`

---

## evaluation/ — Model Evaluation

### Per-Fold Validation

```bash
python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset001_GroundTruth
python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset002_Karies
```

Output: `analysis_results/evaluation/<Dataset>/fold_comparison_<Dataset>.png`, `per_case_dice_fold<N>_<Dataset>.png`  
Options: `--folds 0 1 2`, `--config 3d_fullres`, `--no_save`

### Ensemble Evaluation (Test Set)

```bash
python scripts/analysis/evaluation/evaluate_ensemble.py \
  --dataset Dataset001_GroundTruth \
  --pred_dir ensemble_predictions/Dataset001_GroundTruth_3d_fullres \
  --labels_dir data/nnUNet_raw/Dataset001_GroundTruth/labelsTs
```

Output: `analysis_results/evaluation/<Dataset>/ensemble/`

---

## Overview

| Script | Subfolder | Description |
|--------|-----------|-------------|
| `analyze_grayscale_statistics.py` | `dataset/` | CT intensity statistics and histograms per tissue |
| `create_grayscale_histogram.py` | `dataset/` | Grayscale histogram (line/fill plot) |
| `create_label_histogram.py` | `dataset/` | Label voxel distribution per dataset |
| `analyze_dataset_metadata.py` | `dataset/` | Spacing, dimensions, label statistics |
| `analyze_trial_parameters.py` | `training/` | HPO parameter evolution across trials |
| `analyze_training_and_ensemble.py` | `training/` | Fold convergence and ensemble metrics |
| `evaluate_folds.py` | `evaluation/` | Dice/IoU per label and fold (validation) |
| `evaluate_ensemble.py` | `evaluation/` | Ensemble evaluation via nnUNetv2_evaluate_folder |

**Requirements:** `nibabel` for grayscale/dataset scripts; `matplotlib` for plots.
