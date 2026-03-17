# HPO Methodology

This document describes the methodological approach for hyperparameter optimization (HPO) and the
training/inference pipeline of an nnU-Net v2 model for 3D tooth segmentation. It serves as a
reference for academic documentation (e.g., a bachelor's thesis) and points to the relevant
scripts in the `hpo/` folder.

## Objective

- Improve detail accuracy and edge sharpness through finer voxel spacing and larger model capacity.
- Systematically explore the hyperparameter space despite tight VRAM budgets.
- Maintain a reproducible pipeline from data preparation through training and evaluation.

## Pipeline Phases

### 1) Planning and Data Preparation (HPO / Preprocessing)

- Script: `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`
- Purpose: Optuna generates plan variants, dynamically adjusts spacing/patch/batch/features, and
  runs `nnUNetv2_preprocess` per trial. Results are stored under `hpo/preprocessing_output/<Dataset>/trial_X/`.
- Environment variables (set via `scripts/nnunet_env.sh`): `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.
- Run: `python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials N`
- Per-trial logging: `params.json` (selected hyperparameters), `error.log` (failure cases).

### 2) Training (optionally including Evaluation)

- Script: `hpo/scripts/training/nnunet_train_eval_pipeline.py`
- Purpose: Stages trial-specific preprocessed data, starts `nnUNetv2_train`, optionally runs
  `nnUNetv2_evaluate_folder`, and logs GPU usage.
- Folds: freely selectable (default: fold 0). Resume by re-running the same command (nnU-Net
  automatically continues from the last checkpoint).
- Archiving: After each trial, results are moved to `hpo/training_output/trial_X/nnUNet_results/...`.
- Key flags: `--trials`, `--folds`, `--skip_evaluation`, `--only_evaluate`, `--stop_on_error`, `--eval_timeout`.

### 3) Analysis and Visualization

- `hpo/scripts/analysis/summarize_trials.py`: Aggregates spacing/patch/batch/features and
  validation Dice (from `fold_0/validation/summary.json`) into `hpo/analysis/trials_summary.json`.
- `hpo/scripts/analysis/plot_trials_summary.py`: Visualizes Dice vs. spacing, colored by
  `features_base`. Output: `hpo/analysis/plots/trials_dice_vs_spacing.png`.

## Search Space Evolution (Optuna)

| Phase | Goal | Spacing Candidates | Patch Sizes | Batch | features_base | Motivation |
|-------|------|--------------------|-------------|-------|---------------|------------|
| Early | Stability, first fine-grain tests | 0.25–0.20 mm | moderate | 2–4 | 24–32 | Collect crash-free baselines, reserve VRAM |
| Mid | Focus on <0.20 mm | 0.15, 0.12, 0.10 mm | smaller/adapted | 1–2 | 32–48 | Finer resolution with reduced batch/features, minimize OOM |
| Late | Aggressively fine-grained | 0.10, 0.095, 0.08 mm | larger, context-rich | 1 | 48–64 | More context at fine spacing, near VRAM limit |
| Current (high-end) | Maximum detail depth | 0.075 mm → patch (192×256×128), fb=64; 0.080 mm → patch (224×256×128), fb=72 | 1 | 64/72 | Finest spacings with large patch area; deliberately near VRAM limit for higher detail |

Additional adaptations:
- Fixed combination space (`combo_choices`) to avoid Optuna's "CategoricalDistribution does not support
  dynamic value space" error.
- Dynamic `features_per_stage` scaling depending on `features_base` to maximize capacity while
  managing VRAM.
- `batch_dice` is varied per trial (True/False).
- Trial parameters (`params.json`) and error messages (`error.log`) are saved for traceability.

## Key Operational Details

- Always set environment variables correctly per trial:
  - `nnUNet_preprocessed=/.../hpo/preprocessing_output/<Dataset>/trial_X`
  - `nnUNet_results=/.../hpo/training_output/trial_X/nnUNet_results`
  - `nnUNet_raw` pointing to the dataset (e.g., `data/nnUNet_raw/Dataset001_GroundTruth`)
- Training can be resumed by re-running the same command (no `--continue_training` flag needed).

## Common Error Sources and Fixes

- Missing `_0000` suffixes in `imagesTs` → create symlinks with the suffix.
- Wrong environment variables (pointing to global rather than trial-specific paths) → set correctly
  before training.
- Optuna categorical error → use a static combination space (see above).

## Current Status

- Best known single-fold run: trial_43, validation Dice ~0.78 (foreground_mean).
  Further improvements are pursued via finer spacing and higher `features_base`.

## Notes for Academic Documentation

- Describe the methodology following the phases: Planning → Training → Evaluation → Reflection.
- Highlight the trade-off between detail depth (fine spacing, large patches, high `features_base`)
  and VRAM limits.
- Reference the analysis scripts for reproducibility of rankings and figures.

## Further Documents

- [`hpo/README.md`](README.md) — Runbook: commands and step-by-step instructions
- [`hpo/PREPROCESSING_DETAILS.md`](PREPROCESSING_DETAILS.md) — Technical details of preprocessing
  steps (resampling, normalization, configuration parameters)
