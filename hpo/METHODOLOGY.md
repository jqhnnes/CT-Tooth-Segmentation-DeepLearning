# HPO Methodology

This document describes the methodological approach for hyperparameter optimization (HPO) and the
training/inference pipeline of an nnU-Net v2 model for 3D tooth segmentation. It serves as a
reference for academic documentation (e.g., a bachelor's thesis) and points to the relevant
scripts in the `hpo/` folder.

## Objective

- Improve detail accuracy and edge sharpness through finer voxel spacing and larger model capacity.
- Systematically explore the hyperparameter space despite tight VRAM budgets.
- Maintain a reproducible pipeline from data preparation through evaluation, including TTA and postprocessing.

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

### 3) Inference, TTA, and Postprocessing

- Script: `hpo/scripts/postprocessing/nnunet_tta_postprocess.py`
- Purpose: TTA prediction (`nnUNetv2_predict`, TTA active by default), optionally
  `nnUNetv2_find_best_configuration` (with correct folds), applies `postprocessing.pkl`,
  and optionally runs evaluation.
- Key paths per trial: `nnUNet_results/.../fold_<f>/labelsTs_tta` -> `labelsTs_tta_pp`.
- Fallbacks: detects existing `postprocessing.pkl` in the model root or `crossval_results_folds_0`
  folder to avoid redundant searches.

### 4) Evaluation and Ranking

- Script: `hpo/scripts/postprocessing/evaluate_tta_pp.py`
- Purpose: Evaluates existing `labelsTs_tta_pp` per trial/fold using `nnUNetv2_evaluate_folder`,
  writes `hpo/analysis/<trial>_labelsTs_tta_pp_summary.json`, and ranks by Dice. Shows delta
  Dice vs. baseline/TTA-only where available.
- Flags: `--trials`, `--folds`, `--labels_ts`, `--force`.

### 5) Analysis and Visualization

- `hpo/scripts/analysis/summarize_trials.py`: Aggregates spacing/patch/batch/features and Dice
  (baseline or TTA+PP) into `hpo/analysis/trials_summary.json`.
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

- Always set environment variables correctly per trial, especially for separate inference/PP runs:
  - `nnUNet_preprocessed=/.../hpo/preprocessing_output/<Dataset>/trial_X`
  - `nnUNet_results=/.../hpo/training_output/trial_X/nnUNet_results`
  - `nnUNet_raw` pointing to the dataset (e.g., `data/nnUNet_raw/Dataset001_GroundTruth`)
- TTA is active by default in `nnUNetv2_predict` (`--disable_tta` turns it off).
- `nnUNetv2_find_best_configuration` requires trained folds; for single-fold use `-f 0`
  (implemented in `nnunet_tta_postprocess.py`).
- Postprocessing searches for `postprocessing.pkl` first in `model_root`, then in
  `crossval_results_folds_0`, to avoid unnecessary re-runs.
- Training can be resumed by re-running the same command (no `--continue_training` flag needed).

## Common Error Sources and Fixes

- Missing `_0000` suffixes in `imagesTs` → create symlinks with the suffix.
- Wrong environment variables (pointing to global rather than trial-specific paths) → set correctly
  before inference/PP.
- Optuna categorical error → use a static combination space (see above).
- `nnUNetv2_find_best_configuration` expects 5 folds by default → set `-f 0` explicitly for single-fold.

## Current Status

- Best known single-fold run (trial_43) achieves ~0.78 Dice after TTA+PP (foreground_mean), based
  on available summaries. Further improvements are pursued via finer spacing and higher `features_base`.

## Notes for Academic Documentation

- Describe the methodology following the phases: Planning → Implementation → Evaluation → Reflection.
- Highlight the trade-off between detail depth (fine spacing, large patches, high `features_base`)
  and VRAM limits.
- Document how postprocessing (connected components / `postprocessing.pkl`) and TTA contribute to
  quality improvements.
- Reference the analysis scripts for reproducibility of rankings and figures.

## Further Documents

- [`hpo/README.md`](README.md) — Runbook: commands and step-by-step instructions
- [`hpo/PREPROCESSING_DETAILS.md`](PREPROCESSING_DETAILS.md) — Technical details of preprocessing
  steps (resampling, normalization, configuration parameters)
