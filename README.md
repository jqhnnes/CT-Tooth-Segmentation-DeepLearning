# 3D Tooth Segmentation using nnU-Net

Deep learning pipeline for 3D tooth segmentation on µCT scans using nnU-Net v2. Includes hyperparameter optimization (HPO) with Optuna, training, inference with TTA + postprocessing, and analysis utilities.

## Project Overview

- Task: 3-class segmentation (enamel, dentin, pulp) on volumetric CT.
- Framework: nnU-Net v2.
- Extras: Optuna-driven HPO, TTA + postprocessing, summary/ranking and plotting tools.

## Project Structure (current)

```
CT-Tooth-Segmentation-DeepLearning/
├── hpo/                      # HPO pipeline and outputs
│   ├── scripts/
│   │   ├── preprocessing/    # nnunet_hpo_preprocess.py
│   │   ├── training/         # nnunet_train_eval_pipeline.py
│   │   ├── postprocessing/   # nnunet_tta_postprocess.py, evaluate_tta_pp.py
│   │   └── analysis/         # summarize_trials.py, plot_trials_summary.py
│   ├── config/               # nnUNetPlans_template.json
│   ├── preprocessing_output/ # Trial-specific preprocessed data
│   ├── training_output/      # Archived training runs (nnUNet_results per trial)
│   ├── analysis/             # trials_summary.json, plots/
│   └── results/              # Evaluation logs (if produced)
├── scripts/                  # Runner scripts and env helpers
│   ├── 00_plan.sh, 01_preprocess.sh, 02_training.sh, 03_predict.sh, 04_ensemble.sh
│   ├── analysis/             # analyze_grayscale_statistics, analyze_trial_parameters, etc.
│   ├── nnunet_env.sh / nnunet_env.py
│   └── utils/create_validation_subset.py
├── data/                     # nnUNet_raw / nnUNet_preprocessed / nnUNet_results (not in repo)
├── logs/                     # Training/eval logs
├── notebooks/                # Exploration notebooks
├── ensemble_predictions/     # (if generated)
├── analysis_results/         # (auxiliary analyses)
└── README.md                 # This file
```

For detailed HPO usage, see `hpo/README.md`.

## Setup

```bash
cd /ssd/geiger/CT-Tooth-Segmentation-DeepLearning
conda env create -f environment.yml
conda activate /ssd/geiger/myenv   # or your env name
source scripts/nnunet_env.sh       # exports nnUNet_raw/_preprocessed/_results
```

Prepare data under `nnUNet_raw` following nnU-Net conventions (Dataset001_GroundTruth/imagesTr, labelsTr, imagesTs, dataset.json).

## Core Workflows

### Generate HPO trials (Optuna + preprocess)
```bash
python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 5
```
Creates trial plans under `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X` using the current high-end search space (fine spacings 0.075–0.08 mm, large patches, high features, batch=1).

### Train trials (archived under hpo/training_output)
```bash
python hpo/scripts/training/nnunet_train_eval_pipeline.py --folds 0 --skip_evaluation
```
Options: `--folds 0 1 2`, `--trials trial_43`, `--only_evaluate`, `--stop_on_error`, `--eval_timeout`, `--device cuda:0|cpu`.

Manual single trial (example trial_43, fold 0):
```bash
source scripts/nnunet_env.sh
export nnUNet_preprocessed=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/preprocessing_output/Dataset001_GroundTruth/trial_43
export nnUNet_results=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/training_output/trial_43/nnUNet_results
nnUNetv2_train Dataset001_GroundTruth 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans
```
Re-run the same command in the same results path to resume training (no `--continue_training` flag).

### Inference + TTA + Postprocessing + optional Eval
```bash
/ssd/geiger/myenv/bin/python hpo/scripts/postprocessing/nnunet_tta_postprocess.py \
  --trials trial_43 \
  --folds 0 \
  --input_dir data/nnUNet_raw/Dataset001_GroundTruth/imagesTs \
  --eval_labels data/nnUNet_raw/Dataset001_GroundTruth/labelsTs
```
Produces `labelsTs_tta` and `labelsTs_tta_pp`, optionally evaluates.

### Evaluate existing TTA+PP predictions and rank
```bash
python hpo/scripts/postprocessing/evaluate_tta_pp.py --folds 0
```
Writes summaries to `hpo/analysis/trial_*_labelsTs_tta_pp_summary.json` and prints a ranking with ΔDice vs. baseline and TTA-only (if available).

### Summaries and plots
```bash
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/summarize_trials.py
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/plot_trials_summary.py
```
Creates `hpo/analysis/trials_summary.json` and `hpo/analysis/plots/trials_dice_vs_spacing.png`.

## Current Results (latest state)

- Best single-fold so far: `trial_43` (fold 0) Dice ≈ 0.7826 with TTA+PP.
- Search focuses on finer spacing, large patches, high feature bases to improve detail; additional folds and ensembling are pending for further gains.

## Configuration

Key environment variables (set in `scripts/nnunet_env.sh`):
- `nnUNet_raw`: path to raw dataset
- `nnUNet_preprocessed`: path to preprocessed data (can be trial-specific)
- `nnUNet_results`: path to training results (can be trial-specific)

## Metrics

Primary: Dice. Additional: Hausdorff Distance (HD95), IoU, volume similarity, precision/recall (as reported by `nnUNetv2_evaluate_folder`).

## Key Scripts (quick reference)

- `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py` — generate HPO trials + preprocess
- `hpo/scripts/training/nnunet_train_eval_pipeline.py` — orchestrate training/eval across trials
- `hpo/scripts/postprocessing/nnunet_tta_postprocess.py` — TTA inference + postprocessing (+optional eval)
- `hpo/scripts/postprocessing/evaluate_tta_pp.py` — evaluate TTA+PP predictions and rank
- `hpo/scripts/analysis/summarize_trials.py` — aggregate trial params + scores
- `hpo/scripts/analysis/plot_trials_summary.py` — plot Dice vs spacing
- `scripts/00_plan.sh` … `04_ensemble.sh` — shell helpers for nnU-Net stages
- `scripts/analysis/analyze_training_and_ensemble.py` — training/ensemble analysis
- `scripts/utils/create_validation_subset.py` — build a smaller validation subset

## License

MIT License. See `LICENSE` if present; otherwise treat as MIT per project intent.
