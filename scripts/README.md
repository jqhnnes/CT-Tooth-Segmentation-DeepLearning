# Scripts Overview

This folder contains helper scripts that orchestrate environment setup, preprocessing and other utilities around nnU-Net.

## Shell Scripts

### `nnunet_env.sh`

Loads all required nnU-Net environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) and ensures the conda `bin`
directory is on your `PATH`.

```bash
source scripts/nnunet_env.sh
```

Run this once per shell session before executing any nnU-Net CLI.

### `00_plan.sh`

Fingerprint extraction + experiment planning only (no preprocessing). Allows you to edit the generated `nnUNetPlans.json`.

```
Usage: bash scripts/00_plan.sh <DATASET_NAME> <NUM_PROCESSES>
  <DATASET_NAME>   Dataset folder under nnUNet_raw (default: Dataset001_GroundTruth)
  <NUM_PROCESSES>  CPU processes for fingerprint/planning (default: 8; keep <= number of CPU threads)
Environment:
  PLANS_NAME       (optional) custom plans identifier (default: nnUNetPlans)
```

Example:
```bash
chmod +x scripts/00_plan.sh
bash scripts/00_plan.sh Dataset001_GroundTruth 1
```

### `01_preprocess.sh`

Preprocessing only; expects `nnUNetPlans.json` to exist (e.g., copied from `hpo/best_model` or edited after `00_plan.sh`).
Automatically copies `hpo/best_model/nnUNetPlans.json` into `${nnUNet_preprocessed}/<Dataset>/nnUNetPlans.json` if present.

```
Usage: bash scripts/01_preprocess.sh <DATASET_NAME> <CONFIG> <NUM_PROCESSES>
  <DATASET_NAME>   Dataset folder under nnUNet_raw (default: Dataset001_GroundTruth)
  <CONFIG>         nnU-Net configuration to preprocess (default: 3d_fullres)
  <NUM_PROCESSES>  CPU processes for preprocessing (per configuration)
Environment:
  PLANS_NAME       (optional) custom plans identifier used during preprocessing
```

Example:
```bash
chmod +x scripts/01_preprocess.sh
bash scripts/01_preprocess.sh Dataset001_GroundTruth 3d_fullres 1
```

### `02_training.sh`

Runs `nnUNetv2_train` and simultaneously logs GPU usage via `nvidia-smi`.

```
Usage: bash scripts/02_training.sh <DATASET_NAME> <CONFIG> <FOLD>
  <DATASET_NAME>   Dataset folder (numeric ID auto-extracted)
  <CONFIG>         nnU-Net configuration (e.g., 3d_fullres)
  <FOLD>           Fold index (0–4)
Outputs:
  logs/training_run/train_<timestamp>.log  -> nnUNet training stdout/stderr
  logs/training_run/gpu_<timestamp>.csv    -> GPU metrics sampled every 30s
```

Example:
```bash
chmod +x scripts/02_training.sh
bash scripts/02_training.sh Dataset001_GroundTruth 3d_fullres 0
```

## Python Modules

- `nnunet_env.py`: Python variant of the environment loader (used by other scripts).

All additional automation (HPO, analysis, utilities) lives under `hpo/scripts/`.
