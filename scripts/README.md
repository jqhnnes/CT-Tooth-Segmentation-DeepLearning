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

### `03_predict.sh`

Runs `nnUNetv2_predict` for all trained folds on a common input dataset, producing matching prediction folders (needed before ensembling).

```
Usage: bash scripts/03_predict.sh <DATASET_NAME> <CONFIG> <INPUT_DIR> <OUTPUT_ROOT>
  <DATASET_NAME>   Dataset folder (numeric ID auto-extracted)
  <CONFIG>         nnU-Net configuration (e.g., 3d_fullres)
  <INPUT_DIR>      Directory with images to predict (default: Dataset002_Karies/imagesTr)
  <OUTPUT_ROOT>    Root folder to store per-fold predictions
```

Example:
```bash
chmod +x scripts/03_predict.sh
bash scripts/03_predict.sh Dataset001_GroundTruth 3d_fullres \
     data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/prediction_data/imagesTr \
     data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/predictions
```
Predictions are written with `--save_probabilities` so ensembles can combine softmax probabilities.

### `04_ensemble.sh`

Runs `nnUNetv2_ensemble` on the validation predictions of each trained fold. The script automatically copies `dataset.json`
and `nnUNetPlans.json` into the per-fold validation folders if needed.

```
Usage: bash scripts/04_ensemble.sh <DATASET_NAME> <CONFIG>
  <DATASET_NAME>   Dataset folder (numeric ID auto-extracted)
  <CONFIG>         Configuration to ensemble (e.g., 3d_fullres)
Output:
  ensemble_predictions/<Dataset>_<Config>/
```

Example:
```bash
chmod +x scripts/04_ensemble.sh
bash scripts/04_ensemble.sh Dataset001_GroundTruth 3d_fullres \
     data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/predictions
```

## Python Modules

- `nnunet_env.py`: Python variant of the environment loader (used by other scripts).
- `utils/create_validation_subset.py`: Moves a subset of cases from `imagesTr/labelsTr` into a validation/prediction folder and mirrors them to `imagesTs`. Use for preparing e.g. 50 validation cases:
  ```bash
  python scripts/utils/create_validation_subset.py \
      --dataset Dataset001_GroundTruth \
      --num_cases 50 \
      --target prediction_data
  ```

All additional automation (HPO, analysis, utilities) lives under `hpo/scripts/`.
