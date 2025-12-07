# Scripts Overview

This folder contains helper scripts that orchestrate environment setup, preprocessing and other utilities around nnU-Net.

## Shell Scripts

### `nnunet_env.sh`

Sets the nnU-Net environment variables and ensures the conda binary directory is on your `PATH`.

```bash
source scripts/nnunet_env.sh
```

Run this before any nnU-Net CLI commands.

### `00_plan.sh`

Runs fingerprint extraction + planning only (no preprocessing). After this you can edit `nnUNetPlans.json`.

```bash
chmod +x scripts/00_plan.sh
bash scripts/00_plan.sh Dataset001_GroundTruth 1   # final number = CPU processes for fingerprint/planning
# set PLANS_NAME=CustomPlans if you want a different identifier
```

### `01_preprocess.sh`

Preprocessing only, assumes plans already exist (e.g., edited after running the planner script).

```bash
chmod +x scripts/01_preprocess.sh
bash scripts/01_preprocess.sh Dataset001_GroundTruth 3d_fullres 1   # final number = CPU processes for preprocessing
# use PLANS_NAME=CustomPlans if you created a differently named plans file
```

## Python Modules

- `nnunet_env.py`: Python variant of the environment loader (used when importing from scripts).

All other automation scripts (training, analysis, etc.) live under `hpo/scripts/`.
