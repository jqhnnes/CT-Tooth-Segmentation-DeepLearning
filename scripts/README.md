# Scripts Overview

This folder contains helper scripts that orchestrate environment setup, preprocessing and other utilities around nnU-Net.

## Shell Scripts

### `00_nnunet_env.sh`

Sets the nnU-Net environment variables and ensures the conda binary directory is on your `PATH`.

```bash
source scripts/00_nnunet_env.sh
```

Run this before any nnU-Net CLI commands.

### `01_run_preprocess.sh`

Convenience wrapper around `nnUNetv2_plan_and_preprocess`.

```bash
chmod +x scripts/01_run_preprocess.sh        # once

# Default: Dataset001_GroundTruth, 3d_fullres, 8 processes
bash scripts/01_run_preprocess.sh

# Custom dataset/config/process count
bash scripts/01_run_preprocess.sh Dataset001_GroundTruth 3d_fullres 12
```

Arguments:
1. Dataset name (default `Dataset001_GroundTruth`, numeric ID auto-extracted).
2. Configuration (`3d_fullres`, `2d`, etc.).
3. Number of processes passed to `-np`.

## Python Modules

- `nnunet_env.py`: Python variant of the environment loader (used when importing from scripts).

All other automation scripts (training, analysis, etc.) live under `hpo/scripts/`.

