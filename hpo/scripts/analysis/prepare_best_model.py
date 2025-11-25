#!/usr/bin/env python3
"""
Prepare the best model (trial_8) for further experiments.

This script copies checkpoints, plans, and creates a README with instructions
for using the best performing model from the HPO trials.

Usage:
    python hpo/scripts/analysis/prepare_best_model.py
"""
import json
import shutil
from pathlib import Path

BEST_TRIAL = "trial_8"
# OUTPUT_DIR relative to hpo directory
HPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HPO_DIR / "best_model"
DATASET_NAME = "Dataset001_GroundTruth"

def main():
    print("=" * 80)
    print("PREPARING BEST MODEL (trial_8)")
    print("=" * 80)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy checkpoint
    checkpoint_src = (
        HPO_DIR
        / "training_output"
        / BEST_TRIAL
        / "nnUNet_results"
        / DATASET_NAME
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_best.pth"
    )
    
    if checkpoint_src.exists():
        checkpoint_dst = OUTPUT_DIR / "checkpoint_best.pth"
        shutil.copy2(checkpoint_src, checkpoint_dst)
        print(f"✓ Checkpoint copied: {checkpoint_dst}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_src}")
    
    # 2. Copy plans file
    plans_src = (
        HPO_DIR
        / "preprocessing_output"
        / DATASET_NAME
        / BEST_TRIAL
        / DATASET_NAME
        / "nnUNetPlans.json"
    )
    
    if plans_src.exists():
        plans_dst = OUTPUT_DIR / "nnUNetPlans.json"
        shutil.copy2(plans_src, plans_dst)
        print(f"✓ Plans file copied: {plans_dst}")
    else:
        print(f"⚠ Plans file not found: {plans_src}")
    
    # 3. Load parameters
    if plans_src.exists():
        with open(plans_src, 'r') as f:
            plans = json.load(f)
        
        config = plans.get('configurations', {}).get('3d_fullres', {})
        arch = config.get('architecture', {}).get('arch_kwargs', {})
        
        parameters = {
            'patch_size': config.get('patch_size', []),
            'batch_size': config.get('batch_size'),
            'features_per_stage': arch.get('features_per_stage', []),
            'n_conv_per_stage': arch.get('n_conv_per_stage', []),
            'batch_dice': config.get('batch_dice'),
            'use_mask_for_norm': config.get('use_mask_for_norm', [False])[0] if isinstance(config.get('use_mask_for_norm'), list) else config.get('use_mask_for_norm')
        }
        
        # Save parameters
        params_file = OUTPUT_DIR / "parameters.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        print(f"✓ Parameters saved: {params_file}")
    
    # 4. Create README
    readme_content = f"""# Best Model: {BEST_TRIAL}

## Performance
- **Dice Score**: 0.9725
- **Rank**: 1 of 10 trials

## Parameters
- **Patch Size**: [160, 160, 64]
- **Batch Size**: 4
- **Features Base**: 24
- **Features per Stage**: [24, 48, 96, 192, 240, 240]
- **Convs per Stage**: 2
- **Batch Dice**: False
- **Use Mask for Norm**: False

## Usage

### 1. Generate Predictions

```bash
nnUNetv2_predict \\
    -i /path/to/imagesTs \\
    -o /path/to/output \\
    -d 1 \\
    -c 3d_fullres \\
    -tr nnUNetTrainer \\
    -p nnUNetPlans \\
    -f 0
```

**Important**: Make sure that:
- `nnUNet_results` points to `hpo/training_output/{BEST_TRIAL}/nnUNet_results`
- `nnUNet_preprocessed` points to `hpo/preprocessing_output/{DATASET_NAME}/{BEST_TRIAL}`

### 2. Continue Training

```bash
nnUNetv2_train \\
    1 \\
    3d_fullres \\
    0 \\
    -tr nnUNetTrainer \\
    -p nnUNetPlans \\
    --continue_training
```

### 3. Evaluation

```bash
nnUNetv2_evaluate \\
    -d 1 \\
    -c 3d_fullres \\
    -tr nnUNetTrainer \\
    -p nnUNetPlans \\
    -f 0
```

## Files

- `checkpoint_best.pth`: Best model (Dice: 0.9725)
- `nnUNetPlans.json`: Plans file with parameters
- `parameters.json`: Parameters as JSON

## Original Paths

- **Checkpoint**: `hpo/training_output/{BEST_TRIAL}/nnUNet_results/.../checkpoint_best.pth`
- **Plans**: `hpo/preprocessing_output/{DATASET_NAME}/{BEST_TRIAL}/.../nnUNetPlans.json`
- **Training Output**: `hpo/training_output/{BEST_TRIAL}/`
"""
    
    readme_file = OUTPUT_DIR / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ README created: {readme_file}")
    
    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nBest model prepared in: {OUTPUT_DIR}")
    print(f"\nSee {readme_file} for further instructions.")

if __name__ == "__main__":
    main()

