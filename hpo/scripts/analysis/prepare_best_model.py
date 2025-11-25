#!/usr/bin/env python3
"""
Bereitet das beste Modell (trial_8) für weitere Experimente vor.

Kopiert Checkpoints, Plans und erstellt eine README mit Anweisungen.
"""
import json
import shutil
from pathlib import Path

BEST_TRIAL = "trial_8"
# OUTPUT_DIR relativ zum hpo-Verzeichnis
HPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HPO_DIR / "best_model"
DATASET_NAME = "Dataset001_GroundTruth"

def main():
    print("=" * 80)
    print("BEREITE BESTES MODELL VOR (trial_8)")
    print("=" * 80)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Kopiere Checkpoint
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
        print(f"✓ Checkpoint kopiert: {checkpoint_dst}")
    else:
        print(f"⚠ Checkpoint nicht gefunden: {checkpoint_src}")
    
    # 2. Kopiere Plans-Datei
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
        print(f"✓ Plans-Datei kopiert: {plans_dst}")
    else:
        print(f"⚠ Plans-Datei nicht gefunden: {plans_src}")
    
    # 3. Lade Parameter
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
        
        # Speichere Parameter
        params_file = OUTPUT_DIR / "parameters.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        print(f"✓ Parameter gespeichert: {params_file}")
    
    # 4. Erstelle README
    readme_content = f"""# Bestes Modell: {BEST_TRIAL}

## Performance
- **Dice Score**: 0.9725
- **Rang**: 1 von 10 Trials

## Parameter
- **Patch Size**: [160, 160, 64]
- **Batch Size**: 4
- **Features Base**: 24
- **Features per Stage**: [24, 48, 96, 192, 240, 240]
- **Convs per Stage**: 2
- **Batch Dice**: False
- **Use Mask for Norm**: False

## Verwendung

### 1. Vorhersagen erstellen

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

**Wichtig**: Stelle sicher, dass:
- `nnUNet_results` auf `hpo/training_output/{BEST_TRIAL}/nnUNet_results` zeigt
- `nnUNet_preprocessed` auf `hpo/preprocessing_output/{DATASET_NAME}/{BEST_TRIAL}` zeigt

### 2. Modell weiter trainieren

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

## Dateien

- `checkpoint_best.pth`: Bestes Modell (Dice: 0.9725)
- `nnUNetPlans.json`: Plans-Datei mit Parametern
- `parameters.json`: Parameter als JSON

## Original-Pfade

- **Checkpoint**: `hpo/training_output/{BEST_TRIAL}/nnUNet_results/.../checkpoint_best.pth`
- **Plans**: `hpo/preprocessing_output/{DATASET_NAME}/{BEST_TRIAL}/.../nnUNetPlans.json`
- **Training Output**: `hpo/training_output/{BEST_TRIAL}/`
"""
    
    readme_file = OUTPUT_DIR / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ README erstellt: {readme_file}")
    
    print()
    print("=" * 80)
    print("FERTIG!")
    print("=" * 80)
    print(f"\nBestes Modell vorbereitet in: {OUTPUT_DIR}")
    print(f"\nSiehe {readme_file} für weitere Anweisungen.")

if __name__ == "__main__":
    main()

