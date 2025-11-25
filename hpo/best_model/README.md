# Bestes Modell: trial_8

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
nnUNetv2_predict \
    -i /path/to/imagesTs \
    -o /path/to/output \
    -d 1 \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    -f 0
```

**Wichtig**: Stelle sicher, dass:
- `nnUNet_results` auf `hpo/training_output/trial_8/nnUNet_results` zeigt
- `nnUNet_preprocessed` auf `hpo/preprocessing_output/Dataset001_GroundTruth/trial_8` zeigt

### 2. Modell weiter trainieren

```bash
nnUNetv2_train \
    1 \
    3d_fullres \
    0 \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    --continue_training
```

### 3. Evaluation

```bash
nnUNetv2_evaluate \
    -d 1 \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    -f 0
```

## Dateien

- `checkpoint_best.pth`: Bestes Modell (Dice: 0.9725)
- `nnUNetPlans.json`: Plans-Datei mit Parametern
- `parameters.json`: Parameter als JSON

## Original-Pfade

- **Checkpoint**: `hpo/training_output/trial_8/nnUNet_results/.../checkpoint_best.pth`
- **Plans**: `hpo/preprocessing_output/Dataset001_GroundTruth/trial_8/.../nnUNetPlans.json`
- **Training Output**: `hpo/training_output/trial_8/`
