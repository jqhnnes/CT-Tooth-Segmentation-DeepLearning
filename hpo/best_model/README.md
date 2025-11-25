# Best Model: trial_8

## Performance
- **Dice Score**: 0.9725
- **Rank**: 1 out of 10 trials

## Parameters
- **Patch Size**: [160, 160, 64]
- **Batch Size**: 4
- **Features Base**: 24
- **Features per Stage**: [24, 48, 96, 192, 240, 240]
- **Convs per Stage**: 2
- **Batch Dice**: False
- **Use Mask for Norm**: False

## Usage

### 1. Create Predictions

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

**Important**: Make sure that:
- `nnUNet_results` points to `hpo/training_output/trial_8/nnUNet_results`
- `nnUNet_preprocessed` points to `hpo/preprocessing_output/Dataset001_GroundTruth/trial_8`

### 2. Continue Training

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

## Files

- `checkpoint_best.pth`: Best model (Dice: 0.9725)
- `nnUNetPlans.json`: Plans file with parameters
- `parameters.json`: Parameters as JSON

## Original Paths

- **Checkpoint**: `hpo/training_output/trial_8/nnUNet_results/.../checkpoint_best.pth`
- **Plans**: `hpo/preprocessing_output/Dataset001_GroundTruth/trial_8/.../nnUNetPlans.json`
- **Training Output**: `hpo/training_output/trial_8/`
