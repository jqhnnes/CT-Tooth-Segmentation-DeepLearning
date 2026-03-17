# Preprocessing Steps

## Overview

This document describes the preprocessing steps performed for 3D tooth segmentation with nnU-Net v2.
Preprocessing is carried out in two phases:

1. **Standard nnU-Net Preprocessing** (automated by `nnUNetv2_preprocess`)
2. **Hyperparameter Optimization (HPO) Pipeline** (trial-specific customizations)

---

## 1. Standard nnU-Net Preprocessing

Preprocessing is performed by `nnUNetv2_preprocess` and includes the following steps:

### 1.1 Resampling

- **Goal:** Unify voxel spacing to a consistent target spacing.
- **Method:**
  - **Image data (CT):** 3rd-order B-spline interpolation (`order: 3`) for isotropic resampling.
  - **Labels (segmentation):** Nearest-neighbor interpolation (`order: 1`) to preserve label integrity.
  - **Z-axis:** Separate handling available (`order_z: 0` for both data and labels).
- **Current HPO settings:**
  - Spacing: 0.075 mm or 0.080 mm (trial-dependent).
  - Original spacing: ~0.04 mm (µCT data).

### 1.2 Normalization

- **Scheme:** `CTNormalization` (Z-score normalization based on CT intensity statistics).
- **Mask-based normalization:** `use_mask_for_norm: false` (disabled, as it is not beneficial for teeth).
- **Computation:**
  - Mean and standard deviation are computed per image.
  - Normalization: `(voxel_value - mean) / std`.

### 1.3 Cropping

- **Goal:** Remove background voxels to reduce memory requirements.
- **Method:** Automatic cropping based on label masks.
- **Result:** Reduced image size while retaining all relevant structures.

### 1.4 Patch-Based Processing

- **Patch size:** Optimized via HPO.
  - Current: 192×256×128 (spacing 0.075 mm) or 224×256×128 (spacing 0.080 mm).
- **Purpose:** Enables training on large volumes despite limited GPU memory.

### 1.5 Data Format Conversion

- **Input:** NIfTI format (`.nii.gz`) from `nnUNet_raw/DatasetXXX/imagesTr` and `labelsTr`.
- **Output:** Preprocessed data in `nnUNet_preprocessed/DatasetXXX/`.
- **Format:** NumPy arrays (`.npy`) for efficient storage and fast loading.

---

## 2. HPO Pipeline Preprocessing

The HPO pipeline (`hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`) extends standard
preprocessing with trial-specific customizations:

### 2.1 Optuna-Based Parameter Search

- **Framework:** Optuna for systematic hyperparameter optimization.
- **Search space:**
  - **Spacing:** 0.075 mm, 0.080 mm
  - **Patch size:** (192, 256, 128), (224, 256, 128)
  - **Batch size:** 1 (VRAM-optimized)
  - **Features base:** 64, 72 (model capacity)
  - **Batch Dice:** True/False (varied per trial)

### 2.2 Dynamic Plan Generation

- **Template:** `hpo/config/nnUNetPlans_template.json`
- **Per-trial customizations:**
  - `spacing`: target voxel spacing
  - `patch_size`: patch dimensions
  - `batch_size`: batch size
  - `features_per_stage`: dynamically scaled based on `features_base`
  - `n_conv_per_stage`: number of convolutions per stage
  - `batch_dice`: batch Dice loss flag
  - `use_mask_for_norm`: normalization mask (fixed to `false`)

### 2.3 Trial-Specific Output

- **Path structure:** `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X/`
- **Contents per trial:**
  - `nnUNetPlans.json`: trial-specific configuration
  - `dataset_fingerprint.json`: dataset fingerprint
  - Preprocessed data (`.npy` files)
  - `params.json`: saved hyperparameters for traceability
  - `preprocess.log`: preprocessing log
  - `error.log`: error log (if present)

### 2.4 Model Architecture Scaling

- **Features per stage:** dynamically computed based on `features_base`.
  - Example for `features_base=64`: [64, 128, 256, 512, 640, 640]
  - Example for `features_base=72`: [72, 144, 288, 576, 720, 720]
- **Purpose:** Maximize model capacity within VRAM constraints.

---

## 3. Differences from the Standard Pipeline

| Aspect | Standard nnU-Net | HPO Pipeline |
|--------|------------------|--------------|
| **Spacing** | Automatically chosen (e.g., 0.2 mm) | Manually optimized (0.075–0.080 mm) |
| **Patch size** | Conservative (e.g., 128³) | Aggressive (192×256×128, 224×256×128) |
| **Features base** | Default (32) | High (64–72) |
| **Batch size** | 2–4 | 1 (VRAM-optimized) |
| **Trial management** | Single run | Multiple trials with Optuna |

---

## 4. Integration into the Overall Pipeline

```
1. HPO Preprocessing (Optuna + nnUNetv2_preprocess)
   ↓
2. Training (nnUNetv2_train)
   ↓
3. Inference + TTA (nnUNetv2_predict)
   ↓
4. Postprocessing (nnUNetv2_find_best_configuration + nnUNetv2_apply_postprocessing)
   ↓
5. Evaluation (nnUNetv2_evaluate_folder)
```

**Important:** Each trial has its own preprocessed data version, which is used during training
and inference.

---

## 5. Technical Details

### 5.1 Resampling Parameters

```json
{
  "resampling_fn_data": "resample_data_or_seg_to_shape",
  "resampling_fn_data_kwargs": {
    "is_seg": false,
    "order": 3,
    "order_z": 0,
    "force_separate_z": null
  },
  "resampling_fn_seg": "resample_data_or_seg_to_shape",
  "resampling_fn_seg_kwargs": {
    "is_seg": true,
    "order": 1,
    "order_z": 0,
    "force_separate_z": null
  }
}
```

### 5.2 Normalization

```json
{
  "normalization_schemes": ["CTNormalization"],
  "use_mask_for_norm": [false]
}
```

### 5.3 Example Trial Configuration (trial_43)

- **Spacing:** 0.08 mm
- **Patch:** (224, 256, 128)
- **Batch:** 1
- **Features base:** 72
- **Batch Dice:** false

---

## 6. References

- **nnU-Net v2 Documentation:** [nnunet.readthedocs.io](https://nnunet.readthedocs.io/)
- **Optuna Documentation:** [optuna.org](https://optuna.org/)
- **Related scripts:**
  - `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`
  - `scripts/00_plan.sh` (standard planning)
  - `scripts/01_preprocess.sh` (standard preprocessing)
