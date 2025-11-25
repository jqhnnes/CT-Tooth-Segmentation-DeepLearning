# Best Parameters - HPO Summary

## Top 3 Trials

### trial_8 (Rank 1, Dice: 0.9725)

**Parameters:**
- patch_size: [160, 160, 64]
- batch_size: 4
- features_base: 24
- features_per_stage: [24, 48, 96, 192, 240, 240]
- n_conv_per_stage: 2
- batch_dice: False
- use_mask_for_norm: False

**Checkpoint:** `hpo/training_output/trial_8/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`

### trial_1 (Rank 2, Dice: 0.9723)

**Parameters:**
- patch_size: [128, 64, 128]
- batch_size: 2
- features_base: 48
- features_per_stage: [48, 96, 192, 384, 480, 480]
- n_conv_per_stage: 2
- batch_dice: False
- use_mask_for_norm: False

**Checkpoint:** `hpo/training_output/trial_1/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`

### trial_3 (Rank 3, Dice: 0.9721)

**Parameters:**
- patch_size: [64, 128, 64]
- batch_size: 2
- features_base: 32
- features_per_stage: [32, 64, 128, 256, 320, 320]
- n_conv_per_stage: 2
- batch_dice: True
- use_mask_for_norm: False

**Checkpoint:** `hpo/training_output/trial_3/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`

## Recommended Parameters

### Critical (must be set this way):
- `use_mask_for_norm`: False
  - All Top 3 trials have use_mask_for_norm=False

### Important (should be set this way):
- `n_conv_per_stage`: 2
- `batch_dice`: False
  - 2/3 top trials have batch_dice=False (trial_3 has True, but is still Top 3)

### Flexible (can vary):
- `patch_size`: Can vary: [160,160,64], [128,64,128], [64,128,64]
- `batch_size`: 2 or 4 both work
- `features_base`: 24, 32, or 48 - all work well

## Statistics

- Total Trials: 10
- Successful Trials (Dice > 0.96): 5
- Failed Trials (Dice < 0.52): 5
- Best Dice Score: 0.9725
- Worst Dice Score: 0.4830
- Average: 0.7344
- Standard Deviation: 0.2375
