# HPO Results Analysis

## Overview

This document summarizes the key findings from the Hyperparameter Optimization (HPO) experiment with 10 trials on the CT Tooth Segmentation dataset.

**Dataset:** Dataset001_GroundTruth  
**Total Trials:** 10  
**Best Dice Score:** 0.9725 (trial_8)  
**Worst Dice Score:** 0.4830 (trial_5)  
**Average Dice Score:** 0.7344  
**Standard Deviation:** 0.2375

---

## Performance Distribution

### Top Performers (Dice > 0.96)
1. **trial_8**: 0.9725 (Best) - patch=[160,160,64], batch=4, features=24, n_conv=2, batch_dice=False, use_mask=False
2. **trial_1**: 0.9723 - patch=[128,64,128], batch=2, features=48, n_conv=2, batch_dice=False, use_mask=False
3. **trial_3**: 0.9721 - patch=[64,128,64], batch=2, features=32, n_conv=2, batch_dice=True, use_mask=False
4. **trial_4**: 0.9719 - patch=[128,128,160], batch=2, features=48, n_conv=2, batch_dice=False, use_mask=False
5. **trial_7**: 0.9698 - patch=[128,128,128], batch=2, features=48, n_conv=2, batch_dice=False, use_mask=False

### Poor Performers (Dice < 0.52)
- **trial_5**: 0.4830 - patch=[128,160,160], batch=2, features=48, n_conv=2, batch_dice=True, use_mask=True
- **trial_0**: 0.4897 - patch=[160,160,160], batch=2, features=32, n_conv=2, batch_dice=False, use_mask=True
- **trial_2**: 0.4945 - patch=[128,64,128], batch=4, features=24, n_conv=2, batch_dice=False, use_mask=True
- **trial_6**: 0.5066 - patch=[128,128,128], batch=4, features=24, n_conv=3, batch_dice=False, use_mask=True
- **trial_9**: 0.5115 - patch=[128,128,160], batch=2, features=32, n_conv=3, batch_dice=True, use_mask=True

**Observation:** Clear bimodal distribution - 5 trials achieved excellent performance (>0.96), while 5 trials failed completely (<0.52). This suggests that certain parameter combinations are critical for success.

---

## Key Findings by Parameter

### 1. **use_mask_for_norm: FALSE** (Critical Parameter)

**Finding:** ALL top 5 trials have `use_mask_for_norm = False`

| Trial | Dice | use_mask_for_norm |
|-------|------|-------------------|
| trial_8 | 0.9725 | False |
| trial_1 | 0.9723 | False |
| trial_3 | 0.9721 | False |
| trial_4 | 0.9719 | False |
| trial_7 | 0.9698 | False |

**Conclusion:** For CT tooth segmentation, mask-based normalization is **detrimental**. The model performs significantly better when normalizing without using the mask.

**Recommendation:** Always set `use_mask_for_norm = False` for this task.

---

### 2. **n_conv_per_stage: 2** (Important Parameter)

**Finding:** ALL top 5 trials use `n_conv_per_stage = 2`

| Trial | Dice | n_conv_per_stage |
|-------|------|------------------|
| trial_8 | 0.9725 | 2 |
| trial_1 | 0.9723 | 2 |
| trial_3 | 0.9721 | 2 |
| trial_4 | 0.9719 | 2 |
| trial_7 | 0.9698 | 2 |

**Conclusion:** Using 2 convolutions per stage provides optimal depth. Deeper networks (3 convs) may lead to overfitting or training instability.

**Recommendation:** Use `n_conv_per_stage = 2` for this task.

---

### 3. **batch_dice: FALSE** (Important Parameter)

**Finding:** 4 out of 5 top trials use `batch_dice = False`

| Trial | Dice | batch_dice |
|-------|------|------------|
| trial_8 | 0.9725 | False |
| trial_1 | 0.9723 | False |
| trial_3 | 0.9721 | True |
| trial_4 | 0.9719 | False |
| trial_7 | 0.9698 | False |

**Note:** trial_3 achieved excellent performance with `batch_dice = True`, but the majority of top performers use `False`.

**Conclusion:** Standard Dice loss (not batch Dice) is generally more effective, though batch Dice can also work.

**Recommendation:** Prefer `batch_dice = False`, but `True` is acceptable.

---

### 4. **Features Base: Flexible (24, 32, or 48)**

**Finding:** Top trials use different feature base values

| Trial | Dice | features_base |
|-------|------|--------------|
| trial_8 | 0.9725 | 24 |
| trial_1 | 0.9723 | 48 |
| trial_3 | 0.9721 | 32 |
| trial_4 | 0.9719 | 48 |
| trial_7 | 0.9698 | 48 |

**Conclusion:** Network capacity (features_base) is flexible. Smaller networks (24) can achieve the same performance as larger ones (48), suggesting the task doesn't require maximum capacity.

**Recommendation:** Use `features_base = 24` for efficiency, or `32-48` if memory allows.

---

### 5. **Batch Size: Flexible (2 or 4)**

**Finding:** Both batch sizes work well

| Trial | Dice | batch_size |
|-------|------|------------|
| trial_8 | 0.9725 | 4 |
| trial_1 | 0.9723 | 2 |
| trial_3 | 0.9721 | 2 |
| trial_4 | 0.9719 | 2 |
| trial_7 | 0.9698 | 2 |

**Conclusion:** Batch size has minimal impact on final performance. Use based on GPU memory constraints.

**Recommendation:** Use `batch_size = 4` if memory allows, otherwise `2` is fine.

---

### 6. **Patch Size: Flexible (Various Configurations)**

**Finding:** Different patch sizes achieve similar performance

| Trial | Dice | patch_size | Volume |
|-------|------|------------|--------|
| trial_8 | 0.9725 | [160, 160, 64] | 1,638,400 |
| trial_1 | 0.9723 | [128, 64, 128] | 1,048,576 |
| trial_3 | 0.9721 | [64, 128, 64] | 524,288 |

**Observation:** 
- trial_8 uses the largest patch size (1.6M voxels)
- trial_1 uses medium patch size (1.0M voxels)
- trial_3 uses smaller patch size (0.5M voxels)

**Conclusion:** Patch size is flexible, but larger patches (trial_8) may provide slightly better performance. However, the difference is minimal (0.0004 Dice).

**Recommendation:** Use `patch_size = [160, 160, 64]` for best performance, or smaller if memory constrained.

---

## Why Some Trials Failed

### Common Characteristics of Failed Trials

**Critical Finding:** All failed trials have `use_mask_for_norm = True`.

| Trial | Dice | use_mask_for_norm | n_conv_per_stage |
|-------|------|-------------------|------------------|
| trial_5 | 0.4830 | True | 2 |
| trial_0 | 0.4897 | True | 2 |
| trial_2 | 0.4945 | True | 2 |
| trial_6 | 0.5066 | True | 3 |
| trial_9 | 0.5115 | True | 3 |

**Conclusion:** 
- **100% of successful trials** have `use_mask_for_norm = False`
- **100% of failed trials** have `use_mask_for_norm = True`
- This is the **primary differentiator** between success and failure

**Additional observations:**
- 2 failed trials also use `n_conv_per_stage = 3` (trial_6, trial_9), but this is secondary to `use_mask_for_norm = True`
- Failed trials have various patch sizes, batch sizes, and feature bases, confirming that `use_mask_for_norm` is the critical factor

---

## Recommendations for Future Experiments

### Must-Have Parameters (Critical)
1. **use_mask_for_norm = False** (100% of top trials)
2. **n_conv_per_stage = 2** (100% of top trials)

### Should-Have Parameters (Important)
3. **batch_dice = False** (80% of top trials)

### Flexible Parameters (Can Vary)
4. **features_base**: 24, 32, or 48 (all work well)
5. **batch_size**: 2 or 4 (both work)
6. **patch_size**: Various configurations work, prefer larger if memory allows

### Optimal Configuration (trial_8)
```json
{
  "patch_size": [160, 160, 64],
  "batch_size": 4,
  "features_base": 24,
  "n_conv_per_stage": 2,
  "batch_dice": false,
  "use_mask_for_norm": false
}
```

---

## Statistical Summary

- **Success Rate:** 50% (5/10 trials achieved Dice > 0.96)
- **Performance Gap:** 0.4895 (difference between best and worst)
- **Consistency:** Top 5 trials are very close (0.9698 - 0.9725), suggesting the optimal configuration is well-defined
- **Robustness:** The best configuration (trial_8) achieved 0.9725, which is excellent for medical image segmentation

---

## Conclusion

The HPO experiment successfully identified optimal hyperparameters for CT tooth segmentation. The key finding is that **mask-based normalization is detrimental** for this task, and all successful trials avoid it. The optimal configuration (trial_8) achieves a Dice score of 0.9725, which is excellent performance for medical image segmentation.

**Conclusion:** The parameter space exhibits clear regions of success and failure. The parameter `use_mask_for_norm = False` is absolutely critical, representing the difference between excellent performance (Dice > 0.96) and complete failure (Dice < 0.52). This demonstrates a 100% correlation: all successful trials use `False`, while all failed trials use `True`.

---

*Generated from HPO analysis on Dataset001_GroundTruth*  
*Best Model: trial_8 (Dice: 0.9725)*

