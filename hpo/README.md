# nnU-Net HPO Runbook

This folder contains everything needed to conduct hyperparameter optimization (HPO)
for nnU-Net on the CT Tooth Segmentation task. The goal: generate many plan variants
(via Optuna), train each trial in isolation, and compare how preprocessing choices
impact downstream Dice. All scripts, trial outputs, and logs live here so the entire
workflow can be reproduced from a single README.

## 0. How this folder is organized

```
hpo/
├── scripts/                    # All Python scripts
│   ├── preprocessing/          # Preprocessing scripts
│   │   └── nnunet_hpo_preprocess.py
│   ├── training/               # Training & Evaluation
│   │   └── nnunet_train_eval_pipeline.py
│   ├── analysis/               # Analysis & Comparison
│   │   ├── compare_top3_trials.py
│   │   ├── update_trial_summary.py
│   │   ├── plot_hpo_results.py
│   │   └── prepare_best_model.py
│   └── utils/                  # Utility scripts
│       ├── check_trial_labels.py
│       ├── fix_case_spacing_and_reprocess.py
│       ├── fix_decoder_lengths.py
│       └── remap_labels.py
├── config/                     # Templates & Configuration
│   └── nnUNetPlans_template.json
├── docs/                       # Additional Documentation
│   └── BEST_PARAMETERS_SUMMARY.md
├── analysis/                   # Analysis Results / Plots
│   ├── best_parameters_summary.json
│   ├── top3_analysis.json
│   └── plots/
├── best_model/                 # Best Model (trial_8)
├── preprocessing_output/        # Trial-specific preprocessed datasets
├── training_output/            # Archived training results
└── results/                    # Evaluation logs per trial
```

## 1. Environment

```bash
cd /ssd/geiger/CT-Tooth-Segmentation-DeepLearning
conda activate /ssd/geiger/myenv
source scripts/nnunet_env.sh   # exports nnUNet_raw/_preprocessed/_results
```

- Always execute new commands from the project root after the environment has been activated.

## 2. Preprocessing / HPO trial generation

### Prepare raw dataset / labels (optional)

- If labels contain unexpected IDs, remap them first:

- `remap_labels.py` (fix invalid label IDs before preprocessing):

```bash
python hpo/scripts/utils/remap_labels.py --dataset Dataset001_GroundTruth --max_label 3
```

  - `--dataset`: name under `nnUNet_raw`; script reads `imagesTr/labelsTr`.
  - `--max_label`: clamp any label value above this number down to the limit.
  - Optional `--dry_run` shows stats without writing files.

### Start new trials (Optuna + nnUNetv2_preprocess)

```bash
python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 5
```

- Generates `n_trials` new hyperparameter samples, runs `nnUNetv2_preprocess` for each,
  and stores the outputs in `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X`.
- Important args:
  - `--n_trials`: number of Optuna samples to add.
  - (Inside the script) patch size, batch size, feature configs are randomized.
- Before running, make sure `nnUNet_raw` contains the cleaned dataset (e.g. label remap).

### After preprocessing / trial maintenance

- Verify that every `trial_X` contains `Dataset001_GroundTruth/nnUNetPlans.json`
  and `dataset_fingerprint.json`.
- Some older trials may need their decoder setup fixed:

- `fix_decoder_lengths.py` (repair old plan files so decoder depth matches stages):

```bash
python hpo/scripts/utils/fix_decoder_lengths.py --dataset Dataset001_GroundTruth
```

  - Runs through `hpo/preprocessing_output/Dataset001_GroundTruth/trial_*` and truncates decoder lists.

- Inspect trial labels if needed:

```bash
python hpo/scripts/utils/check_trial_labels.py --source trials --trial trial_0
```

## 3. Fixing a broken case (e.g. AW062 spacing issue)

```bash
python hpo/scripts/utils/fix_case_spacing_and_reprocess.py \
    --case_id AW062-C0005656 \
    --spacing 0.04 0.04 0.04 \
    --backup
```

- Rewrites the raw NIfTI headers (backup stored as `.bak` next to the file).
- Regenerates this case inside every `trial_X` so all trials stay in sync.
- Repeat later with `--skip_raw_fix` to rebuild trial data only.

## 4. Resetting stale fold data (optional)

```bash
rm -rf data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
```

- Removes leftover checkpoints/logs if a previous run aborted mid-validation.
- The pipeline already clears `fold_*`, but manual cleanup can be handy when debugging.

## 5. Launching the training pipeline

### Train only (1 fold, no evaluation)

```bash
python hpo/scripts/training/nnunet_train_eval_pipeline.py --folds 0 --skip_evaluation
```

- Trains all pending trials fold 0, archives results under `hpo/training_output/...`.
- Useful parameters:
  - `--folds 0 1 2`: choose which folds to run (strings accepted).
  - `--trials trial_0 trial_5`: restrict to selected trials.
  - `--skip_evaluation`: train only, leave evaluation for later.
  - `--stop_on_error`: abort on first failure (default: continue).

### Train + evaluate later

1. Train with `--skip_evaluation` (fast screening).
2. Evaluate archived results afterwards:

```bash
python hpo/scripts/training/nnunet_train_eval_pipeline.py \
    --folds 0 \
    --only_evaluate \
    --trials trial_0 trial_1
```

- Expects archived folders in `hpo/training_output/trial_X/...`.
- Additional options:
  - `--only_evaluate`: skip training, read from archived results.
  - `--eval_timeout 7200`: abort evaluation after N seconds (per call).
  - `--device cuda:1` / `--device cpu`: force a specific device.
  - Automatic CPU fallback occurs if GPU evaluation raises OOM.

### Train a specific trial manually (e.g., trial_15 long-run)

```bash
source scripts/nnunet_env.sh
export nnUNet_preprocessed=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/preprocessing_output/Dataset001_GroundTruth/trial_15
export nnUNet_results=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/training_output/trial_15/nnUNet_results
nnUNetv2_train Dataset001_GroundTruth 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --npz
```

- Repeat per fold by changing the last argument (`0 → 1`, etc.).
- After the default schedule finishes you can continue fine-tuning:

```bash
nnUNetv2_train Dataset001_GroundTruth 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --continue_training
```

- Log GPU constraints while training:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 > logs/trial15_gpu_usage.csv
```

## 6. Monitoring

- Training log: `data/nnUNet_results/.../fold_0/training_log_*.txt`
- Evaluation log: `hpo/results/<dataset>/<trial>/<config>/<trainer>/evaluation.log`
- Archived checkpoints: `hpo/training_output/<trial>/nnUNet_results/...`

## 7. Analysis & spacing-focused evaluation

### Regenerate JSON summaries (includes spacing ranking)

```bash
python hpo/scripts/analysis/update_trial_summary.py
```

- Reads every trial under `hpo/results/...` + `hpo/preprocessing_output/...`.
- Produces `hpo/analysis/best_parameters_summary.json` and `top3_analysis.json`
  with explicit spacing statistics.

### Produce plots (Dice vs. parameters, incl. spacing)

```bash
pip install matplotlib  # once
python hpo/scripts/analysis/plot_hpo_results.py
```

- Outputs PNGs into `hpo/analysis/plots/`, e.g. `dice_vs_spacing.png`.
- Use these when reporting how spacing vs Dice behaves or when checking GPU limits:
  cross-reference the chosen spacing with your `nvidia-smi` logs to justify feasible
  resolutions.

### Compare arbitrary trial sets (predictions + evaluation)

```bash
python hpo/scripts/analysis/compare_top3_trials.py \
    --trials trial_1 trial_3 trial_4 trial_8 trial_15 trial_16 trial_17 trial_18 \
    --testset labelsVal \
    --folds 0
```

- `labelsVal` (or `labelsTr`) must exist under `data/nnUNet_raw/Dataset001_GroundTruth`.
- The script automatically switches `nnUNet_preprocessed`/`nnUNet_results` for each trial.

## 8. Troubleshooting snippets

Check spacing/shape of a single preprocessed case:

```bash
/ssd/geiger/myenv/bin/python - <<'PY'
from batchgenerators.utilities.file_and_folder_operations import load_pickle
pkl = load_pickle('hpo/preprocessing_output/Dataset001_GroundTruth/trial_0/'
                  'Dataset001_GroundTruth/nnUNetPlans_3d_fullres/AW062-C0005656.pkl')
print('spacing:', pkl['spacing'])
print('shape:', pkl['shape_after_cropping_and_before_resampling'])
PY
```

List validation predictions produced by the latest run:

```bash
ls data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation
```

---

Update this runbook whenever new operational steps or gotchas come up. A quick
note here usually saves future digging through shell history.

