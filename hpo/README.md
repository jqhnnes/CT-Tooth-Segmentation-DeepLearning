# nnU-Net HPO Runbook

Everything in `hpo/` to run hyperparameter optimization (HPO) for CT tooth segmentation with nnU-Net: generate plan variants (Optuna), train each trial, and analyze results.

## 0. Layout (current files)

```
hpo/
├── scripts/
│   ├── preprocessing/          # HPO + nnUNetv2_preprocess
│   │   └── nnunet_hpo_preprocess.py
│   ├── training/               # Training & optional eval
│   │   └── nnunet_train_eval_pipeline.py
│   ├── analysis/               # Summaries/plots
│   │   ├── summarize_trials.py
│   │   └── plot_trials_summary.py
│   └── utils/
│       └── check_trial_labels.py
├── config/                     # nnUNetPlans_template.json
├── preprocessing_output/       # Trial-specific preprocessed data
├── training_output/            # Archived training runs
├── analysis/                   # trials_summary.json, plots/
└── results/                    # Evaluation logs (if generated)
```

## 1. Environment

```bash
cd /ssd/geiger/CT-Tooth-Segmentation-DeepLearning
conda activate /ssd/geiger/myenv
source scripts/nnunet_env.sh   # exports nnUNet_raw/_preprocessed/_results
```

Run commands from the project root with the environment activated.

## 2. Preprocessing / HPO trial generation (Optuna)

Start new trials (Optuna + nnUNetv2_preprocess):

```bash
python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 5
```

- Creates `n_trials` plan variants under `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X`.
- Current high-end search space (VRAM-heavy, near OOM):
  - Spacing 0.075 mm: patch (192×256×128), batch 1, features_base 64
  - Spacing 0.080 mm: patch (224×256×128), batch 1, features_base 72
  - Batch-Dice toggled per trial (True/False)
- Key option: `--n_trials`.
- Ensure `nnUNet_raw` points to the cleaned dataset before running.

After preprocessing:

```bash
python hpo/scripts/utils/check_trial_labels.py --dataset Dataset001_GroundTruth
```

Use it if you need to re-check labels; each trial should contain `nnUNetPlans.json` and `dataset_fingerprint.json`.

## 3. Resetting stale fold data (optional)

```bash
rm -rf data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
```

Clears leftover checkpoints/logs if a previous run aborted. The training pipeline already cleans folds, but manual cleanup can help debugging.

## 4. Training pipeline

Train only (1 fold, no evaluation):

```bash
python hpo/scripts/training/nnunet_train_eval_pipeline.py --folds 0 --skip_evaluation
```

Useful parameters:
- `--folds 0 1 2`: choose folds
- `--trials trial_0 trial_5`: restrict trials
- `--skip_evaluation`: train only
- `--stop_on_error`: abort on first failure

Train then evaluate later:

```bash
python hpo/scripts/training/nnunet_train_eval_pipeline.py \
  --folds 0 \
  --only_evaluate \
  --trials trial_0 trial_1
```

Options:
- `--only_evaluate`: skip training, read archived results
- `--eval_timeout 7200`: timeout per eval call
- `--device cuda:1` / `--device cpu`: force device (CPU fallback if GPU OOM)

Train a specific trial manually (example: trial_43):

```bash
source scripts/nnunet_env.sh
export nnUNet_preprocessed=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/preprocessing_output/Dataset001_GroundTruth/trial_43
export nnUNet_results=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/training_output/trial_43/nnUNet_results
nnUNetv2_train Dataset001_GroundTruth 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans
```

- Repeat per fold by changing the last argument (`0 → 1`, etc.).
- To continue training, rerun the same command in the same `nnUNet_results` path; nnU-Net resumes from checkpoints (no `--continue_training` flag).
- Log GPU usage while training:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 60 > logs/trial_gpu_usage.csv
```

## 5. Monitoring

- Training logs: `data/nnUNet_results/.../fold_X/training_log_*.txt`
- Evaluation logs: `hpo/results/<dataset>/<trial>/<config>/<trainer>/evaluation.log`
- Archived checkpoints: `hpo/training_output/<trial>/nnUNet_results/...`

## 7. Analysis

Trial parameters + scores overview:

```bash
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/summarize_trials.py
```

Reads all trials in `hpo/training_output`, pulls spacing/patch/batch/features_base from plans and validation Dice from `fold_0/validation/summary.json`. Outputs `hpo/analysis/trials_summary.json`.

Quick plot (Dice vs spacing):

```bash
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/plot_trials_summary.py
```

Uses `trials_summary.json`, plots validation Dice vs spacing, colored by `features_base`. Output: `hpo/analysis/plots/trials_dice_vs_spacing.png`.

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

Keep this runbook up to date when scripts or procedures change.

