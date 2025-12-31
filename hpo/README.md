# nnU-Net HPO Runbook

This folder contains everything needed to conduct hyperparameter optimization (HPO)
for nnU-Net on the CT Tooth Segmentation task. The goal: generate many plan variants
(via Optuna), train each trial in isolation, and compare how preprocessing choices
impact downstream Dice. All scripts, trial outputs, and logs live here so the entire
workflow can be reproduced from a single README.

## 0. How this folder is organized (current files)

```
hpo/
├── scripts/
│   ├── preprocessing/          # HPO + nnUNetv2_preprocess
│   │   └── nnunet_hpo_preprocess.py
│   ├── training/               # Training & optional eval
│   │   └── nnunet_train_eval_pipeline.py
│   ├── postprocessing/         # Inference + PP + eval helpers
│   │   ├── nnunet_tta_postprocess.py
│   │   └── evaluate_tta_pp.py
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

- Always execute new commands from the project root after the environment has been activated.

## 2. Preprocessing / HPO trial generation (Optuna)

### Start new trials (Optuna + nnUNetv2_preprocess)

```bash
python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 5
```

- Generiert `n_trials` neue Plan-Varianten und legt sie unter `hpo/preprocessing_output/Dataset001_GroundTruth/trial_X` ab.
- Aktueller, fokussierter High-End-Suchraum (VRAM-hungrig, nahe OOM):
  - Spacing 0.095: Patches (128×128×96), (128×160×96), (160×160×96); features_base {32,40,48}; batch 1
  - Spacing 0.10:  Patches (128×128×96), (128×160×96), (160×160×96); features_base {32,40,48}; batch 1
  - Spacing 0.105: Patches (128×128×96), (128×160×96); features_base {32,40,48}; batch 1
- Wichtigste Option: `--n_trials` (Anzahl neuer Samples).
- Before running, make sure `nnUNet_raw` contains the cleaned dataset (e.g. label remap).

### Nach dem Preprocessing

- Prüfe, dass jede Trial-Mappe `Dataset001_GroundTruth/nnUNetPlans.json` und `dataset_fingerprint.json` enthält.
- Labels prüfen (falls nötig):

```bash
python hpo/scripts/utils/check_trial_labels.py --source trials --trial trial_0
```

## 3. Resetting stale fold data (optional)

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

### Train a specific trial manually (e.g., trial_43)

```bash
source scripts/nnunet_env.sh
export nnUNet_preprocessed=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/preprocessing_output/Dataset001_GroundTruth/trial_43
export nnUNet_results=/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/training_output/trial_43/nnUNet_results
nnUNetv2_train Dataset001_GroundTruth 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans
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

## 6. Inference + Postprocessing + Evaluation

### TTA-Prediction + Postprocessing (optional Evaluation)

```bash
/ssd/geiger/myenv/bin/python hpo/scripts/postprocessing/nnunet_tta_postprocess.py \
  --trials trial_43 \
  --folds 0 \
  --input_dir data/nnUNet_raw/Dataset001_GroundTruth/imagesTs \
  --eval_labels data/nnUNet_raw/Dataset001_GroundTruth/labelsTs
```

- Schritte: TTA-Predict → find_best_configuration (mit -f Folds) → apply_postprocessing → optional Evaluate.
- Flags:
  - `--trials`: spezifische Trials, sonst alle.
  - `--folds`: Folds (Default 0).
  - `--input_dir`: Eingabebilder (z. B. imagesTs).
  - `--eval_labels`: Groundtruth für Eval; weglassen, wenn nur Preds/PP erzeugt werden sollen.
  - `--pred_subdir` / `--suffix`: Zielordnernamen (Default: labelsTs_tta → labelsTs_tta_pp).
  - `--skip_predict`, `--skip_find`: falls nur PP angewendet werden soll.

### Evaluate existing labelsTs_tta_pp + ranking

```bash
python hpo/scripts/postprocessing/evaluate_tta_pp.py --folds 0
```

- Sucht pro Trial `labelsTs_tta_pp`, schreibt Summary nach `hpo/analysis/trial_X_labelsTs_tta_pp_summary.json`.
- `--force` überschreibt vorhandene Summary.
- Am Ende Ranking mit ΔDice vs. labelsTs / labelsTs_tta (falls vorhanden).

## 7. Monitoring

- Training log: `data/nnUNet_results/.../fold_0/training_log_*.txt`
- Evaluation log: `hpo/results/<dataset>/<trial>/<config>/<trainer>/evaluation.log`
- Archived checkpoints: `hpo/training_output/<trial>/nnUNet_results/...`

## 8. Analysis (quick)

### Trial parameters + scores overview

```bash
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/summarize_trials.py
```

- Reads all trials in `hpo/training_output`, pulls spacing/patch/batch/features_base from plans and Dice from `hpo/analysis/trial_*_labelsTs[_tta_pp]_summary.json`. Writes `hpo/analysis/trials_summary.json`.

### Quick plot Dice vs. spacing

```bash
/ssd/geiger/myenv/bin/python hpo/scripts/analysis/plot_trials_summary.py
```

- Uses `trials_summary.json`, plots Dice (tta_pp if available, else labelsTs) vs spacing, colors by `features_base`. Output: `hpo/analysis/plots/trials_dice_vs_spacing.png`.

## 9. Troubleshooting snippets

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

