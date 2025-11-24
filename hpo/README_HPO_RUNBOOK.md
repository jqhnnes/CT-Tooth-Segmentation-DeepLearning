# nnU-Net HPO Runbook

This folder contains everything needed to conduct hyperparameter optimization (HPO)
for nnU-Net on the CT Tooth Segmentation task. The goal: generate many plan variants
(via Optuna), train each trial in isolation, and compare how preprocessing choices
impact downstream Dice. All scripts, trial outputs, and logs live here so the entire
workflow can be reproduced from a single README.

## 0. How this folder is organized

- `nnunet_hpo_preprocess.py`, `nnunet_train_eval_pipeline.py`, … – helper scripts described per section below.
- `nnUNetPlans_template.json` – plan blueprint that Optuna copies/mutates when generating each trial.
  - Contains default patch size, batch size, network settings; every trial writes its own modified copy.
- `preprocessing_output/` – trial-specific preprocessed datasets (`trial_X/Dataset001...`).
- `training_output/` – archived results (checkpoints/logs) copied after each training run.
- `results/` – evaluation logs per trial/config/trainer.
- Use this README as the “landing page” whenever you open `hpo/` in GitHub.

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
python hpo/remap_labels.py --dataset Dataset001_GroundTruth --max_label 3
```

  - `--dataset`: name under `nnUNet_raw`; script reads `imagesTr/labelsTr`.
  - `--max_label`: clamp any label value above this number down to the limit.
  - Optional `--dry_run` shows stats without writing files.

### Start new trials (Optuna + nnUNetv2_preprocess)

```bash
python hpo/nnunet_hpo_preprocess.py --n_trials 5
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

- `fix_decoder_lengths.py` (repair old plan files):

```bash
python hpo/fix_decoder_lengths.py --root hpo/preprocessing_output/Dataset001_GroundTruth
```

  - `--root`: directory that contains the `trial_X` folders.
  - Optional `--dry_run` prints which files would change.

- Inspect trial labels if needed:

```bash
python hpo/check_trial_labels.py --source trials --trial trial_0
```

## 3. Fixing a broken case (e.g. AW062 spacing issue)

```bash
python hpo/fix_case_spacing_and_reprocess.py \
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
python hpo/nnunet_train_eval_pipeline.py --folds 0 --skip_evaluation
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
python hpo/nnunet_train_eval_pipeline.py \
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

## 6. Monitoring

- Training log: `data/nnUNet_results/.../fold_0/training_log_*.txt`
- Evaluation log: `hpo/results/<dataset>/<trial>/<config>/<trainer>/evaluation.log`
- Archived checkpoints: `hpo/training_output/<trial>/nnUNet_results/...`

## 7. Troubleshooting snippets

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

