#!/usr/bin/env python3
"""
Hyperparameter Optimization (HPO) preprocessing script for nnU-Net.

This script uses Optuna to generate multiple hyperparameter configurations and
runs nnUNetv2_preprocess for each trial. Each trial gets its own preprocessed
dataset with specific parameter settings.

Example:
    python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 50
"""
import argparse
import os
import sys
import json
import shutil
import subprocess
import optuna
import re
from copy import deepcopy

# Ensure project root is in Python path so `scripts` and other modules can be found
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load custom environment setup
from scripts.nnunet_env import load_env

load_env()

# ---- Paths / Configuration ----
dataset_name = "Dataset001_GroundTruth"
# Extract dataset ID from name (e.g., "Dataset001_GroundTruth" -> 1)
match = re.search(r"Dataset(\d+)", dataset_name)
if not match:
    raise ValueError(
        f"Could not extract dataset ID from '{dataset_name}'. "
        "Expected format: 'DatasetXXX_...'"
    )
dataset_id = int(match.group(1))

nnunet_raw = os.environ.get("nnUNet_raw")
nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
nnunet_results = os.environ.get("nnUNet_results")

if not all([nnunet_raw, nnunet_preprocessed, nnunet_results]):
    raise EnvironmentError(
        "nnUNet requires the environment variables 'nnUNet_raw', "
        "'nnUNet_preprocessed' and 'nnUNet_results'. Please set them "
        "e.g. in scripts/nnunet_env.sh."
    )

input_folder = os.path.join(nnunet_raw, dataset_name)
hpo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Base directory for HPO preprocessing outputs within hpo/
hpo_preprocessing_base = os.path.join(hpo_dir, "preprocessing_output")
template_plan_path = os.path.join(hpo_dir, "config", "nnUNetPlans_template.json")
dataset_output_dir = os.path.join(hpo_preprocessing_base, dataset_name)

os.makedirs(hpo_dir, exist_ok=True)
os.makedirs(hpo_preprocessing_base, exist_ok=True)
os.makedirs(dataset_output_dir, exist_ok=True)

# ---- Validation ----
if not os.path.exists(input_folder):
    raise FileNotFoundError(
        f"Input folder '{input_folder}' does not exist. Please set nnUNet_raw correctly."
    )
if not os.path.exists(
    os.path.join(input_folder, "imagesTr")
) or not os.path.exists(os.path.join(input_folder, "labelsTr")):
    raise FileNotFoundError(
        f"imagesTr or labelsTr missing in input folder '{input_folder}'."
    )

if not os.path.exists(template_plan_path):
    raise FileNotFoundError(
        f"Template plan '{template_plan_path}' not found.\n"
        "Create a plan template, e.g., by running nnUNetv2 planning steps once "
        "(see nnUNetv2 README) or create a JSON manually with the plan fields."
    )


# ---- Helper Functions ----
def detect_next_trial_index(base_dir):
    """
    Detects the highest trial index in the base directory.
    
    Args:
        base_dir: Directory containing trial_X folders
        
    Returns:
        Next available trial index (highest + 1)
    """
    max_idx = -1
    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            match = re.match(r"trial_(\d+)", entry)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def reserve_trial_slot(base_dir):
    """
    Returns the next available trial name and creates the directory immediately
    to prevent race conditions with parallel processes.
    
    Args:
        base_dir: Base directory for trial folders
        
    Returns:
        Tuple of (trial_idx, trial_name, trial_dir)
    """
    while True:
        trial_idx = detect_next_trial_index(base_dir)
        trial_name = f"trial_{trial_idx}"
        trial_dir = os.path.join(base_dir, trial_name)
        try:
            os.makedirs(trial_dir)
            return trial_idx, trial_name, trial_dir
        except FileExistsError:
            # Directory was created between detect() and makedirs() -> try again
            continue


def ensure_fingerprint_for_all_trials(base_dir: str, dataset_name: str, dataset_id: int):
    """
    Ensures that every trial directory contains a dataset_fingerprint.json.
    If none exists, it is extracted once and then distributed to all trials.
    
    Args:
        base_dir: Base directory containing trial folders
        dataset_name: Name of the dataset
        dataset_id: nnU-Net dataset ID
    """
    if not os.path.isdir(base_dir):
        return

    trial_dirs = sorted(
        [
            os.path.join(base_dir, entry)
            for entry in os.listdir(base_dir)
            if re.match(r"trial_\d+$", entry)
        ],
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not trial_dirs:
        return

    source_fingerprint = None
    for trial_dir in trial_dirs:
        candidate = os.path.join(trial_dir, dataset_name, "dataset_fingerprint.json")
        if os.path.exists(candidate):
            source_fingerprint = candidate
            break

    if not source_fingerprint:
        global_fingerprint = os.path.join(
            nnunet_preprocessed, dataset_name, "dataset_fingerprint.json"
        )
        if not os.path.exists(global_fingerprint):
            print(
                "[INFO] No dataset_fingerprint.json found. "
                "Extracting once via nnUNetv2_extract_fingerprint ..."
            )
            extract_cmd = ["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)]
            subprocess.run(extract_cmd, check=True)
        if os.path.exists(global_fingerprint):
            source_fingerprint = global_fingerprint

    if not source_fingerprint or not os.path.exists(source_fingerprint):
        print(
            "[WARN] Could not find or create dataset_fingerprint.json. "
            "Please check manually."
        )
        return

    for trial_dir in trial_dirs:
        target = os.path.join(trial_dir, dataset_name, "dataset_fingerprint.json")
        if os.path.abspath(source_fingerprint) == os.path.abspath(target):
            continue
        target_parent = os.path.dirname(target)
        if not os.path.isdir(target_parent):
            continue
        if os.path.exists(target):
            continue
        shutil.copy2(source_fingerprint, target)
        print(f"[{os.path.basename(trial_dir)}] dataset_fingerprint.json added.")

def replace_hpo_parameters(
    obj,
    patch_tuple,
    batch_size,
    features_per_stage=None,
    n_conv_per_stage=None,
    batch_dice=None,
    use_mask_for_norm=None,
    spacing=None,
    original_spacing=None,
    original_shape=None,
):
    """
    Traverses a dict/list and replaces HPO parameters:
    - patch_size: Patch size
    - batch_size: Batch size
    - features_per_stage: Number of features per stage
    - n_conv_per_stage: Number of convolutions per stage
    - batch_dice: Batch-Dice Loss flag
    - use_mask_for_norm: Mask for normalization
    - spacing: Target voxel spacing
    - median_image_size_in_voxels: Updated based on new spacing
    
    This function modifies obj in-place.
    
    Args:
        obj: Dictionary or list to traverse
        patch_tuple: Tuple of (patch_x, patch_y, patch_z)
        batch_size: Batch size value
        features_per_stage: List of features per stage
        n_conv_per_stage: List of convolutions per stage
        batch_dice: Boolean flag for batch dice
        use_mask_for_norm: Boolean flag for mask normalization
        spacing: Tuple of (sx, sy, sz) target spacing values
        original_spacing: Original spacing from template (for calculating new shape)
        original_shape: Original shape from template (for calculating new shape)
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            # Patch size
            if (
                ("patch" in lk or "patch_size" in lk)
                and isinstance(v, (list, tuple))
                and len(v) == 3
                and all(isinstance(x, int) for x in v)
            ):
                obj[k] = [
                    int(patch_tuple[0]),
                    int(patch_tuple[1]),
                    int(patch_tuple[2]),
                ]
            # Batch size
            elif "batch_size" in lk and isinstance(v, int):
                obj[k] = int(batch_size)
            # Features per stage
            elif "features_per_stage" in lk and isinstance(v, list) and features_per_stage is not None:
                obj[k] = features_per_stage
            # n_conv_per_stage_decoder must be handled before generic encoder branch
            elif (
                "n_conv_per_stage_decoder" in lk
                and isinstance(v, list)
                and n_conv_per_stage is not None
            ):
                # Decoder typically has n_stages-1 convolutions
                obj[k] = n_conv_per_stage[:-1] if len(n_conv_per_stage) > 1 else n_conv_per_stage
            # n_conv_per_stage (encoder)
            elif (
                "n_conv_per_stage" in lk
                and "decoder" not in lk
                and isinstance(v, list)
                and n_conv_per_stage is not None
            ):
                obj[k] = n_conv_per_stage
            # batch_dice
            elif "batch_dice" in lk and batch_dice is not None:
                obj[k] = batch_dice
            # use_mask_for_norm
            elif "use_mask_for_norm" in lk and isinstance(v, list) and use_mask_for_norm is not None:
                obj[k] = [use_mask_for_norm]
            # spacings (only adjust config spacing field)
            elif k == "spacing" and isinstance(v, list) and spacing is not None and len(v) == 3:
                obj[k] = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
            # median_image_size_in_voxels: update based on new spacing
            elif (
                "median_image_size_in_voxels" in lk
                and isinstance(v, list)
                and len(v) == 3
                and spacing is not None
                and original_spacing is not None
                and original_shape is not None
            ):
                # Calculate new shape after resampling: new_dim = old_dim * (old_spacing / new_spacing)
                new_shape = [
                    int(original_shape[i] * original_spacing[i] / spacing[i])
                    for i in range(3)
                ]
                obj[k] = [float(new_shape[0]), float(new_shape[1]), float(new_shape[2])]
            else:
                replace_hpo_parameters(
                    v,
                    patch_tuple,
                    batch_size,
                    features_per_stage,
                    n_conv_per_stage,
                    batch_dice,
                    use_mask_for_norm,
                    spacing,
                    original_spacing,
                    original_shape,
                )
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], (dict, list)):
                replace_hpo_parameters(
                    obj[i],
                    patch_tuple,
                    batch_size,
                    features_per_stage,
                    n_conv_per_stage,
                    batch_dice,
                    use_mask_for_norm,
                    spacing,
                    original_spacing,
                    original_shape,
                )
    # Primitive types are not traversed further


# ---- Optuna Objective Function ----
def objective(trial):
    """
    Optuna objective function that generates hyperparameters and runs preprocessing.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Proxy score (placeholder - should be replaced with actual validation metric)
    """
    # ===== PARAMETER SPACE =====
    # Spacing-first search with constrained patch/batch/features to avoid OOM.
    # Optuna requires a static search space per parameter across trials, so we
    # flatten spacing-dependent options into one categorical.
    spacing_candidates = [
        # High-quality options; may OOM on small GPUs.
        {
            "spacing": (0.1, 0.1, 0.1),
            "patches": [
                (64, 64, 64),
                (64, 96, 64),
                (96, 96, 96),
                (96, 128, 96),
                (128, 128, 96),
            ],
            "batch_sizes": [1],
            "features_base": [16, 24, 32],
        },
        {
            "spacing": (0.12, 0.12, 0.12),
            "patches": [
                (64, 64, 64),
                (64, 96, 64),
                (96, 96, 96),
                (96, 128, 96),
                (128, 128, 96),
            ],
            "batch_sizes": [1],
            "features_base": [16, 24, 32],
        },
    ]

    combo_choices = []
    for cfg in spacing_candidates:
        for p in cfg["patches"]:
            for b in cfg["batch_sizes"]:
                for f in cfg["features_base"]:
                    combo_choices.append(
                        (cfg["spacing"], p, b, f)
                    )

    spacing, patch, batch_size, features_base = trial.suggest_categorical(
        "spacing_patch_batch_features", combo_choices
    )
    features_per_stage = [
        features_base,
        features_base * 2,
        features_base * 4,
        features_base * 8,
        features_base * 10,
        features_base * 10
    ]
    
    # Number of convolutions per stage fixed to 2 (best across analysis)
    n_conv_per_stage = 2
    n_conv_list = [n_conv_per_stage] * 6  # 6 stages
    
    # Batch-Dice Loss (mostly false, but allow exploration)
    batch_dice = trial.suggest_categorical("batch_dice", [False, False, True])
    
    # Masked normalization is detrimental -> hardcode False
    use_mask_for_norm = False

    trial_idx, trial_name, trial_output_dir = reserve_trial_slot(dataset_output_dir)
    # Ensure trial directory exists so we can persist params/errors even on failure
    os.makedirs(trial_output_dir, exist_ok=True)
    trial_plan_path = os.path.join(hpo_dir, "config", f"nnUNetPlans_temp_{trial_name}.json")

    # Persist chosen hyperparameters for debugging (even if preprocess fails)
    params_out = os.path.join(trial_output_dir, "params.json")
    with open(params_out, "w") as f:
        json.dump(
            {
                "spacing": spacing,
                "patch": patch,
                "batch_size": batch_size,
                "features_base": features_base,
                "features_per_stage": features_per_stage,
                "batch_dice": batch_dice,
                "trial_name": trial_name,
            },
            f,
            indent=2,
        )
    
    # Temporary directory for nnUNet (nnUNet expects: nnUNet_preprocessed/DatasetXXX/...)
    # We set nnUNet_preprocessed to dataset_output_dir so nnUNet creates DatasetXXX/ there
    # Afterwards we move everything to trial_X/DatasetXXX/
    temp_preprocessed_base = dataset_output_dir
    temp_preprocessed_dir = os.path.join(temp_preprocessed_base, dataset_name)
    os.makedirs(temp_preprocessed_dir, exist_ok=True)

    # 1) Load template, modify and save trial plan
    with open(template_plan_path, "r") as f:
        plan = json.load(f)

    plan_mod = deepcopy(plan)
    
    # Extract original spacing and shape from template for median_image_size_in_voxels calculation
    original_spacing = tuple(plan.get("original_median_spacing_after_transp", [0.04, 0.04, 0.04]))
    original_shape = tuple(plan.get("original_median_shape_after_transp", [239, 253, 254]))
    
    replace_hpo_parameters(
        plan_mod,
        patch,
        batch_size,
        features_per_stage,
        n_conv_list,
        batch_dice,
        use_mask_for_norm,
        spacing,
        original_spacing,
        original_shape,
    )

    # Save the modified plans file in the hpo directory (for traceability)
    with open(trial_plan_path, "w") as f:
        json.dump(plan_mod, f, indent=2)
    
    # 2) Copy the plans file to the temporary location where nnUNet expects it
    # nnUNet looks for: nnUNet_preprocessed/DatasetXXX/nnUNetPlans.json
    plans_in_temp = os.path.join(temp_preprocessed_dir, "nnUNetPlans.json")
    shutil.copy2(trial_plan_path, plans_in_temp)

    # 3) Copy dataset.json because nnUNetv2_preprocess expects it in the nnUNet_preprocessed folder
    dataset_json_src = os.path.join(input_folder, "dataset.json")
    if not os.path.exists(dataset_json_src):
        raise FileNotFoundError(
            f"dataset.json not found at '{dataset_json_src}'. "
            "Please ensure your nnUNet_raw dataset is complete."
        )
    dataset_json_temp = os.path.join(temp_preprocessed_dir, "dataset.json")
    shutil.copy2(dataset_json_src, dataset_json_temp)
    
    # 4) Call nnUNetv2_preprocess with the trial plan
    # Temporarily set nnUNet_preprocessed to dataset_output_dir
    # nnUNet will then create: preprocessing_output/DatasetXXX/DatasetXXX/...
    env = os.environ.copy()
    env["nnUNet_preprocessed"] = os.path.abspath(temp_preprocessed_base)
    
    # nnUNet uses -plans_name (without .json) and looks for the file in the dataset directory
    cmd = [
        "nnUNetv2_preprocess",
        "-d",
        str(dataset_id),
        "-plans_name",
        "nnUNetPlans",  # Name without .json - nnUNet looks for nnUNetPlans.json
        "-c",
        "3d_fullres",
        "--num_processes",
        "1",  # force single worker to minimize RAM
    ]

    print(f"\n[{trial_name}] ===== HPO-Parameter =====")
    print(f"  Patch: {patch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Features base: {features_base} -> {features_per_stage}")
    print(f"  Convs per stage: {n_conv_per_stage}")
    print(f"  Batch Dice: {batch_dice}")
    print(f"  Use mask for norm: {use_mask_for_norm}")
    print(f"  Target spacing: {spacing}")
    print(f"  Output: {trial_output_dir}")
    print("Command:", " ".join(cmd))

    # Capture stdout/stderr for detailed error reporting
    err_path = os.path.join(trial_output_dir, "error.log")
    log_path = os.path.join(trial_output_dir, "preprocess.log")
    
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                check=True,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
    except subprocess.CalledProcessError as e:
        # Read the log file to get detailed error message
        error_details = f"preprocess failed: {e}\n"
        if os.path.exists(log_path):
            with open(log_path, "r") as log_file:
                log_content = log_file.read()
                error_details += f"\n=== Full nnUNetv2_preprocess output ===\n{log_content}\n"
        
        with open(err_path, "w") as ef:
            ef.write(error_details)
        print(f"[{trial_name}] Preprocess failed: {e}")
        print(f"[{trial_name}] Full error log saved to: {err_path}")
        raise optuna.TrialPruned(f"preprocess failed: {e}")
    except Exception as e:
        error_details = f"unexpected failure: {e}\n"
        if os.path.exists(log_path):
            with open(log_path, "r") as log_file:
                log_content = log_file.read()
                error_details += f"\n=== Full nnUNetv2_preprocess output ===\n{log_content}\n"
        
        with open(err_path, "w") as ef:
            ef.write(error_details)
        print(f"[{trial_name}] Unexpected failure: {e}")
        print(f"[{trial_name}] Full error log saved to: {err_path}")
        raise optuna.TrialPruned(f"unexpected failure: {e}")
    
    # 5) Ensure dataset_fingerprint.json exists in the temporary directory
    # (usually created automatically by preprocessing, but check for safety)
    fingerprint_in_temp = os.path.join(temp_preprocessed_dir, "dataset_fingerprint.json")
    if not os.path.exists(fingerprint_in_temp):
        print(f"[{trial_name}] dataset_fingerprint.json missing, extracting now...")
        extract_cmd = ["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)]
        # Use the temporary nnUNet_preprocessed (env was already set above)
        subprocess.run(extract_cmd, check=True, env=env)
        if not os.path.exists(fingerprint_in_temp):
            print(f"[WARN] {trial_name}: Could not create dataset_fingerprint.json.")
    
    # 6) Move all files from temporary directory to trial_X/DatasetXXX/
    trial_dataset_dir = os.path.join(trial_output_dir, dataset_name)
    os.makedirs(trial_dataset_dir, exist_ok=True)
    
    if os.path.exists(temp_preprocessed_dir):
        # Move all contents from DatasetXXX/ to trial_X/
        # (dataset_fingerprint.json is automatically moved with it)
        for item in os.listdir(temp_preprocessed_dir):
            src = os.path.join(temp_preprocessed_dir, item)
            dst = os.path.join(trial_dataset_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        # Delete the empty DatasetXXX directory
        try:
            os.rmdir(temp_preprocessed_dir)
        except OSError:
            pass  # Directory not empty or doesn't exist
    
    # Confirmation
    fingerprint_in_trial = os.path.join(trial_dataset_dir, "dataset_fingerprint.json")
    if os.path.exists(fingerprint_in_trial):
        print(f"[{trial_name}] ✓ dataset_fingerprint.json successfully copied to trial directory")
    else:
        print(f"[WARN] {trial_name}: dataset_fingerprint.json missing in trial directory after move!")

    # === PLACEHOLDER METRIC ===
    # Preprocessing does not provide a performance metric. You must replace this part
    # if you want real validation results (e.g., train a model briefly and evaluate).
    #
    # IMPORTANT: The current proxy score is only a placeholder!
    # It simply maximizes the sum of parameters, which is NOT meaningful.
    # For real HPO, you must use a real metric here (e.g., Dice score after training).
    proxy_score = sum(patch) + batch_size + features_base + n_conv_per_stage - sum(spacing)
    print(f"[{trial_name}] Proxy score: {proxy_score} (PLACEHOLDER ONLY!)")
    return proxy_score


# ---- Start Study ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start nnUNet HPO preprocessing trials."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of new trials to start in this run (default: 10).",
    )
    args = parser.parse_args()

    next_idx = detect_next_trial_index(dataset_output_dir)
    print(
        f"Starting Optuna run with {args.n_trials} new trials. "
        f"Next available directory: trial_{next_idx}."
    )

    # NOTE: With extended parameters, the search space is larger!
    # Total: ~27 * 2 * 3 * 2 * 2 * 2 = ~1,296 possible combinations
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    if study.trials:
        try:
            print("Best trial parameters:", study.best_trial.params)
            print("Best trial value:", study.best_value)
        except ValueError:
            print("[WARN] No trials completed successfully; no best trial available.")
    else:
        print("No completed trials found.")

    # Ensure all trials have a dataset_fingerprint.json
    ensure_fingerprint_for_all_trials(dataset_output_dir, dataset_name, dataset_id)
