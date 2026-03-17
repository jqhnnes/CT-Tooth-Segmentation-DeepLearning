#!/usr/bin/env python3
"""
Automated training and evaluation pipeline for preprocessed HPO trials.

This script trains and optionally evaluates models for HPO trials located in
`hpo/preprocessing_output/<Dataset>/trial_X`. It handles staging of preprocessed
data, training, result archiving, and evaluation.

Example:
    python hpo/scripts/training/nnunet_train_eval_pipeline.py \
        --trials trial_0 trial_3 \
        --folds 0 1 \
        --trainer nnUNetTrainer \
        --configuration 3d_fullres
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nnunet_env import load_env

load_env()

TRAINING_OUTPUT_ROOT = Path("hpo") / "training_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate preprocessed HPO trials sequentially."
    )
    parser.add_argument(
        "--dataset_name",
        default="Dataset001_GroundTruth",
        help="Dataset name (must match folder under nnUNet_raw).",
    )
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Specific trials (e.g., trial_0 trial_5). Default: all available.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds to train (e.g., 0 1 2 3 4).",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Trainer class for nnUNetv2 (default: nnUNetTrainer).",
    )
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnUNetv2 configuration (e.g., 3d_fullres, 3d_lowres).",
    )
    parser.add_argument(
        "--plans_name",
        default="nnUNetPlans",
        help="Name of plans file (without .json). Must be present in each trial folder.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device string for nnUNetv2_train (e.g., cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Train only, no nnUNetv2_evaluate calls.",
    )
    parser.add_argument(
        "--only_evaluate",
        action="store_true",
        help="Skip training and run only nnUNetv2_evaluate.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop after first failed trial (default: continue).",
    )
    parser.add_argument(
        "--eval_timeout",
        type=int,
        default=7200,  # 2 hours default timeout
        help="Timeout for evaluation in seconds (default: 7200 = 2h). 0 = no timeout.",
    )
    return parser.parse_args()


def ensure_env_vars():
    """Ensure all required nnUNet environment variables are set."""
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(
            f"Missing nnUNet environment variables: {', '.join(missing)}. "
            "Please set them in scripts/nnunet_env.sh."
        )


def extract_dataset_id(dataset_name: str) -> int:
    """Extract dataset ID from dataset name (e.g., 'Dataset001_GroundTruth' -> 1)."""
    match = re.search(r"Dataset(\d+)", dataset_name)
    if not match:
        raise ValueError(
            f"Could not extract dataset ID from '{dataset_name}'. "
            "Expected format: 'DatasetXXX_...'."
        )
    return int(match.group(1))


def list_trial_dirs(base_dir: Path) -> List[Path]:
    """List all trial directories in the base directory."""
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Trial base directory '{base_dir}' does not exist. "
            "Please run preprocessing first."
        )
    trial_dirs: List[Path] = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and re.match(r"trial_\d+$", entry.name):
            trial_dirs.append(entry)
    trial_dirs.sort(key=lambda p: int(p.name.split("_")[1]))
    return trial_dirs


def filter_trials(all_trials: List[Path], selection: Iterable[str] | None) -> List[Path]:
    if not selection:
        return all_trials
    selection_set = set(selection)
    resolved: List[Path] = []
    available = {t.name: t for t in all_trials}
    missing = sorted(selection_set - set(available))
    if missing:
        raise FileNotFoundError(
            f"The following trials do not exist under preprocessing_output: "
            f"{', '.join(missing)}"
        )
    for name in selection:
        resolved.append(available[name])
    return resolved


def get_archived_results_dir(trial_name: str, dataset_name: str) -> Path:
    return (
        TRAINING_OUTPUT_ROOT
        / trial_name
        / "nnUNet_results"
        / dataset_name
    ).resolve()


def skip_completed_trials(
    trials: List[Path],
    dataset_name: str,
    only_evaluate: bool,
) -> List[Path]:
    if only_evaluate:
        return trials

    remaining: List[Path] = []
    skipped: List[str] = []
    for trial_dir in trials:
        archived_dir = get_archived_results_dir(trial_dir.name, dataset_name)
        if archived_dir.exists():
            skipped.append(trial_dir.name)
        else:
            remaining.append(trial_dir)

    if skipped:
        print(
            f"[INFO] Skipping already trained trials "
            f"(present in training_output): {skipped}"
        )
    return remaining


def stage_preprocessed_dataset(trial_dir: Path, dataset_name: str) -> Tuple[Path, bool]:
    preprocessed_root = Path(os.environ["nnUNet_preprocessed"]).resolve()
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = preprocessed_root / dataset_name
    if dataset_dir.exists() or dataset_dir.is_symlink():
        if dataset_dir.is_symlink() or dataset_dir.is_file():
            dataset_dir.unlink()
        else:
            shutil.rmtree(dataset_dir)

    source_dataset_dir = trial_dir / dataset_name
    if not source_dataset_dir.exists():
        raise FileNotFoundError(
            f"{source_dataset_dir} does not exist. Please run preprocessing before training."
        )

    try:
        dataset_dir.symlink_to(source_dataset_dir.resolve(), target_is_directory=True)
        return dataset_dir, True
    except OSError:
        shutil.copytree(source_dataset_dir, dataset_dir)
        return dataset_dir, False


def cleanup_staged_dataset(dataset_dir: Path, was_symlink: bool):
    if not dataset_dir.exists() and not dataset_dir.is_symlink():
        return
    if was_symlink or dataset_dir.is_symlink():
        dataset_dir.unlink(missing_ok=True)
    else:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def prepare_eval_directory(
    dataset_name: str, trial_name: str, configuration: str, trainer: str
) -> Path:
    eval_base = (
        Path("hpo")
        / "results"
        / dataset_name
        / trial_name
        / configuration
        / trainer
    )
    eval_base.mkdir(parents=True, exist_ok=True)
    return eval_base


def persist_training_results(
    dataset_name: str, trial_name: str
) -> Path | None:
    results_root = Path(os.environ["nnUNet_results"]).resolve()
    source_dir = results_root / dataset_name
    if not source_dir.exists():
        print(f"[WARN] No nnUNet_results directory found at {source_dir}.")
        return None

    target_dir = get_archived_results_dir(trial_name, dataset_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    return target_dir


def cleanup_fold_results(
    dataset_name: str,
    trainer: str,
    plans_name: str,
    configuration: str,
    fold: str,
):
    """
    Removes any remnants of old training runs (e.g., from previous trials)
    so each training starts with a clean directory.
    """
    results_root = Path(os.environ["nnUNet_results"]).resolve()
    run_dir = (
        results_root
        / dataset_name
        / f"{trainer}__{plans_name}__{configuration}"
        / f"fold_{fold}"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def run_cmd(cmd: List[str], env: dict, log_prefix: str):
    cmd_str = " ".join(cmd)
    print(f"[{log_prefix}] -> {cmd_str}")
    subprocess.run(cmd, check=True, env=env)


def start_gpu_logger(log_file: Path) -> tuple[Optional[subprocess.Popen], Optional[object]]:
    """
    Start nvidia-smi logging to CSV. Returns (process, file_handle).
    """
    try:
        fh = log_file.open("w")
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv",
                "-l",
                "30",
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        return proc, fh
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found; GPU logging disabled.")
        return None, None


def stop_gpu_logger(proc: Optional[subprocess.Popen], fh: Optional[object]) -> None:
    if proc:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if fh:
        fh.close()


def run_eval_cmd(
    cmd: List[str],
    env: dict,
    log_file: Path,
    mode: str = "w",
    timeout: int | None = None,
) -> None:
    """
    Execute an evaluation command with optional timeout.
    
    Args:
        cmd: Command as list
        env: Environment variables
        log_file: Log file path
        mode: File mode ('w' or 'a')
        timeout: Timeout in seconds (None = no timeout)
    """
    with open(log_file, mode) as fh:
        try:
            if timeout and timeout > 0:
                subprocess.run(
                    cmd, 
                    check=True, 
                    env=env, 
                    stdout=fh, 
                    stderr=subprocess.STDOUT,
                    timeout=timeout
                )
            else:
                subprocess.run(
                    cmd, 
                    check=True, 
                    env=env, 
                    stdout=fh, 
                    stderr=subprocess.STDOUT
                )
        except subprocess.TimeoutExpired:
            # Write timeout message to log
            fh.write(f"\n\n[ERROR] Evaluation timeout after {timeout} seconds!\n")
            fh.write("Process was aborted.\n")
            raise


def sanitize_summary_file(summary_file: Path):
    """
    Replaces NaN tokens with null so the JSON is standards-compliant.
    """
    if not summary_file.exists():
        return
    text = summary_file.read_text()
    if "NaN" not in text:
        return
    summary_file.write_text(text.replace("NaN", "null"))


def with_device(cmd: List[str], device: str | None) -> List[str]:
    """
    Returns a copy of cmd with --device <value> replaced (or appended) by `device`.
    """
    new_cmd: List[str] = []
    skip = False
    for token in cmd:
        if skip:
            skip = False
            continue
        if token == "--device":
            skip = True
            continue
        new_cmd.append(token)
    if device:
        new_cmd.extend(["--device", device])
    return new_cmd


def train_trial(
    trial_dir: Path,
    dataset_id: int,
    dataset_name: str,
    folds: List[str],
    configuration: str,
    trainer: str,
    plans_name: str,
    device: str | None,
    env: dict,
):
    for fold in folds:
        cleanup_fold_results(
            dataset_name,
            trainer,
            plans_name,
            configuration,
            str(fold),
        )
        print(f"[{trial_dir.name}/fold{fold}] Starting training...")
        train_cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "-tr",
            trainer,
            "-p",
            plans_name,
        ]
        if device:
            train_cmd.extend(["--device", device])

        log_dir = TRAINING_OUTPUT_ROOT / trial_dir.name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        gpu_log = log_dir / f"gpu_{trial_dir.name}_fold{fold}.csv"
        print(f"[{trial_dir.name}/fold{fold}] GPU log -> {gpu_log}")

        proc, fh = start_gpu_logger(gpu_log)
        try:
            run_cmd(train_cmd, env, f"{trial_dir.name}/fold{fold}/train")
        finally:
            stop_gpu_logger(proc, fh)
        print(f"[{trial_dir.name}/fold{fold}] Training completed.")


def evaluate_trial(
    trial_dir: Path,
    dataset_name: str,
    folds: List[str],
    configuration: str,
    trainer: str,
    plans_name: str,
    env: dict,
    eval_base: Path,
    timeout: int | None = None,
):
    """
    Evaluates each requested fold by comparing the validation predictions against the
    corresponding ground-truth labels using nnUNetv2_evaluate_folder.
    """
    gt_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "labelsTr"
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth labels not found: {gt_dir}. "
            "Please ensure nnUNet_raw is set correctly."
        )

    dataset_json = trial_dir / dataset_name / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"dataset.json for {trial_dir.name} missing at {dataset_json}."
        )

    plans_file = trial_dir / dataset_name / f"{plans_name}.json"
    if not plans_file.exists():
        raise FileNotFoundError(
            f"{plans_name}.json for {trial_dir.name} missing at {plans_file}."
        )

    results_root = Path(env["nnUNet_results"]).resolve()
    pred_root = (
        results_root
        / dataset_name
        / f"{trainer}__{plans_name}__{configuration}"
    )

    timeout_str = f" (Timeout: {timeout}s)" if timeout and timeout > 0 else ""
    log_file = eval_base / "evaluation.log"

    for idx, fold in enumerate(folds):
        pred_dir = pred_root / f"fold_{fold}" / "validation"
        if not pred_dir.exists():
            print(
                f"[WARN] Prediction directory for {trial_dir.name} Fold {fold} missing ({pred_dir}). "
                "Skipping evaluation for this fold."
            )
            continue

        summary_file = eval_base / f"fold_{fold}_summary.json"
        eval_cmd = [
            "nnUNetv2_evaluate_folder",
            str(gt_dir),
            str(pred_dir),
            "-djfile",
            str(dataset_json),
            "-pfile",
            str(plans_file),
            "-o",
            str(summary_file),
            "--chill",
        ]

        print(
            f"[{trial_dir.name}/evaluate] -> {' '.join(eval_cmd)}{timeout_str} "
            f"(log: {log_file})"
        )

        mode = "w" if idx == 0 else "a"
        run_eval_cmd(eval_cmd, env, log_file, mode=mode, timeout=timeout)
        sanitize_summary_file(summary_file)



def main():
    args = parse_args()
    ensure_env_vars()
    dataset_id = extract_dataset_id(args.dataset_name)

    trial_base_dir = (
        Path("hpo") / "preprocessing_output" / args.dataset_name
    ).resolve()
    all_trials = list_trial_dirs(trial_base_dir)
    if not all_trials:
        print(f"No trials found under {trial_base_dir}.")
        return

    selected_trials = filter_trials(all_trials, args.trials)
    print(
        f"Found trials: {[t.name for t in selected_trials]} "
        f"(Dataset ID: {dataset_id}, Trainer: {args.trainer}, Config: {args.configuration})"
    )
    folds = [str(fold) for fold in args.folds]

    TRAINING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_trials = skip_completed_trials(
        selected_trials, args.dataset_name, args.only_evaluate
    )
    if not selected_trials:
        print(
            "No new trials to train. Remove training_output/trial_X if you want "
            "to restart a trial."
        )
        return

    failed_trials: List[str] = []

    for trial_dir in selected_trials:
        print(f"\n===== Starting Trial {trial_dir.name} =====")
        dataset_stage, used_symlink = stage_preprocessed_dataset(
            trial_dir, args.dataset_name
        )
        eval_base = prepare_eval_directory(
            args.dataset_name, trial_dir.name, args.configuration, args.trainer
        )
        env = os.environ.copy()
        pre_archived_dir = get_archived_results_dir(trial_dir.name, args.dataset_name)
        if args.only_evaluate:
            if not pre_archived_dir.exists():
                print(
                    f"[WARN] Trial {trial_dir.name} has no archived results "
                    "under training_output/. Skipping."
                )
                cleanup_staged_dataset(dataset_stage, used_symlink)
                continue
            env["nnUNet_results"] = str(pre_archived_dir.parent)
        archived_results_dir: Path | None = None
        try:
            if not args.only_evaluate:
                train_trial(
                    trial_dir,
                    dataset_id,
                    args.dataset_name,
                    folds,
                    args.configuration,
                    args.trainer,
                    args.plans_name,
                    args.device,
                    env,
                )
                archived_results_dir = persist_training_results(
                    args.dataset_name, trial_dir.name
                )
                if archived_results_dir:
                    print(
                        f"[{trial_dir.name}] Results archived at {archived_results_dir}."
                    )
                    env["nnUNet_results"] = str(archived_results_dir.parent)
                else:
                    print(
                        f"[WARN] Could not archive results for {trial_dir.name}."
                    )
            if not args.skip_evaluation:
                evaluate_trial(
                    trial_dir,
                    args.dataset_name,
                    folds,
                    args.configuration,
                    args.trainer,
                    args.plans_name,
                    env,
                    eval_base,
                    timeout=args.eval_timeout if args.eval_timeout > 0 else None,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                error_msg = f"Timeout after {args.eval_timeout}s"
            else:
                error_msg = f"Exit code {exc.returncode}"
            print(
                f"[WARN] Trial {trial_dir.name} failed ({error_msg})."
            )
            failed_trials.append(trial_dir.name)
            if args.stop_on_error:
                raise
            else:
                print("Continuing with next trial...")
        finally:
            cleanup_staged_dataset(dataset_stage, used_symlink)

    print("\nAll requested trials have been processed.")
    if failed_trials:
        print(f"Failed trials ({len(failed_trials)}): {failed_trials}")
    else:
        print("All trials completed successfully.")


if __name__ == "__main__":
    main()

