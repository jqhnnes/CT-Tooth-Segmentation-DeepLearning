#!/usr/bin/env python3
"""
Run nnUNetv2 postprocessing for archived HPO trials.

For each trial under hpo/training_output/<trial>/nnUNet_results/... this script:
 1) (optional) runs nnUNetv2_find_best_configuration to create postprocessing.pkl
 2) applies nnUNetv2_apply_postprocessing to predictions (default: validation)

Example:
    python hpo/scripts/training/run_postprocessing.py \
        --trials trial_12 trial_13 \
        --folds 0 \
        --pred_subdir labelsTs_tta \
        --suffix pp
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_dataset_id(dataset_name: str) -> int:
    match = re.search(r"Dataset(\d+)", dataset_name)
    if not match:
        raise ValueError(
            f"Could not extract dataset ID from '{dataset_name}'. Expected 'DatasetXXX_...'."
        )
    return int(match.group(1))


def list_trial_dirs(base_dir: Path) -> List[Path]:
    trial_dirs: List[Path] = []
    if not base_dir.exists():
        return trial_dirs
    for entry in base_dir.iterdir():
        if entry.is_dir() and re.match(r"trial_\d+$", entry.name):
            trial_dirs.append(entry)
    trial_dirs.sort(key=lambda p: int(p.name.split("_")[1]))
    return trial_dirs


def run_cmd(cmd: List[str], env: dict, log_file: Path, mode: str = "a"):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, mode) as fh:
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply nnUNetv2 postprocessing to HPO trials.")
    parser.add_argument(
        "--dataset_name",
        default="Dataset001_GroundTruth",
        help="Dataset name (matches nnUNet_raw folder).",
    )
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Trials to process (e.g., trial_12 trial_15). Default: all under training_output.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds to postprocess (e.g., 0 1 2 3 4).",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Trainer class name (default: nnUNetTrainer).",
    )
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnUNet configuration (default: 3d_fullres).",
    )
    parser.add_argument(
        "--plans_name",
        default="nnUNetPlans",
        help="Plans name without .json (default: nnUNetPlans).",
    )
    parser.add_argument(
        "--pred_subdir",
        default="validation",
        help="Prediction subdir inside fold_X (e.g., validation, labelsTs, labelsTs_tta).",
    )
    parser.add_argument(
        "--suffix",
        default="pp",
        help="Suffix for output directory (default: pp).",
    )
    parser.add_argument(
        "--skip_find",
        action="store_true",
        help="Skip running nnUNetv2_find_best_configuration if postprocessing.pkl exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_id = extract_dataset_id(args.dataset_name)

    trial_root = Path("hpo") / "training_output"
    trials = list_trial_dirs(trial_root)
    if args.trials:
        sel = set(args.trials)
        trials = [t for t in trials if t.name in sel]
    if not trials:
        print("No trials to process.")
        return

    for trial_dir in trials:
        print(f"\n=== Postprocessing {trial_dir.name} ===")
        model_root = (
            trial_dir
            / "nnUNet_results"
            / args.dataset_name
            / f"{args.trainer}__{args.plans_name}__{args.configuration}"
        )
        if not model_root.exists():
            print(f"[WARN] Model dir missing: {model_root}, skipping.")
            continue

        # env per trial
        env = os.environ.copy()
        env["nnUNet_results"] = str(trial_dir / "nnUNet_results")
        env["nnUNet_preprocessed"] = str(
            Path("hpo")
            / "preprocessing_output"
            / args.dataset_name
            / trial_dir.name
        )

        pp_pkl = model_root / "postprocessing.pkl"
        log_file = model_root / "postprocessing.log"

        if not args.skip_find or not pp_pkl.exists():
            cmd_find = [
                "nnUNetv2_find_best_configuration",
                str(dataset_id),
                "-tr",
                args.trainer,
                "-c",
                args.configuration,
                "-p",
                args.plans_name,
            ]
            print(f"[{trial_dir.name}] Finding PP config -> {' '.join(cmd_find)}")
            try:
                run_cmd(cmd_find, env, log_file, mode="w")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] find_best_configuration failed for {trial_dir.name}: {e}")

        if not pp_pkl.exists():
            print(f"[WARN] postprocessing.pkl missing for {trial_dir.name}, skipping apply.")
            continue

        for fold in args.folds:
            in_dir = model_root / f"fold_{fold}" / args.pred_subdir
            if not in_dir.exists():
                print(f"[WARN] Predictions missing at {in_dir}, skipping fold {fold}.")
                continue
            out_dir = model_root / f"fold_{fold}" / f"{args.pred_subdir}_{args.suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd_apply = [
                "nnUNetv2_apply_postprocessing",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "-pp_pkl_file",
                str(pp_pkl),
            ]
            print(f"[{trial_dir.name}] Apply PP fold {fold} -> {' '.join(cmd_apply)}")
            try:
                run_cmd(cmd_apply, env, log_file, mode="a")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] apply_postprocessing failed for {trial_dir.name} fold {fold}: {e}")


if __name__ == "__main__":
    main()


