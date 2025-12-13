#!/usr/bin/env python3
"""
Run TTA predictions and postprocessing for archived HPO trials.

Pipeline per trial:
 1) nnUNetv2_predict (TTA is on by default) -> <pred_subdir> (default: labelsTs_tta)
 2) nnUNetv2_find_best_configuration (if postprocessing.pkl is missing)
 3) nnUNetv2_apply_postprocessing -> <pred_subdir>_<suffix> (default: labelsTs_tta_pp)
 4) (optional) nnUNetv2_evaluate_folder if --eval_labels is provided

Example:
    python hpo/scripts/training/run_tta_and_postprocessing.py \
        --trials trial_12 trial_13 \
        --folds 0 \
        --input_dir data/nnUNet_raw/Dataset001_GroundTruth/imagesTs \
        --eval_labels data/nnUNet_raw/Dataset001_GroundTruth/labelsTs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_dataset_id(dataset_name: str) -> int:
    m = re.search(r"Dataset(\d+)", dataset_name)
    if not m:
        raise ValueError(f"Cannot extract dataset id from '{dataset_name}'")
    return int(m.group(1))


def list_trial_dirs(base_dir: Path) -> List[Path]:
    trials: List[Path] = []
    if not base_dir.exists():
        return trials
    for entry in base_dir.iterdir():
        if entry.is_dir() and re.match(r"trial_\d+$", entry.name):
            trials.append(entry)
    trials.sort(key=lambda p: int(p.name.split("_")[1]))
    return trials


def run_cmd(cmd: List[str], env: dict, log_file: Path, mode: str = "a"):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, mode) as fh:
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TTA predict + postprocessing for HPO trials.")
    p.add_argument("--dataset_name", default="Dataset001_GroundTruth")
    p.add_argument("--trials", nargs="*", help="trial_ ids; default: all under training_output")
    p.add_argument("--folds", nargs="+", default=["0"])
    p.add_argument("--trainer", default="nnUNetTrainer")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--plans_name", default="nnUNetPlans")
    p.add_argument("--input_dir", required=True, help="Input images folder (e.g. imagesTs)")
    p.add_argument("--pred_subdir", default="labelsTs_tta")
    p.add_argument("--suffix", default="pp")
    p.add_argument("--skip_predict", action="store_true")
    p.add_argument("--skip_find", action="store_true")
    p.add_argument("--eval_labels", help="Optional GT folder to evaluate after PP")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_id = extract_dataset_id(args.dataset_name)

    trial_root = Path("hpo") / "training_output"
    trials = list_trial_dirs(trial_root)
    if args.trials:
        wanted = set(args.trials)
        trials = [t for t in trials if t.name in wanted]
    if not trials:
        print("No trials to process.")
        return

    for trial_dir in trials:
        print(f"\n=== Trial {trial_dir.name} ===")
        model_root = (
            trial_dir
            / "nnUNet_results"
            / args.dataset_name
            / f"{args.trainer}__{args.plans_name}__{args.configuration}"
        )
        if not model_root.exists():
            print(f"[WARN] model dir missing: {model_root}, skipping.")
            continue

        env = os.environ.copy()
        env["nnUNet_results"] = str(trial_dir / "nnUNet_results")
        env["nnUNet_preprocessed"] = str(
            Path("hpo")
            / "preprocessing_output"
            / args.dataset_name
            / trial_dir.name
        )

        log_file = model_root / "tta_postprocess.log"

        # 1) predict with TTA (default) if not skipped
        for fold in args.folds:
            out_dir = model_root / f"fold_{fold}" / args.pred_subdir
            if not args.skip_predict:
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd_pred = [
                    "nnUNetv2_predict",
                    "-i",
                    args.input_dir,
                    "-o",
                    str(out_dir),
                    "-d",
                    str(dataset_id),
                    "-c",
                    args.configuration,
                    "-tr",
                    args.trainer,
                    "-p",
                    args.plans_name,
                    "-f",
                    str(fold),
                ]
                print(f"[{trial_dir.name}] predict fold {fold} -> {' '.join(cmd_pred)}")
                try:
                    run_cmd(cmd_pred, env, log_file, mode="w" if fold == args.folds[0] else "a")
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] prediction failed for {trial_dir.name} fold {fold}: {e}")

        # 2) find best PP once (respect single-fold runs)
        def find_pp_candidate() -> Path | None:
            candidates = [
                model_root / "postprocessing.pkl",
                model_root / "crossval_results_folds_0" / "postprocessing.pkl",
            ]
            for c in candidates:
                if c.exists():
                    return c
            return None

        pp_pkl = find_pp_candidate()

        if (not args.skip_find) and pp_pkl is None:
            cmd_find = [
                "nnUNetv2_find_best_configuration",
                str(dataset_id),
                "-tr",
                args.trainer,
                "-c",
                args.configuration,
                "-p",
                args.plans_name,
                "-f",
                *args.folds,
            ]
            print(f"[{trial_dir.name}] find PP -> {' '.join(cmd_find)}")
            try:
                run_cmd(cmd_find, env, log_file, mode="a")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] find_best_configuration failed for {trial_dir.name}: {e}")
            pp_pkl = find_pp_candidate()

        if pp_pkl is None:
            print(f"[WARN] postprocessing.pkl missing for {trial_dir.name}, skip apply.")
            continue

        # 3) apply PP
        for fold in args.folds:
            in_dir = model_root / f"fold_{fold}" / args.pred_subdir
            if not in_dir.exists():
                print(f"[WARN] predictions missing at {in_dir}, skip fold {fold}.")
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
            print(f"[{trial_dir.name}] apply PP fold {fold} -> {' '.join(cmd_apply)}")
            try:
                run_cmd(cmd_apply, env, log_file, mode="a")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] apply_postprocessing failed for {trial_dir.name} fold {fold}: {e}")

        # 4) optional evaluation
        if args.eval_labels:
            dj = Path("hpo") / "preprocessing_output" / args.dataset_name / trial_dir.name / args.dataset_name / "dataset.json"
            pj = Path("hpo") / "preprocessing_output" / args.dataset_name / trial_dir.name / args.dataset_name / f"{args.plans_name}.json"
            if not dj.exists() or not pj.exists():
                print(f"[WARN] dataset.json or plans.json missing for {trial_dir.name}, skip eval.")
            else:
                for fold in args.folds:
                    out_dir = model_root / f"fold_{fold}" / f"{args.pred_subdir}_{args.suffix}"
                    if not out_dir.exists():
                        continue
                    summary = model_root / f"fold_{fold}" / f"{args.pred_subdir}_{args.suffix}_summary.json"
                    cmd_eval = [
                        "nnUNetv2_evaluate_folder",
                        args.eval_labels,
                        str(out_dir),
                        "-djfile",
                        str(dj),
                        "-pfile",
                        str(pj),
                        "-o",
                        str(summary),
                        "--chill",
                    ]
                    print(f"[{trial_dir.name}] eval fold {fold} -> {' '.join(cmd_eval)}")
                    try:
                        run_cmd(cmd_eval, env, log_file, mode="a")
                    except subprocess.CalledProcessError as e:
                        print(f"[WARN] evaluate failed for {trial_dir.name} fold {fold}: {e}")


if __name__ == "__main__":
    main()


