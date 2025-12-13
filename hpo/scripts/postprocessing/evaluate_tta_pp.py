#!/usr/bin/env python3
"""
Evaluate all trials that have postprocessed TTA predictions (labelsTs_tta_pp).

For each trial it checks:
  - pred_dir: hpo/training_output/<trial>/nnUNet_results/.../fold_<f>/labelsTs_tta_pp
  - dataset.json and nnUNetPlans.json in the corresponding preprocessing trial
  - runs nnUNetv2_evaluate_folder against labelsTs

Outputs summaries to:
  hpo/analysis/<trial>_labelsTs_tta_pp_summary.json

Usage:
    python hpo/scripts/postprocessing/evaluate_tta_pp.py
    python hpo/scripts/postprocessing/evaluate_tta_pp.py --trials trial_0 trial_12 --folds 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import subprocess
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate labelsTs_tta_pp for trials.")
    p.add_argument("--dataset_name", default="Dataset001_GroundTruth")
    p.add_argument("--trials", nargs="*", help="Specific trials (e.g., trial_0 trial_12). Default: all.")
    p.add_argument("--folds", nargs="+", default=["0"], help="Folds to evaluate (default: 0).")
    p.add_argument("--trainer", default="nnUNetTrainer")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--plans_name", default="nnUNetPlans")
    p.add_argument(
        "--labels_ts",
        default="/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_raw/Dataset001_GroundTruth/labelsTs",
        help="Path to labelsTs ground truth.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if summary already exists.",
    )
    return p.parse_args()


def list_trials(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("trial_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[1]))


def main():
    args = parse_args()
    trial_root = Path("hpo") / "training_output"
    analysis_root = Path("hpo") / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    trials = list_trials(trial_root)
    if args.trials:
        sel = set(args.trials)
        trials = [t for t in trials if t.name in sel]

    evaluated = []

    for trial_dir in trials:
        model_root = (
            trial_dir
            / "nnUNet_results"
            / args.dataset_name
            / f"{args.trainer}__{args.plans_name}__{args.configuration}"
        )
        if not model_root.exists():
            print(f"[WARN] Model dir missing: {model_root}, skip.")
            continue

        preproc_trial = Path("hpo") / "preprocessing_output" / args.dataset_name / trial_dir.name / args.dataset_name
        dataset_json = preproc_trial / "dataset.json"
        plans_json = preproc_trial / f"{args.plans_name}.json"
        if not dataset_json.exists() or not plans_json.exists():
            print(f"[WARN] Missing dataset/plans for {trial_dir.name}, skip.")
            continue

        for fold in args.folds:
            pred_dir = model_root / f"fold_{fold}" / "labelsTs_tta_pp"
            if not pred_dir.exists():
                print(f"[WARN] No labelsTs_tta_pp for {trial_dir.name} fold {fold}, skip.")
                continue
            summary_out = analysis_root / f"{trial_dir.name}_labelsTs_tta_pp_summary.json"
            if summary_out.exists() and not args.force:
                print(f"[INFO] Summary exists for {trial_dir.name}, skip (use --force to re-evaluate).")
                continue
            cmd = [
                "nnUNetv2_evaluate_folder",
                args.labels_ts,
                str(pred_dir),
                "-djfile",
                str(dataset_json),
                "-pfile",
                str(plans_json),
                "-o",
                str(summary_out),
                "--chill",
            ]
            print(f"[{trial_dir.name}] eval fold {fold} -> {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                # collect for ranking
                try:
                    d = json.load(open(summary_out))
                    dice = d.get("foreground_mean", {}).get("Dice")
                    if dice is not None:
                        evaluated.append((trial_dir.name, dice, summary_out))
                except Exception:
                    pass
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Evaluation failed for {trial_dir.name} fold {fold}: {e}")

    if evaluated:
        print("\nRanking (labelsTs_tta_pp, foreground_mean Dice):")
        for t, d, p in sorted(evaluated, key=lambda x: x[1], reverse=True):
            print(f"{t}: {d:.4f}  ({p})")
    else:
        # No new evals; try to rank existing summaries anyway
        existing = []
        for p in analysis_root.glob("trial_*_labelsTs_tta_pp_summary.json"):
            try:
                d = json.load(open(p))
                dice = d.get("foreground_mean", {}).get("Dice")
                if dice is not None:
                    parts = p.name.split("_")
                    trial_name = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else p.name
                    existing.append((trial_name, dice, p))
            except Exception:
                continue
        if existing:
            print("\nRanking from existing labelsTs_tta_pp summaries:")
            for t, d, p in sorted(existing, key=lambda x: x[1], reverse=True):
                # try to load baseline (non-pp) and TTA-only if present
                baseline_path = analysis_root / f"{t}_labelsTs_summary.json"
                tta_path = analysis_root / f"{t}_labelsTs_tta_summary.json"
                delta_base = ""
                delta_tta = ""
                if baseline_path.exists():
                    try:
                        b = json.load(open(baseline_path)).get("foreground_mean", {}).get("Dice")
                        if b is not None:
                            delta = d - b
                            delta_base = f"  Δvs labelsTs: {delta:+.4f}"
                    except Exception:
                        pass
                if tta_path.exists():
                    try:
                        b = json.load(open(tta_path)).get("foreground_mean", {}).get("Dice")
                        if b is not None:
                            delta = d - b
                            delta_tta = f"  Δvs labelsTs_tta: {delta:+.4f}"
                    except Exception:
                        pass
                print(f"{t}: {d:.4f}  ({p}){delta_base}{delta_tta}")
        else:
            print("\nNo new summaries evaluated or found.")


if __name__ == "__main__":
    main()


