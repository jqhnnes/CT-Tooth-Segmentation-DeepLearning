#!/usr/bin/env python3
"""
Standardize trial output structure: keep only validation, remove test-set outputs.

After running this script, Dice scores are read exclusively from
`fold_0/validation/summary.json` (dice_validation), which is consistent
across all trials.

Removes per trial:
  - Per fold: labelsTs/, labelsTs_tta/, labelsTs_tta_pp/
  - Per fold: labelsTs_tta_pp_summary.json (if present)
  - Per config: crossval_results_folds_0/
  - Per config: tta_postprocess.log
  - hpo/analysis/trial_*_labelsTs_summary.json
  - hpo/analysis/trial_*_labelsTs_tta_pp_summary.json

Resulting uniform structure per trial:
  fold_0/
    validation/           # retained — Dice from validation/summary.json
    checkpoint_best.pth, training_log_*.txt, ...
    (no labelsTs*, no crossval_results_folds_0, no test-set summaries)

After cleaning, re-run:
  python hpo/scripts/analysis/summarize_trials.py

Example:
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py --dry_run
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py --show_structure
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


# Directories and files removed per fold to enforce uniform structure (validation only)
PRED_DIRS_TO_REMOVE = ("labelsTs", "labelsTs_tta", "labelsTs_tta_pp")
FOLD_FILES_TO_REMOVE = ("labelsTs_tta_pp_summary.json",)
CONFIG_FILES_TO_REMOVE = ("tta_postprocess.log",)
ANALYSIS_GLOBS_TO_REMOVE = ("trial_*_labelsTs_summary.json", "trial_*_labelsTs_tta_pp_summary.json")


def _print_structure() -> None:
    """Print the target per-trial directory structure after cleaning."""
    print("Uniform structure per trial (Dice from validation only):\n")
    print("  hpo/training_output/trial_<N>/")
    print("    nnUNet_results/<Dataset>/<Trainer>__<Plans>__<Config>/")
    print("      fold_0/")
    print("        validation/           # retained — Dice from validation/summary.json")
    print("        checkpoint_best.pth, training_log_*.txt, ...")
    print("        (no labelsTs/, labelsTs_tta/, labelsTs_tta_pp/)")
    print("      (no crossval_results_folds_0/, no tta_postprocess.log)")
    print("")
    print("  hpo/analysis/")
    print("    (no trial_*_labelsTs*.json)")
    print("")
    print("Removed: labelsTs*, crossval_results_folds_0, tta_postprocess.log, trial_*_labelsTs*.json.")


def main():
    """Parse arguments and run the cleanup."""
    parser = argparse.ArgumentParser(
        description="Standardize trial structure: keep only validation, remove test-set outputs."
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print what would be deleted.")
    parser.add_argument("--show_structure", action="store_true",
                        help="Print target per-trial structure and exit.")
    args = parser.parse_args()

    root = Path("hpo")
    if args.show_structure:
        _print_structure()
        return
    training_root = root / "training_output"
    analysis_root = root / "analysis"

    if not training_root.exists():
        print(f"{training_root} not found.")
        return

    removed_dirs = 0
    removed_files = 0

    trial_dirs = [d for d in training_root.iterdir() if d.is_dir() and re.match(r"trial_\d+$", d.name)]
    trial_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
    for trial_dir in trial_dirs:
        results = trial_dir / "nnUNet_results"
        if not results.exists():
            continue
        for dataset_dir in results.iterdir():
            if not dataset_dir.is_dir():
                continue
            for config_dir in dataset_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                crossval_dir = config_dir / "crossval_results_folds_0"
                if crossval_dir.exists():
                    if args.dry_run:
                        print(f"[DRY] would remove: {crossval_dir}")
                    else:
                        shutil.rmtree(crossval_dir)
                        print(f"Removed: {crossval_dir}")
                    removed_dirs += 1
                for file_name in CONFIG_FILES_TO_REMOVE:
                    f = config_dir / file_name
                    if f.exists():
                        if args.dry_run:
                            print(f"[DRY] would remove: {f}")
                        else:
                            f.unlink()
                            print(f"Removed: {f}")
                        removed_files += 1
                for fold_dir in config_dir.iterdir():
                    if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                        continue
                    for dir_name in PRED_DIRS_TO_REMOVE:
                        d = fold_dir / dir_name
                        if d.exists():
                            if args.dry_run:
                                print(f"[DRY] would remove: {d}")
                            else:
                                shutil.rmtree(d)
                                print(f"Removed: {d}")
                            removed_dirs += 1
                    for file_name in FOLD_FILES_TO_REMOVE:
                        f = fold_dir / file_name
                        if f.exists():
                            if args.dry_run:
                                print(f"[DRY] would remove: {f}")
                            else:
                                f.unlink()
                                print(f"Removed: {f}")
                            removed_files += 1

    for pattern in ANALYSIS_GLOBS_TO_REMOVE:
        for summary in analysis_root.glob(pattern):
            if args.dry_run:
                print(f"[DRY] would remove: {summary}")
            else:
                summary.unlink()
                print(f"Removed: {summary}")
            removed_files += 1

    print(f"\nSummary: {removed_dirs} dir(s), {removed_files} file(s) removed.")
    if args.dry_run:
        print("Run without --dry_run to actually delete.")
    if not args.dry_run and (removed_dirs + removed_files) > 0:
        print("\nDice is now sourced exclusively from validation. Next steps:")
        print("  python hpo/scripts/analysis/summarize_trials.py")
        print("  python scripts/analysis/training/analyze_trial_parameters.py")


if __name__ == "__main__":
    main()
