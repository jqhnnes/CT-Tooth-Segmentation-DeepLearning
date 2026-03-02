#!/usr/bin/env python3
"""
Einheitliche Trial-Struktur: Nur Validation behalten, Rest löschen.

Dice kommt dann nur aus validation/summary.json (dice_validation) – reicht für alle Trials.
Es werden alle Test-Set-Ausgaben und zugehörigen Summaries entfernt.

Gelöscht:
  - In jedem fold: labelsTs/, labelsTs_tta/, labelsTs_tta_pp/
  - In jedem fold: labelsTs_tta_pp_summary.json (falls vorhanden)
  - crossval_results_folds_0/ (pro Config, pro Trial)
  - TTA-Logs: tta_postprocess.log (pro Config)
  - hpo/analysis/trial_*_labelsTs_summary.json
  - hpo/analysis/trial_*_labelsTs_tta_pp_summary.json

Einheitliche Struktur danach (pro Trial gleich):
  fold_0/
    validation/           # bleibt – hierher kommt der Dice (foreground_mean in summary.json)
    checkpoint_best.pth, training_log_*.txt, ...
  (keine labelsTs*, kein crossval_results_folds_0, keine Test-Set-Summaries in hpo/analysis)

Anschließend: python hpo/scripts/analysis/summarize_trials.py
  → liest dice_validation aus fold_0/validation/summary.json, schreibt trials_summary.json.

Beispiel:
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py --dry_run
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py
  python hpo/scripts/postprocessing/clean_postprocessed_outputs.py --show_structure
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


# Ordner/Dateien pro fold, die gelöscht werden (einheitliche Struktur = nur validation)
PRED_DIRS_TO_REMOVE = ("labelsTs", "labelsTs_tta", "labelsTs_tta_pp")
FOLD_FILES_TO_REMOVE = ("labelsTs_tta_pp_summary.json",)
CONFIG_FILES_TO_REMOVE = ("tta_postprocess.log",)  # TTA+PP-Log pro Config
ANALYSIS_GLOBS_TO_REMOVE = ("trial_*_labelsTs_summary.json", "trial_*_labelsTs_tta_pp_summary.json")


def _print_structure(root: Path) -> None:
    """Zielstruktur pro Trial: nur Validation für Dice."""
    print("Einheitliche Struktur pro Trial (Dice nur aus Validation):\n")
    print("  hpo/training_output/trial_<N>/")
    print("    nnUNet_results/<Dataset>/<Trainer>__<Plans>__<Config>/")
    print("      fold_0/")
    print("        validation/           # bleibt – Dice aus validation/summary.json")
    print("        checkpoint_best.pth, training_log_*.txt, ...")
    print("        (keine labelsTs/, labelsTs_tta/, labelsTs_tta_pp/)")
    print("      (kein crossval_results_folds_0/, kein tta_postprocess.log)")
    print("")
    print("  hpo/analysis/")
    print("    (keine trial_*_labelsTs*.json)")
    print("")
    print("Gelöscht wird: labelsTs*, crossval_results_folds_0, tta_postprocess.log, trial_*_labelsTs*.json.")


def main():
    p = argparse.ArgumentParser(description="Einheitliche Struktur: nur Validation behalten, Test-Ausgaben löschen.")
    p.add_argument("--dry_run", action="store_true", help="Only print what would be deleted.")
    p.add_argument("--show_structure", action="store_true", help="Print target per-trial structure and exit.")
    args = p.parse_args()

    root = Path("hpo")
    if args.show_structure:
        _print_structure(root)
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
        print("\nDice kommt nur noch aus validation. Als nächstes:")
        print("  python hpo/scripts/analysis/summarize_trials.py")
        print("  python scripts/analysis/analyze_trial_parameters.py")


if __name__ == "__main__":
    main()
