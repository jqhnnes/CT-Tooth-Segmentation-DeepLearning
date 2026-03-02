#!/usr/bin/env python3
"""
Summarize all trials: parameters + metrics (validation, labelsTs, labelsTs_tta_pp).

Outputs:
  hpo/analysis/trials_summary.json

Validation-Dice: Aus fold_0/validation/summary.json (foreground_mean.Dice).
Reicht für vergleichbare Dice-Kurve über Trials – jeder Trial braucht nur den validation-Ordner.

Each entry:
{
  "trial": "trial_XX",
  "spacing": [...],
  "dice_validation": float | null,   # aus validation/summary.json, einheitlich für alle
  "dice_labelsTs": float | null,
  "dice_labelsTs_tta_pp": float | null,
  ...
}

Usage:
  python hpo/scripts/analysis/summarize_trials.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def main():
    root = Path("hpo")
    training_root = root / "training_output"
    analysis_root = root / "analysis"

    results: List[Dict[str, Any]] = []

    for trial_dir in sorted(training_root.glob("trial_*"), key=lambda p: int(p.name.split("_")[1])):
        plan_path = trial_dir / "nnUNet_results" / "Dataset001_GroundTruth" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "plans.json"
        if not plan_path.exists():
            continue
        plan = load_json(plan_path)
        if not plan:
            continue
        cfg = plan.get("configurations", {}).get("3d_fullres", {})
        arch = cfg.get("architecture", {}).get("arch_kwargs", {})
        spacing = cfg.get("spacing")
        patch = cfg.get("patch_size")
        batch_size = cfg.get("batch_size")
        features_per_stage = arch.get("features_per_stage", [])
        features_base = features_per_stage[0] if features_per_stage else None

        tname = trial_dir.name
        # Validation-Dice: aus fold_0/validation/summary.json – reicht für alle Trials (einheitlich)
        model_base = trial_dir / "nnUNet_results" / "Dataset001_GroundTruth" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        val_summary = model_base / "fold_0" / "validation" / "summary.json"
        dice_validation = None
        if val_summary.exists():
            d = load_json(val_summary)
            if d:
                dice_validation = d.get("foreground_mean", {}).get("Dice")

        lbl_summary_path = analysis_root / f"{tname}_labelsTs_summary.json"
        tta_pp_summary_path = analysis_root / f"{tname}_labelsTs_tta_pp_summary.json"
        dice_lbl = None
        dice_tta_pp = None
        if lbl_summary_path.exists():
            d = load_json(lbl_summary_path)
            if d:
                dice_lbl = d.get("foreground_mean", {}).get("Dice")
        if tta_pp_summary_path.exists():
            d = load_json(tta_pp_summary_path)
            if d:
                dice_tta_pp = d.get("foreground_mean", {}).get("Dice")

        results.append(
            {
                "trial": tname,
                "spacing": spacing,
                "patch_size": patch,
                "batch_size": batch_size,
                "features_base": features_base,
                "dice_validation": dice_validation,
                "dice_labelsTs": dice_lbl,
                "dice_labelsTs_tta_pp": dice_tta_pp,
                "summary_labelsTs": str(lbl_summary_path) if lbl_summary_path.exists() else None,
                "summary_labelsTs_tta_pp": str(tta_pp_summary_path) if tta_pp_summary_path.exists() else None,
            }
        )

    out_path = analysis_root / "trials_summary.json"
    analysis_root.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path} with {len(results)} entries.")


if __name__ == "__main__":
    main()

