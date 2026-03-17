#!/usr/bin/env python3
"""
Aggregate trial hyperparameters and Dice metrics into a single JSON summary.

Reads each trial's ``plans.json`` for architecture/spacing parameters and
``fold_0/validation/summary.json`` for the validation Dice score.
Optionally includes Dice from ``labelsTs`` and ``labelsTs_tta_pp`` summaries
if they exist under ``hpo/analysis/``.

Output: ``hpo/analysis/trials_summary.json``

Each entry has the structure::

    {
        "trial": "trial_XX",
        "spacing": [...],
        "patch_size": [...],
        "batch_size": int,
        "features_base": int,
        "dice_validation": float | null,
        "dice_labelsTs": float | null,
        "dice_labelsTs_tta_pp": float | null,
    }

Usage:
    python hpo/scripts/analysis/summarize_trials.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` on any error.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object, or ``None`` if the file is missing or malformed.
    """
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    """Collect per-trial parameters and metrics, write trials_summary.json."""
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

        trial_name = trial_dir.name
        model_base = trial_dir / "nnUNet_results" / "Dataset001_GroundTruth" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        val_summary = model_base / "fold_0" / "validation" / "summary.json"
        dice_validation = None
        if val_summary.exists():
            d = load_json(val_summary)
            if d:
                dice_validation = d.get("foreground_mean", {}).get("Dice")

        lbl_summary_path = analysis_root / f"{trial_name}_labelsTs_summary.json"
        tta_pp_summary_path = analysis_root / f"{trial_name}_labelsTs_tta_pp_summary.json"
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
                "trial": trial_name,
                "spacing": spacing,
                "patch_size": patch,
                "batch_size": batch_size,
                "features_base": features_base,
                "dice_validation": dice_validation,
                "dice_labelsTs": dice_lbl,
                "dice_labelsTs_tta_pp": dice_tta_pp,
                "summary_labelsTs": str(lbl_summary_path) if lbl_summary_path.exists() else None,
                "summary_labelsTs_tta_pp": (
                    str(tta_pp_summary_path) if tta_pp_summary_path.exists() else None
                ),
            }
        )

    out_path = analysis_root / "trials_summary.json"
    analysis_root.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path} with {len(results)} entries.")


if __name__ == "__main__":
    main()

