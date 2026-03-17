#!/usr/bin/env python3
"""
Plot overview from trials_summary.json:
 - Dice vs. spacing (foreground_mean), labelsTs_tta_pp if available else labelsTs.
 - Colors by features_base.
 - Saves to hpo/analysis/plots/trials_dice_vs_spacing.png

Usage:
  python hpo/scripts/analysis/plot_trials_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_trials(summary_path: Path) -> List[Dict[str, Any]]:
    """Load the trials summary JSON produced by ``summarize_trials.py``.

    Args:
        summary_path: Path to ``trials_summary.json``.

    Returns:
        List of trial dictionaries, each containing keys such as ``spacing``,
        ``dice_labelsTs_tta_pp``, ``features_base``, etc.

    Raises:
        FileNotFoundError: If ``summary_path`` does not exist.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run summarize_trials.py first.")
    with summary_path.open() as f:
        return json.load(f)


def main() -> None:
    """Read trials_summary.json and save a Dice-vs-spacing scatter plot."""
    root = Path("hpo")
    analysis_root = root / "analysis"
    plots_root = analysis_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    trials = load_trials(analysis_root / "trials_summary.json")

    xs = []
    ys = []
    colors = []
    labels = []
    for t in trials:
        dice = t.get("dice_labelsTs_tta_pp") or t.get("dice_labelsTs")
        spacing = t.get("spacing")
        fbase = t.get("features_base")
        if dice is None or spacing is None or fbase is None:
            continue
        xs.append(spacing[0])  # isotropic spacings
        ys.append(dice)
        colors.append(fbase)
        labels.append(t.get("trial"))

    if not xs:
        print("No trials with spacing/dice/features_base found.")
        return

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(xs, ys, c=colors, cmap="viridis", s=60, alpha=0.8, edgecolors="k", linewidths=0.5)
    for x, y, name in zip(xs, ys, labels):
        plt.text(x, y, name, fontsize=7, ha="center", va="bottom", alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label("features_base")
    plt.xlabel("spacing (mm)")
    plt.ylabel("Dice (foreground_mean)")
    plt.title("Dice vs. spacing (using labelsTs_tta_pp where available)")
    plt.grid(True, alpha=0.3)
    out_path = plots_root / "trials_dice_vs_spacing.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

