#!/usr/bin/env python3
"""
Analyze and visualize how HPO parameters evolved across trials.

This script reads all trial template files from hpo/config/ and extracts
key parameters (spacing, patch_size, batch_size, features_base, etc.),
then creates visualizations showing how these parameters changed over time.

Example:
    python scripts/analysis/analyze_trial_parameters.py
    python scripts/analysis/analyze_trial_parameters.py --config_dir hpo/config --output_dir analysis_results/trials
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    MaxNLocator = None
    MultipleLocator = None
    print("Warning: matplotlib not available. Visualizations will be skipped.")


def extract_trial_number(filename: str) -> Optional[int]:
    """Extract trial number from filename like 'nnUNetPlans_temp_trial_43.json'."""
    match = re.search(r"trial_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def extract_features_base(features_per_stage: List[int]) -> int:
    """Extract features_base from features_per_stage (first element)."""
    if features_per_stage and len(features_per_stage) > 0:
        return features_per_stage[0]
    return None


def calculate_patch_volume(patch_size: List[int]) -> int:
    """Calculate patch volume."""
    return int(np.prod(patch_size))


def load_trial_parameters(config_dir: Path) -> List[Dict]:
    """Load parameters from all trial template files."""
    trial_files = sorted(config_dir.glob("nnUNetPlans_temp_trial_*.json"))
    
    if not trial_files:
        raise FileNotFoundError(f"No trial template files found in {config_dir}")
    
    trials_data = []
    
    for trial_file in trial_files:
        trial_num = extract_trial_number(trial_file.name)
        if trial_num is None:
            continue
        
        try:
            with open(trial_file, "r") as f:
                data = json.load(f)
            
            config = data.get("configurations", {}).get("3d_fullres", {})
            arch = config.get("architecture", {}).get("arch_kwargs", {})
            
            spacing = config.get("spacing", [None, None, None])
            patch_size = config.get("patch_size", [None, None, None])
            batch_size = config.get("batch_size", None)
            features_per_stage = arch.get("features_per_stage", [])
            features_base = extract_features_base(features_per_stage)
            n_conv_per_stage = arch.get("n_conv_per_stage", [])
            batch_dice = config.get("batch_dice", None)
            use_mask_for_norm = config.get("use_mask_for_norm", [False])[0]
            
            # Calculate derived metrics
            patch_volume = calculate_patch_volume(patch_size) if all(p is not None for p in patch_size) else None
            spacing_mean = np.mean(spacing) if all(s is not None for s in spacing) else None
            
            trial_info = {
                "trial_number": trial_num,
                "trial_name": f"trial_{trial_num}",
                "spacing": spacing,
                "spacing_mean": spacing_mean,
                "patch_size": patch_size,
                "patch_volume": patch_volume,
                "batch_size": batch_size,
                "features_per_stage": features_per_stage,
                "features_base": features_base,
                "n_conv_per_stage": n_conv_per_stage,
                "batch_dice": batch_dice,
                "use_mask_for_norm": use_mask_for_norm,
            }
            
            trials_data.append(trial_info)
        except Exception as e:
            print(f"Warning: Failed to load {trial_file.name}: {e}")
            continue
    
    # Sort by trial number
    trials_data.sort(key=lambda x: x["trial_number"])
    
    return trials_data


def load_trials_summary_dice(trials_summary_path: Path) -> Dict[int, Dict]:
    """
    Load Dice per trial from hpo/analysis/trials_summary.json.
    Returns dict: trial_number -> {dice_validation, dice_labelsTs, dice_labelsTs_tta_pp}.
    dice_validation kommt aus fold_0/validation/summary.json – reicht für alle Trials.
    """
    if not trials_summary_path or not trials_summary_path.exists():
        return {}
    try:
        with open(trials_summary_path, "r") as f:
            summary_list = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load trials summary {trials_summary_path}: {e}")
        return {}
    out = {}
    for entry in summary_list:
        trial_name = entry.get("trial", "")
        num = extract_trial_number(trial_name)
        if num is None:
            continue
        out[num] = {
            "dice_validation": entry.get("dice_validation"),
            "dice_labelsTs": entry.get("dice_labelsTs"),
            "dice_labelsTs_tta_pp": entry.get("dice_labelsTs_tta_pp"),
        }
    return out


# Größere Schrift für bessere Lesbarkeit (primär Achsen)
AXIS_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 12


def create_visualizations(trials_data: List[Dict], output_dir: Path):
    """Create comprehensive visualizations of parameter evolution (+ Dice evolution if available)."""
    if not trials_data:
        print("No trial data to visualize.")
        return
    
    trial_numbers = [t["trial_number"] for t in trials_data]
    # X-Achse bei 0 starten, damit kein Eindruck negativer Trials entsteht
    x_max_trials = max(trial_numbers) + max(1, (max(trial_numbers) - min(trial_numbers)) * 0.05)
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Spacing over trials – Y-Achse: Resolution in mm, Werte bei 0 anfangen
    ax1 = fig.add_subplot(gs[0, 0])
    spacing_means = [t["spacing_mean"] for t in trials_data]
    ax1.plot(trial_numbers, spacing_means, "o-", linewidth=2, markersize=6, color="#2E86AB")
    ax1.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel("Resolution in mm", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title("Spacing Evolution", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax1.set_xlim(0, x_max_trials)
    if spacing_means:
        ax1.set_ylim(0, max(spacing_means) * 1.05)  # Wertebereich inkl. 0
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # 0 oben = feinste Resolution
    
    # 2. Patch Volume over trials – Y bei 0
    ax2 = fig.add_subplot(gs[0, 1])
    patch_volumes = [t["patch_volume"] for t in trials_data if t["patch_volume"] is not None]
    patch_trials = [t["trial_number"] for t in trials_data if t["patch_volume"] is not None]
    ax2.plot(patch_trials, patch_volumes, "s-", linewidth=2, markersize=6, color="#A23B72")
    ax2.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel("Patch Volume (voxels³)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title("Patch Volume Evolution", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax2.set_xlim(0, x_max_trials)
    if patch_volumes:
        ax2.set_ylim(0, max(patch_volumes) * 1.05)
    ax2.grid(True, alpha=0.3)
    
    # 3. Features Base over trials – Y bei 0
    ax3 = fig.add_subplot(gs[0, 2])
    features_bases = [t["features_base"] for t in trials_data if t["features_base"] is not None]
    features_trials = [t["trial_number"] for t in trials_data if t["features_base"] is not None]
    ax3.plot(features_trials, features_bases, "^-", linewidth=2, markersize=6, color="#F18F01")
    ax3.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax3.set_ylabel("Features Base", fontsize=AXIS_LABEL_FONTSIZE)
    ax3.set_title("Model Capacity Evolution", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax3.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax3.set_xlim(0, x_max_trials)
    if features_bases:
        ax3.set_ylim(0, max(features_bases) * 1.05)
    ax3.grid(True, alpha=0.3)
    
    # 4. Patch Dimensions – Y bei 0
    ax4 = fig.add_subplot(gs[1, 0])
    patch_x = [t["patch_size"][0] for t in trials_data if t["patch_size"][0] is not None]
    patch_y = [t["patch_size"][1] for t in trials_data if t["patch_size"][1] is not None]
    patch_z = [t["patch_size"][2] for t in trials_data if t["patch_size"][2] is not None]
    patch_dim_trials = [t["trial_number"] for t in trials_data if t["patch_size"][0] is not None]
    ax4.plot(patch_dim_trials, patch_x, "o-", label="X", linewidth=2, markersize=5)
    ax4.plot(patch_dim_trials, patch_y, "s-", label="Y", linewidth=2, markersize=5)
    ax4.plot(patch_dim_trials, patch_z, "^-", label="Z", linewidth=2, markersize=5)
    ax4.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax4.set_ylabel("Patch Dimension (voxels)", fontsize=AXIS_LABEL_FONTSIZE)
    ax4.set_title("Patch Dimensions", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax4.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax4.set_xlim(0, x_max_trials)
    if patch_x or patch_y or patch_z:
        all_dim = patch_x + patch_y + patch_z
        ax4.set_ylim(0, max(all_dim) * 1.05)
    ax4.legend(fontsize=TICK_LABEL_FONTSIZE)
    ax4.grid(True, alpha=0.3)
    
    # 5. Batch Size over trials – Y bei 0
    ax5 = fig.add_subplot(gs[1, 1])
    batch_sizes = [t["batch_size"] for t in trials_data if t["batch_size"] is not None]
    batch_trials = [t["trial_number"] for t in trials_data if t["batch_size"] is not None]
    ax5.plot(batch_trials, batch_sizes, "D-", linewidth=2, markersize=6, color="#C73E1D")
    ax5.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax5.set_ylabel("Batch Size", fontsize=AXIS_LABEL_FONTSIZE)
    ax5.set_title("Batch Size Evolution", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax5.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax5.set_xlim(0, x_max_trials)
    ax5.set_yticks([1, 2, 3, 4])
    if batch_sizes:
        ax5.set_ylim(0, max(4, max(batch_sizes)) * 1.1)
    ax5.grid(True, alpha=0.3)
    
    # 6. Batch Dice flag over trials – Y bereits 0–1
    ax6 = fig.add_subplot(gs[1, 2])
    batch_dice_flags = [1 if t["batch_dice"] else 0 for t in trials_data if t["batch_dice"] is not None]
    batch_dice_trials = [t["trial_number"] for t in trials_data if t["batch_dice"] is not None]
    ax6.plot(batch_dice_trials, batch_dice_flags, "o-", linewidth=2, markersize=8, color="#6A994E")
    ax6.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax6.set_ylabel("Batch Dice (1=True, 0=False)", fontsize=AXIS_LABEL_FONTSIZE)
    ax6.set_title("Batch Dice Usage", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax6.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax6.set_xlim(0, x_max_trials)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(["False", "True"])
    ax6.grid(True, alpha=0.3)
    
    # 7. Spacing vs Patch Volume (scatter)
    ax7 = fig.add_subplot(gs[2, 0])
    spacing_for_scatter = [t["spacing_mean"] for t in trials_data if t["spacing_mean"] is not None and t["patch_volume"] is not None]
    volume_for_scatter = [t["patch_volume"] for t in trials_data if t["spacing_mean"] is not None and t["patch_volume"] is not None]
    features_for_scatter = [t["features_base"] for t in trials_data if t["spacing_mean"] is not None and t["patch_volume"] is not None and t["features_base"] is not None]
    
    if spacing_for_scatter and volume_for_scatter:
        scatter = ax7.scatter(
            spacing_for_scatter,
            volume_for_scatter,
            c=features_for_scatter if features_for_scatter else None,
            s=100,
            alpha=0.6,
            cmap="viridis",
            edgecolors="black",
            linewidths=1
        )
        ax7.set_xlabel("Mean Spacing (mm)", fontsize=AXIS_LABEL_FONTSIZE)
        ax7.set_ylabel("Patch Volume (voxels³)", fontsize=AXIS_LABEL_FONTSIZE)
        ax7.set_title("Spacing vs Patch Volume", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax7.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax7.invert_xaxis()  # Smaller spacing is better
        if features_for_scatter:
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            cbar.set_label("Features Base", fontsize=AXIS_LABEL_FONTSIZE)
        ax7.grid(True, alpha=0.3)
    
    # 8. Features Base vs Patch Volume
    ax8 = fig.add_subplot(gs[2, 1])
    if features_for_scatter and volume_for_scatter:
        ax8.scatter(
            features_for_scatter,
            volume_for_scatter,
            c=[t["spacing_mean"] for t in trials_data if t["spacing_mean"] is not None and t["patch_volume"] is not None and t["features_base"] is not None],
            s=100,
            alpha=0.6,
            cmap="plasma",
            edgecolors="black",
            linewidths=1
        )
        ax8.set_xlabel("Features Base", fontsize=AXIS_LABEL_FONTSIZE)
        ax8.set_ylabel("Patch Volume (voxels³)", fontsize=AXIS_LABEL_FONTSIZE)
        ax8.set_title("Model Capacity vs Patch Size", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax8.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        cbar = plt.colorbar(ax8.collections[0], ax=ax8)
        cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        cbar.set_label("Spacing (mm)", fontsize=AXIS_LABEL_FONTSIZE)
        ax8.grid(True, alpha=0.3)
    
    # 9. Parameter combinations heatmap or summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    
    # Create summary statistics
    summary_text = "Parameter Summary\n" + "=" * 30 + "\n\n"
    summary_text += f"Total Trials: {len(trials_data)}\n\n"
    
    if spacing_means:
        summary_text += f"Spacing Range:\n  {min(spacing_means):.3f} - {max(spacing_means):.3f} mm\n\n"
    
    if patch_volumes:
        summary_text += f"Patch Volume Range:\n  {min(patch_volumes):,} - {max(patch_volumes):,} voxels³\n\n"
    
    if features_bases:
        summary_text += f"Features Base Range:\n  {min(features_bases)} - {max(features_bases)}\n\n"
    
    if batch_sizes:
        unique_batches = sorted(set(batch_sizes))
        summary_text += f"Batch Sizes Used:\n  {', '.join(map(str, unique_batches))}\n\n"
    
    batch_dice_count = sum(1 for t in trials_data if t.get("batch_dice") is True)
    batch_dice_total = sum(1 for t in trials_data if t.get("batch_dice") is not None)
    if batch_dice_total > 0:
        summary_text += f"Batch Dice Usage:\n  {batch_dice_count}/{batch_dice_total} trials\n"
    
    ax9.text(0.1, 0.5, summary_text, fontsize=TICK_LABEL_FONTSIZE, family="monospace", verticalalignment="center")
    
    plt.suptitle("HPO Parameter Evolution Across Trials", fontsize=16, fontweight="bold", y=0.995)
    
    # Save figure
    output_path = output_dir / "trial_parameters_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def save_dice_evolution_figure(trials_data: List[Dict], output_dir: Path) -> None:
    """Dice-Visualisierung: Validation-Dice, Bestes Trial hervorgehoben, Höhen/Tiefen erkennbar."""
    dice_val = [t.get("dice_validation") for t in trials_data]
    dice_lbl = [t.get("dice_labelsTs") for t in trials_data]
    use_val = any(x is not None for x in dice_val)
    use_lbl = any(x is not None for x in dice_lbl)
    if not use_val and not use_lbl:
        return
    trial_numbers = [t["trial_number"] for t in trials_data]
    if use_val:
        vals = [x if x is not None else float("nan") for x in dice_val]
        title_suffix = "Validation"
    else:
        vals = [x if x is not None else float("nan") for x in dice_lbl]
        title_suffix = "labelsTs"
    valid_pairs = [(trial_numbers[i], v) for i, v in enumerate(vals) if not np.isnan(v)]
    if not valid_pairs:
        return
    best_trial, best_dice = max(valid_pairs, key=lambda x: x[1])
    worst_trial, worst_dice = min(valid_pairs, key=lambda x: x[1])
    val_min, val_max = worst_dice, best_dice
    y_range = val_max - val_min
    y_pad = max(0.05, y_range * 0.15) if y_range > 0 else 0.1
    y_lo = max(0, val_min - y_pad)
    y_hi = min(1.02, val_max + y_pad)

    x_max = max(trial_numbers) + max(1, (max(trial_numbers) - min(trial_numbers)) * 0.05)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(trial_numbers, vals, "o-", linewidth=2, markersize=6, color="#2E86AB", label=f"Dice ({title_suffix})", zorder=2)
    # Niedrigsten Punkt markieren (Lows sichtbar)
    ax.scatter([worst_trial], [worst_dice], s=80, zorder=3, color="#C1121F", edgecolors="darkred", linewidths=1.5, marker="v", label=f"Low: {worst_dice:.4f} (Trial {worst_trial})")
    # Besten Punkt wie Low hervorheben, nur anderer Marker + Annotation mit Wert
    ax.scatter([best_trial], [best_dice], s=80, zorder=3, color="#2E86AB", edgecolors="darkgreen", linewidths=2, marker="^", label=f"Best: {best_dice:.4f} (Trial {best_trial})")
    ax.annotate(
        f"Best: {best_dice:.4f}\nTrial {best_trial}",
        xy=(best_trial, best_dice),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=TICK_LABEL_FONTSIZE + 1,
        fontweight="bold",
        color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", edgecolor="darkgreen", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
    )
    ax.set_title(f"Dice Evolution over Trials ({title_suffix})", fontsize=TITLE_FONTSIZE + 1, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Dice", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="lower right", fontsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(MaxNLocator(20))
    ax.grid(True, which="minor", alpha=0.15)
    plt.tight_layout()
    out_path = output_dir / "dice_evolution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Dice visualization saved to: {out_path}")


def save_dice_vs_parameters_figure(
    trials_data: List[Dict],
    output_dir: Path,
    trial_min: int = 16,
    trial_max: int = 30,
) -> None:
    """Relation Dice vs. Parameter: 4 Scatter-Plots (4×1), Dice-Achse fein skaliert (0.01), Abstände gut sichtbar."""
    subset = [t for t in trials_data if trial_min <= t["trial_number"] <= trial_max]
    subset = [t for t in subset if t.get("dice_validation") is not None]
    if not subset:
        return
    trial_numbers = [t["trial_number"] for t in subset]
    dice = np.array([t["dice_validation"] for t in subset])
    spacing_mean = [t.get("spacing_mean") for t in subset]
    patch_volume = [t.get("patch_volume") for t in subset]
    batch_size = [t.get("batch_size") for t in subset]
    features_base = [t.get("features_base") for t in subset]

    # Dice-Bereich aus Daten + Puffer, auf 0.01 gerundet (feine Skala für Abstände)
    dice_min, dice_max = float(np.nanmin(dice)), float(np.nanmax(dice))
    dice_span = max(dice_max - dice_min, 0.05)
    pad = dice_span * 0.12
    y_lo = np.floor((dice_min - pad) * 100) / 100
    y_hi = np.ceil((dice_max + pad) * 100) / 100
    y_lo = max(0.0, y_lo)
    y_hi = min(1.0, y_hi)
    if y_hi <= y_lo:
        y_lo, y_hi = 0.0, 1.0
    norm = plt.Normalize(vmin=y_lo, vmax=y_hi)
    cmap = plt.cm.RdYlGn
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    DICE_TICK_STEP = 0.01

    def scatter_dice(ax, x_vals, y_vals, xlabel, title):
        valid = [(i, x, y) for i, (x, y) in enumerate(zip(x_vals, y_vals)) if x is not None and y is not None]
        if not valid:
            ax.set_visible(False)
            return
        xx = np.array([v[1] for v in valid])
        yy = np.array([v[2] for v in valid])
        indices = [v[0] for v in valid]
        ax.scatter(xx, yy, s=120, c=yy, cmap=cmap, norm=norm, edgecolors="black", linewidths=0.6, zorder=2)
        order = np.argsort(yy)
        for idx in sorted(set([order[-1], order[0]])):
            ax.annotate(
                str(trial_numbers[indices[idx]]),
                (xx[idx], yy[idx]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=11,
                ha="left",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.9),
            )
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Dice (Validation)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(MultipleLocator(DICE_TICK_STEP))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax.grid(True, alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)

    scatter_dice(axes[0], spacing_mean, dice, "Resolution (mm)", "Dice vs. Spacing")
    scatter_dice(axes[1], patch_volume, dice, "Patch Volume (voxels³)", "Dice vs. Patch Volume")
    scatter_dice(axes[2], batch_size, dice, "Batch Size", "Dice vs. Batch Size")
    scatter_dice(axes[3], features_base, dice, "Features Base", "Dice vs. Features Base")

    fig.suptitle(f"Dice vs. Parameters (Trials {trial_min}–{trial_max}) — Dice in 0.01-Schritten, Farbe = Dice", fontsize=TITLE_FONTSIZE + 1, fontweight="bold", y=1.01)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax, label="Dice")
    cbar.ax.yaxis.set_major_locator(MultipleLocator(DICE_TICK_STEP))
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    out_path = output_dir / f"dice_vs_parameters_trials_{trial_min}_{trial_max}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Dice vs. parameters (trials {trial_min}–{trial_max}) saved to: {out_path}")


def create_individual_plots(trials_data: List[Dict], output_dir: Path):
    """Create individual plots for each parameter."""
    plots_dir = output_dir / "trial_parameters_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    trial_numbers = [t["trial_number"] for t in trials_data]
    x_max_trials = max(trial_numbers) + max(1, (max(trial_numbers) - min(trial_numbers)) * 0.05)
    
    # 1. Spacing – Y: Resolution in mm, X bei 0
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing_means = [t["spacing_mean"] for t in trials_data]
    ax.plot(trial_numbers, spacing_means, "o-", linewidth=2.5, markersize=8, color="#2E86AB")
    ax.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Resolution in mm", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Spacing Evolution Across Trials", fontsize=TITLE_FONTSIZE + 1, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, x_max_trials)
    if spacing_means:
        ax.set_ylim(0, max(spacing_means) * 1.05)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / "spacing_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Patch Volume – Y bei 0, X bei 0
    fig, ax = plt.subplots(figsize=(12, 6))
    patch_volumes = [t["patch_volume"] for t in trials_data if t["patch_volume"] is not None]
    patch_trials = [t["trial_number"] for t in trials_data if t["patch_volume"] is not None]
    ax.plot(patch_trials, patch_volumes, "s-", linewidth=2.5, markersize=8, color="#A23B72")
    ax.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Patch Volume (voxels³)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Patch Volume Evolution Across Trials", fontsize=TITLE_FONTSIZE + 1, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, x_max_trials)
    if patch_volumes:
        ax.set_ylim(0, max(patch_volumes) * 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "patch_volume_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Features Base – Y bei 0, X bei 0
    fig, ax = plt.subplots(figsize=(12, 6))
    features_bases = [t["features_base"] for t in trials_data if t["features_base"] is not None]
    features_trials = [t["trial_number"] for t in trials_data if t["features_base"] is not None]
    ax.plot(features_trials, features_bases, "^-", linewidth=2.5, markersize=8, color="#F18F01")
    ax.set_xlabel("Trial Number", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Features Base", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Model Capacity (Features Base) Evolution Across Trials", fontsize=TITLE_FONTSIZE + 1, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, x_max_trials)
    if features_bases:
        ax.set_ylim(0, max(features_bases) * 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "features_base_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Individual plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize HPO parameter evolution across trials."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="hpo/config",
        help="Directory containing trial template files (default: hpo/config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results/trials",
        help="Output directory for results (default: analysis_results/trials)",
    )
    parser.add_argument(
        "--individual_plots",
        action="store_true",
        help="Create individual plots for each parameter",
    )
    parser.add_argument(
        "--trials_summary",
        type=str,
        default=None,
        help="Path to hpo/analysis/trials_summary.json for Dice per trial (default: <config_dir>/../analysis/trials_summary.json)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save files, only print to stdout",
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    print(f"Loading trial parameters from: {config_dir}")
    trials_data = load_trial_parameters(config_dir)
    
    if not trials_data:
        print("No trial data found.")
        return
    
    # Optional: Dice per trial from hpo/analysis/trials_summary.json (from summarize_trials.py)
    trials_summary_path = Path(args.trials_summary) if args.trials_summary else (config_dir.parent / "analysis" / "trials_summary.json")
    dice_per_trial = load_trials_summary_dice(trials_summary_path)
    if dice_per_trial:
        for t in trials_data:
            num = t["trial_number"]
            d = dice_per_trial.get(num, {})
            t["dice_validation"] = d.get("dice_validation")
            t["dice_labelsTs"] = d.get("dice_labelsTs")
            t["dice_labelsTs_tta_pp"] = d.get("dice_labelsTs_tta_pp")
        print(f"Loaded Dice for {len(dice_per_trial)} trials from {trials_summary_path}")
    else:
        print("No trials_summary.json found — run hpo/scripts/analysis/summarize_trials.py first for Dice evolution plot.")
    
    print(f"Loaded {len(trials_data)} trials")
    
    # Save JSON data
    if not args.no_save:
        json_path = output_dir / "trial_parameters.json"
        with open(json_path, "w") as f:
            json.dump(trials_data, f, indent=2)
        print(f"✓ Parameter data saved to: {json_path}")
    
    # Create visualizations (skip if matplotlib not available)
    if not args.no_save and HAS_MATPLOTLIB:
        create_visualizations(trials_data, output_dir)
        save_dice_evolution_figure(trials_data, output_dir)
        save_dice_vs_parameters_figure(trials_data, output_dir, trial_min=16, trial_max=30)
        if args.individual_plots:
            create_individual_plots(trials_data, output_dir)
    elif not args.no_save and not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available). JSON and summary still written.")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PARAMETER EVOLUTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal trials analyzed: {len(trials_data)}")
    
    spacing_means = [t["spacing_mean"] for t in trials_data if t["spacing_mean"] is not None]
    if spacing_means:
        print(f"\nSpacing (mm):")
        print(f"  Range: {min(spacing_means):.3f} - {max(spacing_means):.3f}")
        print(f"  Mean: {np.mean(spacing_means):.3f} ± {np.std(spacing_means):.3f}")
    
    patch_volumes = [t["patch_volume"] for t in trials_data if t["patch_volume"] is not None]
    if patch_volumes:
        print(f"\nPatch Volume (voxels³):")
        print(f"  Range: {min(patch_volumes):,} - {max(patch_volumes):,}")
        print(f"  Mean: {np.mean(patch_volumes):,.0f} ± {np.std(patch_volumes):,.0f}")
    
    features_bases = [t["features_base"] for t in trials_data if t["features_base"] is not None]
    if features_bases:
        print(f"\nFeatures Base:")
        print(f"  Range: {min(features_bases)} - {max(features_bases)}")
        print(f"  Mean: {np.mean(features_bases):.1f} ± {np.std(features_bases):.1f}")
    
    batch_sizes = [t["batch_size"] for t in trials_data if t["batch_size"] is not None]
    if batch_sizes:
        unique_batches = sorted(set(batch_sizes))
        print(f"\nBatch Sizes Used: {', '.join(map(str, unique_batches))}")
    
    batch_dice_count = sum(1 for t in trials_data if t.get("batch_dice") is True)
    batch_dice_total = sum(1 for t in trials_data if t.get("batch_dice") is not None)
    if batch_dice_total > 0:
        print(f"\nBatch Dice: {batch_dice_count}/{batch_dice_total} trials used batch_dice=True")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
