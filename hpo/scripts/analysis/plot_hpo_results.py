#!/usr/bin/env python3
"""
Plot HPO results: Dice Score vs Parameters.

This script creates visualizations showing how different hyperparameters
affect the Dice score across all HPO trials.

Usage:
    python hpo/scripts/analysis/plot_hpo_results.py
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path("hpo") / "results" / "Dataset001_GroundTruth"
ANALYSIS_DIR = Path("hpo") / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "plots"


def load_trial_results() -> Dict[str, Dict]:
    """Load Dice scores and parameters for all trials."""
    trials_data = {}
    
    # Load all trial results from summary.json files
    for trial_dir in RESULTS_DIR.iterdir():
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue
        
        summary_file = trial_dir / "3d_fullres" / "nnUNetTrainer" / "fold_0_summary.json"
        if not summary_file.exists():
            continue
        
        trial_name = trial_dir.name
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract foreground mean Dice (overall Dice score)
        dice_score = summary.get("foreground_mean", {}).get("Dice", None)
        
        if dice_score is None:
            continue
        
        plans_file = Path("hpo") / "preprocessing_output" / "Dataset001_GroundTruth" / trial_name / "Dataset001_GroundTruth" / "nnUNetPlans.json"
        parameters: Dict[str, Any] = {}
        if plans_file.exists():
            with open(plans_file, 'r') as f:
                plans = json.load(f)
            
            config = plans.get("configurations", {}).get("3d_fullres", {})
            arch = config.get("architecture", {}).get("arch_kwargs", {})
            
            features_per_stage = arch.get("features_per_stage", [])
            features_base = features_per_stage[0] if features_per_stage else None
            
            parameters = {
                "patch_size": config.get("patch_size", []),
                "batch_size": config.get("batch_size"),
                "features_base": features_base,
                "features_per_stage": features_per_stage,
                "n_conv_per_stage": arch.get("n_conv_per_stage"),
                "batch_dice": config.get("batch_dice"),
                "use_mask_for_norm": config.get("use_mask_for_norm", [False])[0] if isinstance(config.get("use_mask_for_norm"), list) else config.get("use_mask_for_norm"),
                "spacing": config.get("spacing"),
            }
        trials_data[trial_name] = {
            "dice": dice_score,
            "parameters": parameters,
        }
    
    return trials_data


def plot_parameter_vs_dice(trials_data: Dict[str, Dict], param_name: str, param_extractor, output_file: Path):
    """Plot Dice score vs a specific parameter."""
    x_values = []
    y_values = []
    trial_names = []
    
    for trial_name, data in sorted(trials_data.items()):
        dice = data.get("dice")
        params = data.get("parameters", {})
        
        if dice is None:
            continue
        
        param_value = param_extractor(params)
        if param_value is not None:
            x_values.append(param_value)
            y_values.append(dice)
            trial_names.append(trial_name)
    
    if not x_values:
        print(f"[WARN] No data available for parameter {param_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    scatter = ax.scatter(x_values, y_values, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add trial labels
    for i, trial_name in enumerate(trial_names):
        ax.annotate(trial_name, (x_values[i], y_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Trend line if numeric
    if all(isinstance(x, (int, float)) for x in x_values):
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x_values), p(sorted(x_values)), "r--", alpha=0.5, label=f'Trend line')
        ax.legend()
    
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Dice Score vs {param_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_patch_size_vs_dice(trials_data: Dict[str, Dict], output_file: Path):
    """Plot Dice vs patch size (as total volume)."""
    x_values = []
    y_values = []
    trial_names = []
    
    for trial_name, data in sorted(trials_data.items()):
        dice = data.get("dice")
        params = data.get("parameters", {})
        
        if dice is None:
            continue
        
        patch_size = params.get("patch_size", [])
        if len(patch_size) == 3:
            # Use total volume as x-axis
            volume = patch_size[0] * patch_size[1] * patch_size[2]
            x_values.append(volume)
            y_values.append(dice)
            trial_names.append(trial_name)
    
    if not x_values:
        print("[WARN] No patch size data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(x_values, y_values, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    for i, trial_name in enumerate(trial_names):
        patch = trials_data[trial_name]["parameters"].get("patch_size", [])
        label = f"{trial_name}\n{patch}"
        ax.annotate(label, (x_values[i], y_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    ax.plot(sorted(x_values), p(sorted(x_values)), "r--", alpha=0.5, label='Trend line')
    ax.legend()
    
    ax.set_xlabel('Patch Size Volume (x × y × z)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Dice Score vs Patch Size Volume', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_spacing_vs_dice(trials_data: Dict[str, Dict], output_file: Path):
    """Plot Dice vs target spacing."""
    x_values = []
    y_values = []
    labels = []
    
    for trial_name, data in sorted(trials_data.items()):
        dice = data.get("dice")
        spacing = data.get("parameters", {}).get("spacing")
        if dice is None or not spacing or len(spacing) != 3:
            continue
        iso_spacing = sum(spacing) / 3.0
        x_values.append(iso_spacing)
        y_values.append(dice)
        labels.append(f"{trial_name}\n{spacing}")
    
    if not x_values:
        print("[WARN] No spacing data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (x_values[i], y_values[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    sorted_x = sorted(x_values)
    ax.plot(sorted_x, p(sorted_x), "r--", alpha=0.5, label='Trend line')
    ax.legend()
    
    ax.set_xlabel('Isotropic Spacing (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Dice Score vs Target Spacing', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_plot(trials_data: Dict[str, Dict], output_file: Path):
    """Create a summary plot showing all trials sorted by Dice score."""
    sorted_trials = sorted(trials_data.items(), key=lambda x: x[1].get("dice", 0), reverse=True)
    
    trial_names = [name for name, _ in sorted_trials]
    dice_scores = [data.get("dice", 0) for _, data in sorted_trials]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if d > 0.96 else 'orange' if d > 0.5 else 'red' for d in dice_scores]
    bars = ax.bar(range(len(trial_names)), dice_scores, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('HPO Results: Dice Score by Trial (Sorted)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(trial_names)))
    ax.set_xticklabels(trial_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Dice > 0.96 (Good)'),
        Patch(facecolor='orange', alpha=0.7, label='0.5 < Dice ≤ 0.96 (Moderate)'),
        Patch(facecolor='red', alpha=0.7, label='Dice ≤ 0.5 (Poor)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading trial results...")
    trials_data = load_trial_results()
    
    if not trials_data:
        print("[ERROR] No trial data found. Please run evaluation first.")
        return
    
    print(f"Loaded data for {len(trials_data)} trials")
    
    # Create summary plot
    print("\nCreating summary plot...")
    create_summary_plot(trials_data, OUTPUT_DIR / "hpo_summary.png")
    
    # Plot individual parameters
    print("\nCreating parameter plots...")
    
    # Features base
    plot_parameter_vs_dice(
        trials_data,
        "Features Base",
        lambda p: p.get("features_base"),
        OUTPUT_DIR / "dice_vs_features_base.png"
    )
    
    # Batch size
    plot_parameter_vs_dice(
        trials_data,
        "Batch Size",
        lambda p: p.get("batch_size"),
        OUTPUT_DIR / "dice_vs_batch_size.png"
    )
    
    # n_conv_per_stage
    plot_parameter_vs_dice(
        trials_data,
        "Convolutions per Stage",
        lambda p: (
            p.get("n_conv_per_stage")[0]
            if isinstance(p.get("n_conv_per_stage"), list) and p.get("n_conv_per_stage")
            else p.get("n_conv_per_stage")
        ),
        OUTPUT_DIR / "dice_vs_n_conv.png"
    )
    
    # Patch size (volume)
    plot_patch_size_vs_dice(trials_data, OUTPUT_DIR / "dice_vs_patch_size.png")
    
    # Spacing
    plot_spacing_vs_dice(trials_data, OUTPUT_DIR / "dice_vs_spacing.png")
    
    # Batch dice (boolean)
    plot_parameter_vs_dice(
        trials_data,
        "Batch Dice",
        lambda p: "True" if p.get("batch_dice") else "False",
        OUTPUT_DIR / "dice_vs_batch_dice.png"
    )
    
    # Use mask for norm (boolean)
    plot_parameter_vs_dice(
        trials_data,
        "Use Mask for Norm",
        lambda p: "True" if p.get("use_mask_for_norm") else "False",
        OUTPUT_DIR / "dice_vs_use_mask_for_norm.png"
    )
    
    print(f"\n{'='*80}")
    print("All plots saved to:", OUTPUT_DIR)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

