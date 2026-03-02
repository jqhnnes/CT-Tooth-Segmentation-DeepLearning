#!/usr/bin/env python3
"""
Create a histogram of label value distribution from segmentation masks.

This script loads all label files from a specified dataset and creates
a histogram showing the distribution of label values (voxel counts per class).

Example:
    python scripts/analysis/create_label_histogram.py --dataset Dataset002_Karies
    python scripts/analysis/create_label_histogram.py --dataset Dataset002_Karies --output plots/label_histogram.png
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")


def load_dataset_info(dataset_path: Path) -> Dict:
    """Load dataset.json to get label names and other metadata."""
    dataset_json = dataset_path / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json}")
    
    with open(dataset_json, "r") as f:
        return json.load(f)


def load_label_files(labels_dir: Path) -> List[np.ndarray]:
    """Load all label files from the directory."""
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    if not label_files:
        raise FileNotFoundError(f"No .nii.gz files found in {labels_dir}")
    
    print(f"Loading {len(label_files)} label files...")
    label_arrays = []
    for label_file in label_files:
        img = nib.load(str(label_file))
        data = img.get_fdata().astype(np.int32)
        label_arrays.append(data)
    
    return label_arrays


def compute_label_counts(label_arrays: List[np.ndarray], label_map: Dict[str, int]) -> Dict[int, int]:
    """Count voxels for each label value across all label files."""
    label_counts = {label_id: 0 for label_id in label_map.values()}
    
    print("Computing label distribution...")
    for arr in label_arrays:
        unique, counts = np.unique(arr, return_counts=True)
        for label_id, count in zip(unique, counts):
            label_id = int(label_id)
            if label_id in label_counts:
                label_counts[label_id] += count
    
    return label_counts


def get_label_colors(labels: List[str]) -> List[str]:
    """Get meaningful colors for medical labels."""
    color_map = {
        "background": "#2C3E50",  # Dark gray/blue
        "pulp": "#E74C3C",  # Red
        "dentin": "#F39C12",  # Orange
        "enamel": "#ECF0F1",  # Light gray/white
        "enamel_caries": "#8E44AD",  # Purple
        "dentin_caries": "#C0392B",  # Dark red
    }
    return [color_map.get(label.lower().replace("_", ""), "#95A5A6") for label in labels]


def create_histogram(
    label_counts: Dict[int, int],
    label_map: Dict[str, int],
    output_path: Path,
    title: str = "Label Distribution",
    use_log_scale: bool = False,
):
    """Create and save a histogram of label distribution."""
    # Set style
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn" in plt.style.available else "default")
    
    # Create reverse mapping: label_id -> label_name
    id_to_name = {v: k for k, v in label_map.items()}
    
    # Sort by label ID
    sorted_ids = sorted(label_counts.keys())
    labels = [id_to_name.get(lid, f"Unknown_{lid}") for lid in sorted_ids]
    counts = [label_counts[lid] for lid in sorted_ids]
    
    # Calculate percentages
    total_voxels = sum(counts)
    percentages = [100 * c / total_voxels for c in counts]
    
    # Get colors
    colors = get_label_colors(labels)
    
    # Create figure with better proportions
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Main bar plot (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Format labels for display
    display_labels = [label.replace("_", " ").title() for label in labels]
    
    bars = ax1.bar(range(len(labels)), counts, color=colors, edgecolor="black", linewidth=1.5, alpha=0.85)
    ax1.set_xlabel("Label Class", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Voxel Count", fontsize=13, fontweight="bold")
    ax1.set_title(f"{title} - Voxel Counts", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=11, fontweight="medium")
    ax1.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Use log scale if requested and if there are large differences
    if use_log_scale or (max(counts) / min(counts) > 100):
        ax1.set_yscale("log")
        ax1.set_ylabel("Voxel Count (log scale)", fontsize=13, fontweight="bold")
    
    # Add value labels on bars (only if not too crowded)
    max_count = max(counts)
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        # Only show labels if bar is tall enough
        if height > max_count * 0.05:
            label_text = f"{count:,}\n({pct:.1f}%)"
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.02,
                label_text,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
            )
    
    # Pie chart (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=display_labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "medium"},
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )
    ax2.set_title("Percentage Distribution", fontsize=14, fontweight="bold", pad=15)
    
    # Improve text readability in pie chart
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)
    
    # Statistics table (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    
    # Create table data
    table_data = []
    for i, (lid, label, count, pct) in enumerate(zip(sorted_ids, display_labels, counts, percentages)):
        table_data.append([
            f"ID {lid}",
            label,
            f"{count:,}",
            f"{pct:.2f}%",
        ])
    
    table = ax3.table(
        cellText=table_data,
        colLabels=["ID", "Label", "Voxels", "Percentage"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#34495E")
                cell.set_text_props(weight="bold", color="white")
            else:
                if i % 2 == 0:
                    cell.set_facecolor("#ECF0F1")
                else:
                    cell.set_facecolor("white")
                cell.set_edgecolor("#BDC3C7")
    
    ax3.set_title("Detailed Statistics", fontsize=14, fontweight="bold", pad=15)
    
    # Add total voxels as text
    fig.text(0.5, 0.02, f"Total Voxels: {total_voxels:,}", 
             ha="center", fontsize=12, fontweight="bold", style="italic")
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Histogram saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Label Distribution Summary")
    print("=" * 60)
    for lid in sorted_ids:
        name = id_to_name.get(lid, f"Unknown_{lid}")
        count = label_counts[lid]
        pct = 100 * count / total_voxels
        print(f"  {name:20s} (ID {lid}): {count:>12,} voxels ({pct:>6.2f}%)")
    print("=" * 60)
    print(f"Total voxels: {total_voxels:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create histogram of label distribution from segmentation masks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset002_Karies",
        help="Dataset name (default: Dataset002_Karies)",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        default="data/nnUNet_raw",
        help="Path to nnUNet_raw directory (default: data/nnUNet_raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for histogram (default: plots/<dataset>_label_histogram.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the histogram (default: '<Dataset> Label Distribution')",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use logarithmic scale for y-axis (useful for large count differences)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    nnunet_raw = Path(args.nnunet_raw)
    dataset_path = nnunet_raw / args.dataset
    labels_dir = dataset_path / "labelsTr"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Load dataset info
    dataset_info = load_dataset_info(dataset_path)
    label_map = dataset_info.get("labels", {})
    
    if not label_map:
        raise ValueError(f"No 'labels' found in dataset.json for {args.dataset}")
    
    print(f"Dataset: {args.dataset}")
    print(f"Labels: {label_map}")
    
    # Load all label files
    label_arrays = load_label_files(labels_dir)
    
    # Compute label distribution
    label_counts = compute_label_counts(label_arrays, label_map)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path("analysis_results") / "datasets"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / f"label_histogram_{args.dataset}_{timestamp}.png"
    
    # Determine title
    title = args.title if args.title else f"{args.dataset} Label Distribution"
    
    # Create histogram
    create_histogram(label_counts, label_map, output_path, title, use_log_scale=args.log_scale)


if __name__ == "__main__":
    main()
