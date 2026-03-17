#!/usr/bin/env python3
"""
Create a histogram of CT intensity (grayscale) distributions for different tooth tissues.

This script loads CT images and their corresponding labels, extracts intensity values
for each tissue type (enamel, dentin, pulp, background), and creates a histogram
showing the characteristic peaks and overlap regions.

Example:
    python scripts/analysis/dataset/create_grayscale_histogram.py --dataset Dataset002_Karies
    python scripts/analysis/dataset/create_grayscale_histogram.py --dataset Dataset002_Karies --output images/grayscale_histogram.png
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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


def get_matching_files(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get matching pairs of image and label files."""
    image_files = sorted(images_dir.glob("*.nii.gz"))
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    
    # Match by filename (images have _0000 suffix, labels don't)
    matches = []
    # Create dict: base_name (without _0000) -> label_file
    label_dict = {}
    for label_file in label_files:
        base_name = label_file.stem  # e.g., "AW002-C0005280"
        label_dict[base_name] = label_file
    
    for img_file in image_files:
        # Remove _0000 suffix if present
        base_name = img_file.stem.replace("_0000", "")  # e.g., "AW002-C0005280_0000" -> "AW002-C0005280"
        if base_name in label_dict:
            matches.append((img_file, label_dict[base_name]))
    
    if not matches:
        raise FileNotFoundError(
            f"No matching image-label pairs found in {images_dir} and {labels_dir}. "
            f"Found {len(image_files)} images and {len(label_files)} labels."
        )
    
    return matches


def extract_intensities_by_tissue(
    image_path: Path,
    label_path: Path,
    label_map: Dict[str, int],
    tissues_of_interest: List[str] = None,
    max_voxels_per_tissue: int = 50000,
) -> Dict[str, np.ndarray]:
    """Extract CT intensities for each tissue type with optional sampling."""
    # Load image and label
    img = nib.load(str(image_path))
    label_img = nib.load(str(label_path))
    
    image_data = img.get_fdata()
    label_data = label_img.get_fdata().astype(np.int32)
    
    # Ensure same shape
    if image_data.shape != label_data.shape:
        raise ValueError(f"Shape mismatch: image {image_data.shape} vs label {label_data.shape}")
    
    # Extract intensities per tissue
    tissue_intensities = {}
    
    if tissues_of_interest is None:
        tissues_of_interest = ["background", "pulp", "dentin", "enamel"]
    
    for tissue_name in tissues_of_interest:
        if tissue_name not in label_map:
            continue
        label_id = label_map[tissue_name]
        mask = (label_data == label_id)
        intensities = image_data[mask]
        
        # Sample if too many voxels to save memory
        if len(intensities) > max_voxels_per_tissue:
            # Random sampling
            indices = np.random.choice(len(intensities), max_voxels_per_tissue, replace=False)
            intensities = intensities[indices]
        
        if len(intensities) > 0:
            tissue_intensities[tissue_name] = intensities
    
    return tissue_intensities


def aggregate_intensities_all_files(
    file_pairs: List[Tuple[Path, Path]],
    label_map: Dict[str, int],
    max_files: int = None,
    tissues_of_interest: List[str] = None,
    max_voxels_per_tissue: int = 50000,
    max_total_voxels_per_tissue: int = 1000000,
) -> Dict[str, np.ndarray]:
    """Aggregate intensities across all files with memory-efficient sampling."""
    all_intensities = {tissue: [] for tissue in (tissues_of_interest or ["background", "pulp", "dentin", "enamel"])}
    
    num_files = min(len(file_pairs), max_files) if max_files else len(file_pairs)
    print(f"Processing {num_files} image-label pairs...")
    print(f"Sampling up to {max_voxels_per_tissue:,} voxels per tissue per file")
    print(f"Maximum total: {max_total_voxels_per_tissue:,} voxels per tissue")
    
    for i, (img_path, label_path) in enumerate(file_pairs[:num_files]):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_files} files...")
            # Show current memory usage
            for tissue, values in all_intensities.items():
                if len(values) > 0:
                    print(f"    {tissue}: {len(values):,} voxels collected")
        
        try:
            intensities = extract_intensities_by_tissue(
                img_path, label_path, label_map, tissues_of_interest, max_voxels_per_tissue
            )
            for tissue, values in intensities.items():
                current_count = len(all_intensities[tissue])
                if current_count < max_total_voxels_per_tissue:
                    remaining = max_total_voxels_per_tissue - current_count
                    if len(values) > remaining:
                        # Sample if we're approaching the limit
                        indices = np.random.choice(len(values), remaining, replace=False)
                        values = values[indices]
                    all_intensities[tissue].extend(values.flatten())
        except Exception as e:
            print(f"  Warning: Failed to process {img_path.name}: {e}")
            continue
    
    # Convert to numpy arrays
    result = {}
    for tissue, values in all_intensities.items():
        if len(values) > 0:
            result[tissue] = np.array(values)
            print(f"Final {tissue}: {len(result[tissue]):,} voxels")
    
    return result


def create_grauwert_histogram(
    tissue_intensities: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Grauwertverteilung der verschiedenen Zahngewebe",
    bins: int = 200,
):
    """Create a professional histogram of grayscale value distributions."""
    # German labels for tissues
    tissue_labels_de = {
        "background": "Hintergrund",
        "pulp": "Pulpa",
        "dentin": "Dentin",
        "enamel": "Schmelz",
        "enamel_caries": "Schmelz-Karies",
        "dentin_caries": "Dentin-Karies",
    }
    
    # Colors for each tissue
    tissue_colors = {
        "background": "#2C3E50",  # Dark gray
        "pulp": "#E74C3C",  # Red
        "dentin": "#F39C12",  # Orange
        "enamel": "#ECF0F1",  # Light gray/white
        "enamel_caries": "#8E44AD",  # Purple
        "dentin_caries": "#C0392B",  # Dark red
    }
    
    # Determine global range
    all_values = np.concatenate(list(tissue_intensities.values()))
    min_val = np.percentile(all_values, 0.1)
    max_val = np.percentile(all_values, 99.9)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histograms for each tissue
    alpha = 0.7
    linewidth = 2
    
    for tissue, intensities in tissue_intensities.items():
        if len(intensities) == 0:
            continue
        
        label_de = tissue_labels_de.get(tissue, tissue)
        color = tissue_colors.get(tissue, "#95A5A6")
        
        # Create histogram
        counts, bin_edges = np.histogram(intensities, bins=bins, range=(min_val, max_val))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize to density
        density = counts / (len(intensities) * (bin_edges[1] - bin_edges[0]))
        
        # Plot
        ax.plot(bin_centers, density, label=label_de, color=color, linewidth=linewidth, alpha=alpha)
        ax.fill_between(bin_centers, density, alpha=0.3, color=color)
    
    # Styling
    ax.set_xlabel("Grauwert (CT-Intensität)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dichte (normalisiert)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Add statistics text box
    stats_text = []
    for tissue, intensities in tissue_intensities.items():
        if len(intensities) == 0:
            continue
        label_de = tissue_labels_de.get(tissue, tissue)
        mean_val = np.mean(intensities)
        std_val = np.std(intensities)
        stats_text.append(f"{label_de}: μ={mean_val:.1f}, σ={std_val:.1f}")
    
    stats_str = "\n".join(stats_text)
    ax.text(
        0.02, 0.98, stats_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Histogram saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Grauwert-Statistiken")
    print("=" * 70)
    for tissue, intensities in tissue_intensities.items():
        if len(intensities) == 0:
            continue
        label_de = tissue_labels_de.get(tissue, tissue)
        mean_val = np.mean(intensities)
        std_val = np.std(intensities)
        median_val = np.median(intensities)
        min_val_tissue = np.min(intensities)
        max_val_tissue = np.max(intensities)
        print(f"{label_de:20s}: μ={mean_val:8.2f}, σ={std_val:8.2f}, Median={median_val:8.2f}, Range=[{min_val_tissue:8.2f}, {max_val_tissue:8.2f}]")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Create histogram of CT intensity distributions for different tooth tissues."
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
        help="Output path for histogram (default: images/grauwert_histogramm.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the histogram",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins (default: 200)",
    )
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=["background", "pulp", "dentin", "enamel"],
        help="Tissues to include (default: background pulp dentin enamel)",
    )
    parser.add_argument(
        "--max_voxels_per_file",
        type=int,
        default=50000,
        help="Maximum voxels to sample per tissue per file (default: 50000)",
    )
    parser.add_argument(
        "--max_total_voxels",
        type=int,
        default=1000000,
        help="Maximum total voxels to collect per tissue (default: 1000000)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    nnunet_raw = Path(args.nnunet_raw)
    dataset_path = nnunet_raw / args.dataset
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Load dataset info
    dataset_info = load_dataset_info(dataset_path)
    label_map = dataset_info.get("labels", {})
    
    if not label_map:
        raise ValueError(f"No 'labels' found in dataset.json for {args.dataset}")
    
    print(f"Dataset: {args.dataset}")
    print(f"Labels: {label_map}")
    print(f"Tissues to analyze: {args.tissues}")
    
    # Get matching file pairs
    file_pairs = get_matching_files(images_dir, labels_dir)
    print(f"Found {len(file_pairs)} matching image-label pairs")
    
    # Extract intensities
    tissue_intensities = aggregate_intensities_all_files(
        file_pairs,
        label_map,
        max_files=args.max_files,
        tissues_of_interest=args.tissues,
        max_voxels_per_tissue=args.max_voxels_per_file,
        max_total_voxels_per_tissue=args.max_total_voxels,
    )
    
    if not tissue_intensities:
        raise ValueError("No intensity values extracted. Check that tissues exist in labels.")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path("analysis_results") / "grayscale"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / f"grauwert_histogramm_{args.dataset}_{timestamp}.png"
    
    # Determine title
    title = args.title if args.title else "Grauwertverteilung der verschiedenen Zahngewebe in einem µCT-Datensatz"
    
    # Create histogram
    create_grauwert_histogram(tissue_intensities, output_path, title, bins=args.bins)


if __name__ == "__main__":
    main()
