#!/usr/bin/env python3
"""
Analyze CT intensity statistics for different tooth tissues (tooth grayscale distribution).

Extracts statistics per tissue, prints/saves a report and optionally creates
bar-chart histograms: anatomical (background, pulp, dentin, enamel) and
pathological (enamel_caries, dentin_caries). Output files use the base name
tooth_grayscale_distribution (e.g. tooth_grayscale_distribution.txt, .png).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Tissue groups for separate histograms (same color depth across all)
ANATOMICAL_TISSUES = ["background", "pulp", "dentin", "enamel"]
PATHOLOGICAL_TISSUES = ["enamel_caries", "dentin_caries"]


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
    
    matches = []
    label_dict = {}
    for label_file in label_files:
        base_name = label_file.stem
        label_dict[base_name] = label_file
    
    for img_file in image_files:
        base_name = img_file.stem.replace("_0000", "")
        if base_name in label_dict:
            matches.append((img_file, label_dict[base_name]))
    
    return matches


def extract_intensities_by_tissue(
    image_path: Path,
    label_path: Path,
    label_map: Dict[str, int],
    tissues_of_interest: List[str] = None,
    max_voxels_per_tissue: int = 50000,
) -> Dict[str, np.ndarray]:
    """Extract CT intensities for each tissue type with optional sampling."""
    img = nib.load(str(image_path))
    label_img = nib.load(str(label_path))
    
    image_data = img.get_fdata()
    label_data = label_img.get_fdata().astype(np.int32)
    
    if image_data.shape != label_data.shape:
        raise ValueError(f"Shape mismatch: image {image_data.shape} vs label {label_data.shape}")
    
    tissue_intensities = {}
    
    if tissues_of_interest is None:
        tissues_of_interest = ["background", "pulp", "dentin", "enamel"]
    
    for tissue_name in tissues_of_interest:
        if tissue_name not in label_map:
            continue
        label_id = label_map[tissue_name]
        mask = (label_data == label_id)
        intensities = image_data[mask]
        
        if len(intensities) > max_voxels_per_tissue:
            indices = np.random.choice(len(intensities), max_voxels_per_tissue, replace=False)
            intensities = intensities[indices]
        
        if len(intensities) > 0:
            tissue_intensities[tissue_name] = intensities
    
    return tissue_intensities


def compute_statistics(intensities: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for intensity values."""
    return {
        "mean": float(np.mean(intensities)),
        "median": float(np.median(intensities)),
        "std": float(np.std(intensities)),
        "min": float(np.min(intensities)),
        "max": float(np.max(intensities)),
        "q25": float(np.percentile(intensities, 25)),
        "q75": float(np.percentile(intensities, 75)),
        "q5": float(np.percentile(intensities, 5)),
        "q95": float(np.percentile(intensities, 95)),
    }


def _filter_suppress_zero(values: np.ndarray, suppress_zero: bool) -> np.ndarray:
    """Optionally remove zeros to avoid dominant spike in histogram."""
    if not suppress_zero or len(values) == 0:
        return values
    return values[values != 0]


def create_tooth_grayscale_distribution_histograms(
    all_intensities: Dict[str, List[float]],
    output_path: Path,
    bins: int = 70,
    suppress_zero: bool = True,
    title_base: str = "Grauwertverteilung (Tooth Grayscale Distribution)",
) -> None:
    """
    Create bar-chart histograms: anatomical and (if present) pathological.
    Same value range (color depth) for all panels. Optionally suppress 0 to reduce spikes.
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping histogram plot.")
        return

    tissue_labels_de = {
        "background": "Hintergrund",
        "pulp": "Pulpa",
        "dentin": "Dentin",
        "enamel": "Schmelz",
        "enamel_caries": "Schmelz-Karies",
        "dentin_caries": "Dentin-Karies",
    }
    # Deutlich unterscheidbare Farben; Schmelz nicht hell/weiß (auf weißem Hintergrund schlecht sichtbar)
    tissue_colors = {
        "background": "#2D3748",   # Dunkelgrau
        "pulp": "#E53E3E",         # Rot
        "dentin": "#D69E2E",       # Amber/Gold
        "enamel": "#2B6CB0",       # Kräftiges Blau (statt Weiß)
        "enamel_caries": "#805AD5",  # Violett
        "dentin_caries": "#D53F8C",  # Magenta/Pink
    }

    def get_tissue_arrays(tissues: List[str]) -> Dict[str, np.ndarray]:
        out = {}
        for t in tissues:
            if t not in all_intensities or len(all_intensities[t]) == 0:
                continue
            arr = np.array(all_intensities[t], dtype=np.float64)
            arr = _filter_suppress_zero(arr, suppress_zero)
            if len(arr) > 0:
                out[t] = arr
        return out

    anatomical = get_tissue_arrays(ANATOMICAL_TISSUES)
    pathological_only = get_tissue_arrays(PATHOLOGICAL_TISSUES)
    # Pathologisches Histogramm: anatomische + pathologische Gewebe, damit man sieht,
    # wo sich Karies zwischen den Graustufen einpendelt
    anatomical_and_pathological = dict(anatomical)
    anatomical_and_pathological.update(pathological_only)

    all_arrays = list(anatomical.values()) + list(pathological_only.values())
    if not all_arrays:
        print("Warning: No intensity data for histograms.")
        return

    # Global range (same color depth for all images/panels)
    concat = np.concatenate(all_arrays)
    vmin = float(np.percentile(concat, 0.5))
    vmax = float(np.percentile(concat, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1

    n_plots = 1 if not pathological_only else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    # Zeichenreihenfolge: zuerst anatomisch, dann pathologisch (für Legende und Überlagerung)
    order_anatomical_then_pathological = ANATOMICAL_TISSUES + PATHOLOGICAL_TISSUES

    def draw_bar_histogram(ax, data_dict: Dict[str, np.ndarray], subplot_title: str, order: List[str] = None) -> None:
        tissues = order if order else list(data_dict.keys())
        for tissue in tissues:
            if tissue not in data_dict:
                continue
            values = data_dict[tissue]
            label_de = tissue_labels_de.get(tissue, tissue)
            color = tissue_colors.get(tissue, "#95A5A6")
            counts, bin_edges = np.histogram(values, bins=bins, range=(vmin, vmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = (bin_edges[1] - bin_edges[0]) * 0.85
            ax.bar(bin_centers, counts, width=width, label=label_de, color=color, alpha=0.7, edgecolor="none")
        ax.set_xlabel("Grauwert (CT-Intensität)", fontsize=12)
        ax.set_ylabel("Anzahl Voxel", fontsize=12)
        ax.set_title(subplot_title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(vmin, vmax)

    draw_bar_histogram(axes[0], anatomical, "Anatomisches Histogramm", order=ANATOMICAL_TISSUES)
    if pathological_only:
        draw_bar_histogram(
            axes[1],
            anatomical_and_pathological,
            "Pathologisches Histogramm (anatomisch + Karies)",
            order=order_anatomical_then_pathological,
        )

    fig.suptitle(title_base, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CT intensity statistics for different tooth tissues."
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
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
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
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=["background", "pulp", "dentin", "enamel", "enamel_caries", "dentin_caries"],
        help="Tissues to include (default: anatomical + enamel_caries dentin_caries)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file path (default: analysis_results/grayscale/tooth_grayscale_distribution.txt)",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Output histogram plot path (default: analysis_results/grayscale/tooth_grayscale_distribution.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=70,
        choices=[50, 70, 100],
        help="Number of histogram bins (default: 70, fewer bins reduce spikes)",
    )
    parser.add_argument(
        "--no_suppress_zero",
        action="store_true",
        help="Do not suppress zero in histograms (default: suppress 0 to reduce background spike)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not create histogram plot, only statistics and text report",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save file, only print to stdout",
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
    print(f"Tissues to analyze: {args.tissues}")
    print("=" * 80)
    
    # Get matching file pairs
    file_pairs = get_matching_files(images_dir, labels_dir)
    num_files = min(len(file_pairs), args.max_files) if args.max_files else len(file_pairs)
    print(f"Processing {num_files} image-label pairs...")
    print("=" * 80)
    
    # Aggregate intensities
    all_intensities = {tissue: [] for tissue in args.tissues}
    
    for i, (img_path, label_path) in enumerate(file_pairs[:num_files]):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_files} files...")
        
        try:
            intensities = extract_intensities_by_tissue(
                img_path, label_path, label_map, args.tissues, args.max_voxels_per_file
            )
            for tissue, values in intensities.items():
                current_count = len(all_intensities[tissue])
                if current_count < args.max_total_voxels:
                    remaining = args.max_total_voxels - current_count
                    if len(values) > remaining:
                        indices = np.random.choice(len(values), remaining, replace=False)
                        values = values[indices]
                    all_intensities[tissue].extend(values.flatten())
        except Exception as e:
            print(f"  Warning: Failed to process {img_path.name}: {e}")
            continue
    
    # Convert to numpy arrays and compute statistics
    print("\n" + "=" * 80)
    print("GRAUWERT-STATISTIKEN")
    print("=" * 80)
    
    tissue_labels_de = {
        "background": "Hintergrund",
        "pulp": "Pulpa",
        "dentin": "Dentin",
        "enamel": "Schmelz",
        "enamel_caries": "Schmelz-Karies",
        "dentin_caries": "Dentin-Karies",
    }
    
    results = {}
    for tissue, values in all_intensities.items():
        if len(values) == 0:
            continue
        arr = np.array(values)
        stats = compute_statistics(arr)
        results[tissue] = stats
        
        label_de = tissue_labels_de.get(tissue, tissue)
        print(f"\n{label_de} ({tissue}):")
        print(f"  Anzahl Voxel: {len(arr):,}")
        print(f"  Mittelwert (μ):     {stats['mean']:10.2f}")
        print(f"  Median:             {stats['median']:10.2f}")
        print(f"  Standardabw. (σ):   {stats['std']:10.2f}")
        print(f"  Minimum:            {stats['min']:10.2f}")
        print(f"  Maximum:            {stats['max']:10.2f}")
        print(f"  5. Perzentil:       {stats['q5']:10.2f}")
        print(f"  25. Perzentil:      {stats['q25']:10.2f}")
        print(f"  75. Perzentil:      {stats['q75']:10.2f}")
        print(f"  95. Perzentil:      {stats['q95']:10.2f}")
    
    # Evaluate plausibility
    print("\n" + "=" * 80)
    print("BEWERTUNG DER ERGEBNISSE")
    print("=" * 80)
    
    if len(results) < 2:
        print("⚠️  Zu wenige Gewebe gefunden für eine Bewertung.")
        if not args.no_save:
            analysis_dir = Path("analysis_results") / "grayscale"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = analysis_dir / "tooth_grayscale_distribution.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"GRAUWERT-STATISTIKEN ANALYSE\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write("⚠️  Zu wenige Gewebe gefunden für eine Bewertung.\n")
            print(f"\n✓ Statistics saved to: {output_path}")
        return
    
    # Expected order: background < pulp < dentin < enamel (by mean intensity)
    expected_order = ["background", "pulp", "dentin", "enamel"]
    found_tissues = [t for t in expected_order if t in results]
    
    if len(found_tissues) >= 2:
        means = {t: results[t]["mean"] for t in found_tissues}
        sorted_tissues = sorted(found_tissues, key=lambda t: means[t])
        
        print(f"\nReihenfolge nach mittlerem Grauwert (niedrig → hoch):")
        for i, tissue in enumerate(sorted_tissues):
            label_de = tissue_labels_de.get(tissue, tissue)
            print(f"  {i+1}. {label_de:20s}: μ = {means[tissue]:8.2f}")
        
        # Check if order is plausible
        expected_means = [means.get(t, None) for t in expected_order if t in means]
        is_sorted = all(expected_means[i] <= expected_means[i+1] for i in range(len(expected_means)-1))
        
        if is_sorted:
            print("\n✅ Die Reihenfolge ist plausibel!")
            print("   (Hintergrund < Pulpa < Dentin < Schmelz)")
        else:
            print("\n⚠️  Die Reihenfolge weicht von der erwarteten ab.")
            print("   Erwartet: Hintergrund < Pulpa < Dentin < Schmelz")
        
        # Check overlaps
        print(f"\nÜberlappungsbereiche (5%-95% Perzentile):")
        for tissue in found_tissues:
            label_de = tissue_labels_de.get(tissue, tissue)
            q5 = results[tissue]["q5"]
            q95 = results[tissue]["q95"]
            print(f"  {label_de:20s}: [{q5:8.2f}, {q95:8.2f}]")
        
        # Check separation
        print(f"\nTrennung zwischen Geweben (Abstand der Mittelwerte):")
        for i in range(len(sorted_tissues) - 1):
            t1, t2 = sorted_tissues[i], sorted_tissues[i+1]
            label1 = tissue_labels_de.get(t1, t1)
            label2 = tissue_labels_de.get(t2, t2)
            diff = means[t2] - means[t1]
            std1 = results[t1]["std"]
            std2 = results[t2]["std"]
            avg_std = (std1 + std2) / 2
            separation = diff / avg_std if avg_std > 0 else 0
            print(f"  {label1:20s} → {label2:20s}: Δμ = {diff:8.2f} ({separation:.2f} × σ)")
            
            if separation > 2:
                print(f"    ✅ Gute Trennung (>2σ)")
            elif separation > 1:
                print(f"    ⚠️  Moderate Trennung (1-2σ)")
            else:
                print(f"    ❌ Schwache Trennung (<1σ)")
    
    print("\n" + "=" * 80)
    print("Typische Erwartungen für µCT-Grauwerte:")
    print("  - Hintergrund (Luft): sehr niedrig (oft < 0)")
    print("  - Pulpa (weiches Gewebe): niedrig-mittel")
    print("  - Dentin (mineralisiert): mittel-hoch")
    print("  - Schmelz (stark mineralisiert): sehr hoch")
    print("=" * 80)
    
    # Save to file
    if not args.no_save:
        analysis_dir = Path("analysis_results") / "grayscale"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = analysis_dir / "tooth_grayscale_distribution.txt"
        
        # Generate report text
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"GRAUWERT-STATISTIKEN ANALYSE")
        report_lines.append(f"Dataset: {args.dataset}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTissues analyzed: {', '.join(args.tissues)}")
        report_lines.append(f"Files processed: {num_files}")
        report_lines.append("\n" + "=" * 80)
        report_lines.append("GRAUWERT-STATISTIKEN")
        report_lines.append("=" * 80)
        
        for tissue, stats in results.items():
            label_de = tissue_labels_de.get(tissue, tissue)
            report_lines.append(f"\n{label_de} ({tissue}):")
            report_lines.append(f"  Anzahl Voxel: {len(all_intensities[tissue]):,}")
            report_lines.append(f"  Mittelwert (μ):     {stats['mean']:10.2f}")
            report_lines.append(f"  Median:             {stats['median']:10.2f}")
            report_lines.append(f"  Standardabw. (σ):   {stats['std']:10.2f}")
            report_lines.append(f"  Minimum:            {stats['min']:10.2f}")
            report_lines.append(f"  Maximum:            {stats['max']:10.2f}")
            report_lines.append(f"  5. Perzentil:       {stats['q5']:10.2f}")
            report_lines.append(f"  25. Perzentil:      {stats['q25']:10.2f}")
            report_lines.append(f"  75. Perzentil:      {stats['q75']:10.2f}")
            report_lines.append(f"  95. Perzentil:      {stats['q95']:10.2f}")
        
        if len(results) >= 2:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("BEWERTUNG DER ERGEBNISSE")
            report_lines.append("=" * 80)
            
            expected_order = ["background", "pulp", "dentin", "enamel"]
            found_tissues = [t for t in expected_order if t in results]
            
            if len(found_tissues) >= 2:
                means = {t: results[t]["mean"] for t in found_tissues}
                sorted_tissues = sorted(found_tissues, key=lambda t: means[t])
                
                report_lines.append(f"\nReihenfolge nach mittlerem Grauwert (niedrig → hoch):")
                for i, tissue in enumerate(sorted_tissues):
                    label_de = tissue_labels_de.get(tissue, tissue)
                    report_lines.append(f"  {i+1}. {label_de:20s}: μ = {means[tissue]:8.2f}")
                
                expected_means = [means.get(t, None) for t in expected_order if t in means]
                is_sorted = all(expected_means[i] <= expected_means[i+1] for i in range(len(expected_means)-1))
                
                if is_sorted:
                    report_lines.append("\n✅ Die Reihenfolge ist plausibel!")
                    report_lines.append("   (Hintergrund < Pulpa < Dentin < Schmelz)")
                else:
                    report_lines.append("\n⚠️  Die Reihenfolge weicht von der erwarteten ab.")
                    report_lines.append("   Erwartet: Hintergrund < Pulpa < Dentin < Schmelz")
                
                report_lines.append(f"\nÜberlappungsbereiche (5%-95% Perzentile):")
                for tissue in found_tissues:
                    label_de = tissue_labels_de.get(tissue, tissue)
                    q5 = results[tissue]["q5"]
                    q95 = results[tissue]["q95"]
                    report_lines.append(f"  {label_de:20s}: [{q5:8.2f}, {q95:8.2f}]")
                
                report_lines.append(f"\nTrennung zwischen Geweben (Abstand der Mittelwerte):")
                for i in range(len(sorted_tissues) - 1):
                    t1, t2 = sorted_tissues[i], sorted_tissues[i+1]
                    label1 = tissue_labels_de.get(t1, t1)
                    label2 = tissue_labels_de.get(t2, t2)
                    diff = means[t2] - means[t1]
                    std1 = results[t1]["std"]
                    std2 = results[t2]["std"]
                    avg_std = (std1 + std2) / 2
                    separation = diff / avg_std if avg_std > 0 else 0
                    report_lines.append(f"  {label1:20s} → {label2:20s}: Δμ = {diff:8.2f} ({separation:.2f} × σ)")
                    
                    if separation > 2:
                        report_lines.append(f"    ✅ Gute Trennung (>2σ)")
                    elif separation > 1:
                        report_lines.append(f"    ⚠️  Moderate Trennung (1-2σ)")
                    else:
                        report_lines.append(f"    ❌ Schwache Trennung (<1σ)")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("Typische Erwartungen für µCT-Grauwerte:")
        report_lines.append("  - Hintergrund (Luft): sehr niedrig (oft < 0)")
        report_lines.append("  - Pulpa (weiches Gewebe): niedrig-mittel")
        report_lines.append("  - Dentin (mineralisiert): mittel-hoch")
        report_lines.append("  - Schmelz (stark mineralisiert): sehr hoch")
        report_lines.append("=" * 80)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print(f"\n✓ Statistics saved to: {output_path}")

        # Histogram plot (tooth_grayscale_distribution.png)
        if not args.no_plot and plt is not None:
            if args.output_plot:
                plot_path = Path(args.output_plot)
            else:
                plot_path = analysis_dir / "tooth_grayscale_distribution.png"
            create_tooth_grayscale_distribution_histograms(
                all_intensities,
                plot_path,
                bins=args.bins,
                suppress_zero=not args.no_suppress_zero,
                title_base=f"Grauwertverteilung – {args.dataset} (Tooth Grayscale Distribution)",
            )


if __name__ == "__main__":
    main()
