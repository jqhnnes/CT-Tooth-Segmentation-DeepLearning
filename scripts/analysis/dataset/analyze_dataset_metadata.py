#!/usr/bin/env python3
"""
Extract comprehensive metadata and statistics from a nnU-Net dataset.

This script analyzes the dataset structure, image properties, and provides
statistics suitable for documentation in academic papers.

Example:
    python scripts/analysis/dataset/analyze_dataset_metadata.py --dataset Dataset002_Karies
    python scripts/analysis/dataset/analyze_dataset_metadata.py --dataset Dataset001_GroundTruth --output dataset_metadata.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np

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


def analyze_image_properties(image_path: Path) -> Dict:
    """Extract image properties (spacing, dimensions, etc.)."""
    img = nib.load(str(image_path))
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    
    # Get spacing from affine matrix (diagonal elements)
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    
    # Get intensity statistics
    intensity_stats = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
    }
    
    return {
        "shape": list(data.shape),
        "spacing_mm": spacing.tolist(),
        "spacing_um": (spacing * 1000).tolist(),  # Convert to micrometers
        "voxel_count": int(np.prod(data.shape)),
        "intensity_stats": intensity_stats,
        "data_type": str(data.dtype),
    }


def analyze_label_properties(label_path: Path, label_map: Dict[str, int]) -> Dict:
    """Extract label properties and distribution."""
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.int32)
    
    # Count voxels per label
    unique, counts = np.unique(data, return_counts=True)
    label_counts = {int(label_id): int(count) for label_id, count in zip(unique, counts)}
    
    # Create reverse mapping
    id_to_name = {v: k for k, v in label_map.items()}
    
    # Calculate percentages
    total_voxels = np.prod(data.shape)
    label_percentages = {
        id_to_name.get(lid, f"Unknown_{lid}"): (count / total_voxels * 100)
        for lid, count in label_counts.items()
    }
    
    return {
        "shape": list(data.shape),
        "label_counts": label_counts,
        "label_percentages": label_percentages,
        "unique_labels": sorted([int(lid) for lid in unique]),
    }


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


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract comprehensive metadata from a nnU-Net dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Dataset002_Karies)",
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
        help="Output JSON file path (default: analysis_results/datasets/dataset_metadata_<dataset>_<timestamp>.json)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save files, only print to stdout",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to analyze (default: all)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed per-file statistics",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    nnunet_raw = Path(args.nnunet_raw)
    dataset_path = nnunet_raw / args.dataset
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"
    images_ts_dir = dataset_path / "imagesTs" if (dataset_path / "imagesTs").exists() else None
    labels_ts_dir = dataset_path / "labelsTs" if (dataset_path / "labelsTs").exists() else None
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Load dataset info
    dataset_info = load_dataset_info(dataset_path)
    label_map = dataset_info.get("labels", {})
    num_training = dataset_info.get("numTraining", 0)
    file_ending = dataset_info.get("file_ending", ".nii.gz")
    
    print(f"Analyzing dataset: {args.dataset}")
    print("=" * 80)
    
    # Analyze training data
    metadata = {
        "dataset_name": args.dataset,
        "dataset_info": {
            "num_training": num_training,
            "file_ending": file_ending,
            "labels": label_map,
            "channel_names": dataset_info.get("channel_names", {}),
        },
    }
    
    if images_dir.exists() and labels_dir.exists():
        file_pairs = get_matching_files(images_dir, labels_dir)
        num_files = min(len(file_pairs), args.max_files) if args.max_files else len(file_pairs)
        
        print(f"\nTraining Data (imagesTr/labelsTr):")
        print(f"  Found {len(file_pairs)} matching image-label pairs")
        print(f"  Analyzing {num_files} files...")
        
        # Collect statistics
        spacings_x = []
        spacings_y = []
        spacings_z = []
        shapes = []
        voxel_counts = []
        all_label_counts = defaultdict(int)
        per_file_stats = [] if args.detailed else None
        
        for i, (img_path, label_path) in enumerate(file_pairs[:num_files]):
            if (i + 1) % 20 == 0:
                print(f"    Processing {i + 1}/{num_files}...")
            
            try:
                img_props = analyze_image_properties(img_path)
                label_props = analyze_label_properties(label_path, label_map)
                
                spacings_x.append(img_props["spacing_um"][0])
                spacings_y.append(img_props["spacing_um"][1])
                spacings_z.append(img_props["spacing_um"][2])
                shapes.append(img_props["shape"])
                voxel_counts.append(img_props["voxel_count"])
                
                for lid, count in label_props["label_counts"].items():
                    all_label_counts[lid] += count
                
                if args.detailed:
                    per_file_stats.append({
                        "file": img_path.name,
                        "image_properties": img_props,
                        "label_properties": label_props,
                    })
            except Exception as e:
                print(f"    Warning: Failed to process {img_path.name}: {e}")
                continue
        
        # Compute aggregated statistics
        metadata["training_data"] = {
            "num_files": num_files,
            "spacing_statistics_um": {
                "x": compute_statistics(spacings_x),
                "y": compute_statistics(spacings_y),
                "z": compute_statistics(spacings_z),
            },
            "dimension_statistics": {
                "width": compute_statistics([s[0] for s in shapes]),
                "height": compute_statistics([s[1] for s in shapes]),
                "depth": compute_statistics([s[2] for s in shapes]),
            },
            "voxel_count_statistics": compute_statistics(voxel_counts),
            "label_distribution": {
                "total_voxels": sum(all_label_counts.values()),
                "voxels_per_label": dict(all_label_counts),
                "percentages": {
                    label_map.get(lid, f"Unknown_{lid}"): (count / sum(all_label_counts.values()) * 100)
                    for lid, count in all_label_counts.items()
                },
            },
        }
        
        if per_file_stats:
            metadata["training_data"]["per_file_statistics"] = per_file_stats
    
    # Analyze test data if available
    if images_ts_dir and images_ts_dir.exists():
        test_images = sorted(images_ts_dir.glob("*.nii.gz"))
        metadata["test_data"] = {
            "num_files": len(test_images),
            "files": [f.name for f in test_images],
        }
        print(f"\nTest Data (imagesTs):")
        print(f"  Found {len(test_images)} test images")
    
    if labels_ts_dir and labels_ts_dir.exists():
        test_labels = sorted(labels_ts_dir.glob("*.nii.gz"))
        if "test_data" not in metadata:
            metadata["test_data"] = {}
        metadata["test_data"]["num_label_files"] = len(test_labels)
        metadata["test_data"]["label_files"] = [f.name for f in test_labels]
        print(f"  Found {len(test_labels)} test label files")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Training samples: {metadata['dataset_info']['num_training']}")
    print(f"File format: {file_ending}")
    
    if "training_data" in metadata:
        td = metadata["training_data"]
        print(f"\nAnalyzed files: {td['num_files']}")
        
        # Spacing
        sp_x = td["spacing_statistics_um"]["x"]
        sp_y = td["spacing_statistics_um"]["y"]
        sp_z = td["spacing_statistics_um"]["z"]
        print(f"\nVoxel spacing (µm):")
        print(f"  X: {sp_x['mean']:.2f} ± {sp_x['std']:.2f} (range: {sp_x['min']:.2f} - {sp_x['max']:.2f})")
        print(f"  Y: {sp_y['mean']:.2f} ± {sp_y['std']:.2f} (range: {sp_y['min']:.2f} - {sp_y['max']:.2f})")
        print(f"  Z: {sp_z['mean']:.2f} ± {sp_z['std']:.2f} (range: {sp_z['min']:.2f} - {sp_z['max']:.2f})")
        
        # Dimensions
        dim = td["dimension_statistics"]
        print(f"\nImage dimensions (voxels):")
        print(f"  Width:  {dim['width']['mean']:.0f} ± {dim['width']['std']:.0f} (range: {dim['width']['min']:.0f} - {dim['width']['max']:.0f})")
        print(f"  Height: {dim['height']['mean']:.0f} ± {dim['height']['std']:.0f} (range: {dim['height']['min']:.0f} - {dim['height']['max']:.0f})")
        print(f"  Depth:   {dim['depth']['mean']:.0f} ± {dim['depth']['std']:.0f} (range: {dim['depth']['min']:.0f} - {dim['depth']['max']:.0f})")
        
        # Label distribution
        print(f"\nLabel distribution:")
        for label_name, percentage in td["label_distribution"]["percentages"].items():
            count = td["label_distribution"]["voxels_per_label"].get(
                label_map.get(label_name, -1), 0
            )
            print(f"  {label_name:20s}: {percentage:6.2f}% ({count:,} voxels)")
    
    if "test_data" in metadata:
        print(f"\nTest data: {metadata['test_data']['num_files']} images")
    
    # Save files to analysis_results/datasets
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path("analysis_results") / "datasets"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        if args.output:
            json_path = Path(args.output)
        else:
            json_path = analysis_dir / f"dataset_metadata_{args.dataset}_{timestamp}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ JSON metadata saved to: {json_path}")
        
        # Save text report
        report_path = analysis_dir / f"dataset_metadata_{args.dataset}_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"DATASET METADATA REPORT: {args.dataset}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Training samples: {metadata['dataset_info']['num_training']}\n")
            f.write(f"File format: {file_ending}\n\n")
            
            if "training_data" in metadata:
                td = metadata["training_data"]
                f.write(f"Analyzed files: {td['num_files']}\n\n")
                
                # Spacing
                sp_x = td["spacing_statistics_um"]["x"]
                sp_y = td["spacing_statistics_um"]["y"]
                sp_z = td["spacing_statistics_um"]["z"]
                f.write("Voxel spacing (µm):\n")
                f.write(f"  X: {sp_x['mean']:.2f} ± {sp_x['std']:.2f} (range: {sp_x['min']:.2f} - {sp_x['max']:.2f})\n")
                f.write(f"  Y: {sp_y['mean']:.2f} ± {sp_y['std']:.2f} (range: {sp_y['min']:.2f} - {sp_y['max']:.2f})\n")
                f.write(f"  Z: {sp_z['mean']:.2f} ± {sp_z['std']:.2f} (range: {sp_z['min']:.2f} - {sp_z['max']:.2f})\n\n")
                
                # Dimensions
                dim = td["dimension_statistics"]
                f.write("Image dimensions (voxels):\n")
                f.write(f"  Width:  {dim['width']['mean']:.0f} ± {dim['width']['std']:.0f} (range: {dim['width']['min']:.0f} - {dim['width']['max']:.0f})\n")
                f.write(f"  Height: {dim['height']['mean']:.0f} ± {dim['height']['std']:.0f} (range: {dim['height']['min']:.0f} - {dim['height']['max']:.0f})\n")
                f.write(f"  Depth:   {dim['depth']['mean']:.0f} ± {dim['depth']['std']:.0f} (range: {dim['depth']['min']:.0f} - {dim['depth']['max']:.0f})\n\n")
                
                # Label distribution
                f.write("Label distribution:\n")
                for label_name, percentage in td["label_distribution"]["percentages"].items():
                    count = td["label_distribution"]["voxels_per_label"].get(
                        label_map.get(label_name, -1), 0
                    )
                    f.write(f"  {label_name:20s}: {percentage:6.2f}% ({count:,} voxels)\n")
            
            if "test_data" in metadata:
                f.write(f"\nTest data: {metadata['test_data']['num_files']} images\n")
        
        print(f"✓ Text report saved to: {report_path}")
    else:
        print("\n" + "=" * 80)
        print("Full metadata (JSON):")
        print("=" * 80)
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
