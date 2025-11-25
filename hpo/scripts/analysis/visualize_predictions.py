#!/usr/bin/env python3
"""
Visualize model predictions for qualitative evaluation.

This script creates visualizations comparing ground truth, predictions,
and input images for selected cases. Can be used for:
1. Training/validation set cases (for debugging)
2. Test set cases (for final evaluation)

Usage:
    # Visualize validation predictions
    python hpo/scripts/analysis/visualize_predictions.py \
        --trial trial_8 \
        --dataset Dataset001_GroundTruth \
        --source validation \
        --cases AW062-C0005656 AW123-C0005733 \
        --output_dir hpo/analysis/visualizations
    
    # Visualize test set predictions
    python hpo/scripts/analysis/visualize_predictions.py \
        --trial trial_8 \
        --source test \
        --output_dir hpo/analysis/visualizations
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nnunet_env import load_env

load_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model predictions for qualitative evaluation."
    )
    parser.add_argument(
        "--trial",
        default="trial_8",
        help="Trial name to visualize (default: trial_8).",
    )
    parser.add_argument(
        "--dataset_name",
        default="Dataset001_GroundTruth",
        help="Dataset name (default: Dataset001_GroundTruth).",
    )
    parser.add_argument(
        "--source",
        choices=["validation", "test", "train"],
        default="validation",
        help="Source of predictions: validation (from training), test (test set), or train (training set).",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Specific case IDs to visualize (e.g., AW062-C0005656). If not specified, visualizes all available cases.",
    )
    parser.add_argument(
        "--fold",
        default="0",
        help="Fold number (default: 0).",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=5,
        help="Number of slices to visualize per case (default: 5).",
    )
    parser.add_argument(
        "--output_dir",
        default="hpo/analysis/visualizations",
        help="Output directory for visualizations (default: hpo/analysis/visualizations).",
    )
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration (default: 3d_fullres).",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Trainer name (default: nnUNetTrainer).",
    )
    parser.add_argument(
        "--plans_name",
        default="nnUNetPlans",
        help="Plans name (default: nnUNetPlans).",
    )
    return parser.parse_args()


def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file and return numpy array."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    img = nib.load(str(file_path))
    return img.get_fdata()


def find_prediction_dir(
    trial_name: str,
    dataset_name: str,
    source: str,
    fold: str,
    configuration: str,
    trainer: str,
    plans_name: str,
) -> Path:
    """Find the directory containing predictions."""
    results_root = Path("hpo") / "training_output" / trial_name / "nnUNet_results" / dataset_name
    pred_dir = results_root / f"{trainer}__{plans_name}__{configuration}" / f"fold_{fold}" / source
    
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {pred_dir}\n"
            f"Make sure predictions exist. For test set, run:\n"
            f"  python hpo/scripts/analysis/compare_top3_trials.py --testset labelsTs"
        )
    
    return pred_dir


def find_ground_truth_dir(dataset_name: str, source: str) -> Path:
    """Find the directory containing ground truth labels."""
    if source == "validation":
        # Validation predictions are compared against labelsTr
        gt_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "labelsTr"
    elif source == "test":
        gt_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "labelsTs"
    elif source == "train":
        gt_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "labelsTr"
    else:
        raise ValueError(f"Unknown source: {source}")
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    return gt_dir


def find_image_dir(dataset_name: str, source: str) -> Path:
    """Find the directory containing input images."""
    if source == "validation":
        img_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "imagesTr"
    elif source == "test":
        img_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "imagesTs"
    elif source == "train":
        img_dir = Path(os.environ["nnUNet_raw"]) / dataset_name / "imagesTr"
    else:
        raise ValueError(f"Unknown source: {source}")
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    return img_dir


def get_case_list(pred_dir: Path, cases: Optional[List[str]]) -> List[str]:
    """Get list of cases to visualize."""
    if cases:
        return cases
    
    # Get all cases from prediction directory
    pred_files = list(pred_dir.glob("*.nii.gz"))
    case_list = [f.stem.replace("_0000", "") for f in pred_files]
    return sorted(case_list)


def visualize_case(
    case_id: str,
    pred_dir: Path,
    gt_dir: Path,
    img_dir: Path,
    output_dir: Path,
    num_slices: int,
    dataset_name: str,
):
    """Create visualization for a single case."""
    # Find files
    pred_file = pred_dir / f"{case_id}_0000.nii.gz"
    if not pred_file.exists():
        # Try without _0000 suffix
        pred_file = pred_dir / f"{case_id}.nii.gz"
    
    gt_file = gt_dir / f"{case_id}.nii.gz"
    img_file = img_dir / f"{case_id}_0000.nii.gz"
    
    if not pred_file.exists():
        print(f"[WARN] Prediction not found for {case_id}, skipping...")
        return
    
    if not gt_file.exists():
        print(f"[WARN] Ground truth not found for {case_id}, skipping...")
        return
    
    if not img_file.exists():
        print(f"[WARN] Image not found for {case_id}, skipping...")
        return
    
    # Load data
    pred = load_nifti(pred_file)
    gt = load_nifti(gt_file)
    img = load_nifti(img_file)
    
    # Get middle slices
    depth = pred.shape[2]
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slice_indices):
        # Input image
        axes[i, 0].imshow(img[:, :, slice_idx], cmap='gray')
        axes[i, 0].set_title(f'Input Image (Slice {slice_idx})', fontsize=10)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(gt[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
        axes[i, 1].set_title(f'Ground Truth (Slice {slice_idx})', fontsize=10)
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
        axes[i, 2].set_title(f'Prediction (Slice {slice_idx})', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Case: {case_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"{case_id}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("VISUALIZING PREDICTIONS")
    print("=" * 80)
    print(f"Trial: {args.trial}")
    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print()
    
    # Find directories
    pred_dir = find_prediction_dir(
        args.trial,
        args.dataset_name,
        args.source,
        args.fold,
        args.configuration,
        args.trainer,
        args.plans_name,
    )
    
    gt_dir = find_ground_truth_dir(args.dataset_name, args.source)
    img_dir = find_image_dir(args.dataset_name, args.source)
    
    print(f"Prediction directory: {pred_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Image directory: {img_dir}")
    print()
    
    # Get cases
    cases = get_case_list(pred_dir, args.cases)
    print(f"Visualizing {len(cases)} cases...")
    print()
    
    # Visualize each case
    for case_id in cases:
        try:
            visualize_case(
                case_id,
                pred_dir,
                gt_dir,
                img_dir,
                output_dir,
                args.slices,
                args.dataset_name,
            )
        except Exception as e:
            print(f"[ERROR] Failed to visualize {case_id}: {e}")
            continue
    
    print()
    print("=" * 80)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

