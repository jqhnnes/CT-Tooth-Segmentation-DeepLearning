#!/usr/bin/env python3
"""
Create a validation subset from nnUNet imagesTr/labelsTr folders.

Example:
    python scripts/utils/create_validation_subset.py \
        --dataset Dataset001_GroundTruth \
        --num_cases 50 \
        --target prediction_data
"""

import argparse
import shutil
from pathlib import Path


def copy_cases(src_root: Path, dst_root: Path, case_ids: list[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for case in case_ids:
        src_path = src_root / case
        if src_path.is_file():
            shutil.copy2(src_path, dst_root / case)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create validation subset from nnUNet imagesTr/labelsTr.")
    parser.add_argument(
        "--dataset",
        default="Dataset001_GroundTruth",
        help="Dataset folder under data/nnUNet_raw (default: Dataset001_GroundTruth)",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=50,
        help="Number of cases to copy into the subset (default: 50).",
    )
    parser.add_argument(
        "--target",
        default="prediction_data",
        help="Name of the subset folder to create under the dataset (default: prediction_data).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "data" / "nnUNet_raw" / args.dataset
    images_src = dataset_root / "imagesTr"
    labels_src = dataset_root / "labelsTr"
    pred_subdir = (
        project_root
        / "data"
        / "nnUNet_results"
        / args.dataset
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / args.target
    )

    if not images_src.is_dir():
        raise SystemExit(f"imagesTr not found at {images_src}")

    case_names = sorted([f.name for f in images_src.iterdir() if f.is_file()])[: args.num_cases]
    if not case_names:
        raise SystemExit("No cases found to copy.")

    # Copy subset for prediction data (imagesTr/labelsTr)
    images_dst = pred_subdir / "imagesTr"
    labels_dst = pred_subdir / "labelsTr"
    copy_cases(images_src, images_dst, case_names)
    if labels_src.is_dir():
        copy_cases(labels_src, labels_dst, case_names)
        # Remove copied cases from labelsTr so they are not used in training
        for case in case_names:
            src_lbl = labels_src / case
            if src_lbl.exists():
                src_lbl.unlink()
    # Remove copied images from imagesTr
    for case in case_names:
        src_img = images_src / case
        if src_img.exists():
            src_img.unlink()

    # Place subset also under dataset/imagesTs for predict/ensemble convenience
    images_ts = dataset_root / "imagesTs"
    images_ts.mkdir(exist_ok=True)
    for case in case_names:
        src = images_dst / case
        shutil.copy2(src, images_ts / case)

    print(
        f"[OK] Moved {len(case_names)} cases from imagesTr/labelsTr into "
        f"{pred_subdir}, and copied them to {images_ts}"
    )


if __name__ == "__main__":
    main()

