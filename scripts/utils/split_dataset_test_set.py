#!/usr/bin/env python3
"""
Split an nnUNet dataset into training and test set, moving 10% to imagesTs/labelsTs.
Mirrors the structure used for Dataset001_GroundTruth.

Example:
    python scripts/utils/split_dataset_test_set.py --dataset Dataset002_Karies --test_fraction 0.1
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def get_case_ids(images_dir: Path, file_ending: str = ".nii.gz") -> list[str]:
    """Extract unique case IDs from nnU-Net image filenames.

    Strips the ``_0000`` channel suffix so that ``AW001-C0005278_0000.nii.gz``
    becomes ``AW001-C0005278``.

    Args:
        images_dir: Directory containing the ``_0000`` image files.
        file_ending: File extension to look for (default: ``".nii.gz"``).

    Returns:
        Sorted list of unique case ID strings.
    """
    case_ids = set()
    for f in images_dir.iterdir():
        if f.is_file() and f.name.endswith(file_ending) and "_0000" in f.name:
            case_id = f.name.replace("_0000" + file_ending, "")
            case_ids.add(case_id)
    return sorted(case_ids)


def main() -> None:
    """Entry point: parse arguments and perform the train/test split."""
    parser = argparse.ArgumentParser(
        description="Split nnUNet dataset: move test_fraction of cases to imagesTs/labelsTs."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset002_Karies",
        help="Dataset folder under data/nnUNet_raw",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.1,
        help="Fraction of cases for test set (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be done, do not move files",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "data" / "nnUNet_raw" / args.dataset
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"

    if not images_tr.is_dir():
        raise SystemExit(f"imagesTr not found at {images_tr}")
    if not labels_tr.is_dir():
        raise SystemExit(f"labelsTr not found at {labels_tr}")

    # Load dataset.json for file_ending
    dataset_json_path = dataset_root / "dataset.json"
    if dataset_json_path.exists():
        with open(dataset_json_path) as f:
            dataset_json = json.load(f)
        file_ending = dataset_json.get("file_ending", ".nii.gz")
    else:
        file_ending = ".nii.gz"

    case_ids = get_case_ids(images_tr, file_ending)
    if not case_ids:
        raise SystemExit("No cases found in imagesTr.")

    # Filter to cases that have labels
    case_ids_with_labels = [
        c for c in case_ids
        if (labels_tr / f"{c}{file_ending}").exists()
    ]
    if len(case_ids_with_labels) < len(case_ids):
        missing = set(case_ids) - set(case_ids_with_labels)
        print(f"[WARN] {len(missing)} cases without labels, skipping: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    case_ids = case_ids_with_labels
    n_total = len(case_ids)
    n_test = max(1, int(round(n_total * args.test_fraction)))
    n_train = n_total - n_test

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    test_indices = set(perm[:n_test])
    test_cases = [case_ids[i] for i in sorted(test_indices)]

    print(f"Dataset: {args.dataset}")
    print(f"Total cases: {n_total}")
    print(f"Test set: {n_test} ({100 * n_test / n_total:.1f}%)")
    print(f"Training set: {n_train}")
    print(f"Test cases: {test_cases}")

    if args.dry_run:
        print("\n[DRY RUN] No files moved.")
        return

    # Create output dirs
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    # Check if imagesTs already has content (would overwrite)
    existing_ts = list(images_ts.glob("*_0000" + file_ending))
    if existing_ts:
        raise SystemExit(
            f"imagesTs already contains {len(existing_ts)} files. "
            "Remove or backup before running, or use a different dataset."
        )

    # Move files
    for case_id in test_cases:
        img_src = images_tr / f"{case_id}_0000{file_ending}"
        lbl_src = labels_tr / f"{case_id}{file_ending}"
        img_dst = images_ts / img_src.name
        lbl_dst = labels_ts / lbl_src.name

        if img_src.exists():
            shutil.move(str(img_src), str(img_dst))
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(lbl_dst))

    # Update dataset.json
    if dataset_json_path.exists():
        dataset_json["numTraining"] = n_train
        with open(dataset_json_path, "w") as f:
            json.dump(dataset_json, f, indent=4)
        print(f"\n[OK] Updated dataset.json: numTraining = {n_train}")

    print(f"\n[OK] Moved {n_test} cases to imagesTs/labelsTs. Training set: {n_train} cases.")


if __name__ == "__main__":
    main()
