#!/usr/bin/env python3
"""
Durchsucht alle Trials in hpo/preprocessing_output/<Dataset>/trial_X nach
Segmentierungen, die Labelwerte außerhalb der in dataset.json angegebenen
Klassen enthalten.

Beispiel:
    python hpo/check_trial_labels.py --dataset Dataset001_GroundTruth
"""
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit("Please run `pip install nibabel` first.") from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check raw segmentations for unexpected label values."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset001_GroundTruth",
        help="Dataset name under hpo/preprocessing_output/.",
    )
    parser.add_argument(
        "--nnunet_raw_dir",
        default="data/nnUNet_raw",
        help="Path to nnUNet_raw (default: data/nnUNet_raw).",
    )
    parser.add_argument(
        "--config",
        default="3d_fullres",
        help="nnUNet configuration (default: 3d_fullres, info only).",
    )
    return parser.parse_args()


def load_dataset_labels(dataset_json: Path) -> List[int]:
    with open(dataset_json, "r") as f:
        data = json.load(f)
    labels = data.get("labels", {})
    return sorted(set(labels.values()))


def scan_segmentation(seg_path: Path, valid_labels: List[int]) -> Tuple[bool, List[int]]:
    img = nib.load(str(seg_path))
    arr = img.get_fdata().astype(int)
    uniq = sorted(set(arr.flatten()))
    invalid = [v for v in uniq if v not in valid_labels]
    return len(invalid) == 0, invalid


def list_raw_segmentations(dataset_name: str, raw_root: Path) -> List[Path]:
    labels_dir = raw_root / dataset_name / "labelsTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr not found at '{labels_dir}'.")
    return sorted(labels_dir.glob("*.nii*"))


def main():
    args = parse_args()
    raw_root = Path(args.nnunet_raw_dir)
    dataset_json = raw_root / args.dataset / "dataset.json"
    if not dataset_json.exists():
        fallback = Path("data") / args.dataset / "dataset.json"
        dataset_json = fallback if fallback.exists() else dataset_json

    labels = load_dataset_labels(dataset_json)
    print(f"Valid label values according to dataset.json: {labels}")

    issues: Dict[str, List[Tuple[str, List[int]]]] = {}
    seg_files = list_raw_segmentations(args.dataset, raw_root)
    seg_iter = tqdm(seg_files, desc="labelsTr", unit="seg") if tqdm else seg_files
    for seg_path in seg_iter:
        ok, invalid = scan_segmentation(seg_path, labels)
        if not ok:
            issues.setdefault("nnUNet_raw", []).append((seg_path.name, invalid))

    if not issues:
        print("All checked segmentations contain only valid labels.")
        return

    print("\nFound deviations:")
    for trial_name, seg_list in issues.items():
        print(f"- {trial_name}:")
        for seg_name, invalid in seg_list:
            print(f"    {seg_name} -> invalid: {invalid}")


if __name__ == "__main__":
    main()

