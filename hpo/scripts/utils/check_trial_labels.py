#!/usr/bin/env python3
"""
Check raw segmentation labels for values not declared in dataset.json.

Scans all NIfTI segmentation files in the ``labelsTr`` directory of the
given dataset and reports any voxel values outside the set of valid label
IDs defined in ``dataset.json``.

Example:
    python hpo/scripts/utils/check_trial_labels.py --dataset Dataset001_GroundTruth
"""
import argparse
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
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with ``dataset``, ``nnunet_raw_dir``, and
        ``config`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Check raw segmentations for unexpected label values."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset001_GroundTruth",
        help="Dataset name under data/nnUNet_raw (default: Dataset001_GroundTruth).",
    )
    parser.add_argument(
        "--nnunet_raw_dir",
        default="data/nnUNet_raw",
        help="Path to nnUNet_raw (default: data/nnUNet_raw).",
    )
    parser.add_argument(
        "--config",
        default="3d_fullres",
        help="nnUNet configuration (informational only; default: 3d_fullres).",
    )
    return parser.parse_args()


def load_dataset_labels(dataset_json: Path) -> List[int]:
    """Load valid label IDs from a dataset.json file.

    Args:
        dataset_json: Path to the ``dataset.json`` file.

    Returns:
        Sorted list of integer label IDs declared in the ``labels`` field.
    """
    with open(dataset_json, "r") as f:
        data = json.load(f)
    labels = data.get("labels", {})
    return sorted(set(labels.values()))


def scan_segmentation(seg_path: Path, valid_labels: List[int]) -> Tuple[bool, List[int]]:
    """Check a single NIfTI segmentation file for invalid label values.

    Args:
        seg_path: Path to the ``.nii`` or ``.nii.gz`` segmentation file.
        valid_labels: List of permitted integer label values.

    Returns:
        A tuple ``(is_valid, invalid_values)`` where ``is_valid`` is ``True``
        when all voxel values are in ``valid_labels``, and ``invalid_values``
        is the sorted list of any offending values.
    """
    img = nib.load(str(seg_path))
    arr = img.get_fdata().astype(int)
    uniq = sorted(set(arr.flatten()))
    invalid = [v for v in uniq if v not in valid_labels]
    return len(invalid) == 0, invalid


def list_raw_segmentations(dataset_name: str, raw_root: Path) -> List[Path]:
    """Return all NIfTI segmentation files in the dataset's labelsTr directory.

    Args:
        dataset_name: Dataset folder name (e.g. ``"Dataset001_GroundTruth"``).
        raw_root: Path to the ``nnUNet_raw`` root directory.

    Returns:
        Sorted list of paths to ``.nii`` / ``.nii.gz`` files.

    Raises:
        FileNotFoundError: If the ``labelsTr`` directory does not exist.
    """
    labels_dir = raw_root / dataset_name / "labelsTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr not found at '{labels_dir}'.")
    return sorted(labels_dir.glob("*.nii*"))


def main() -> None:
    """Entry point: check all segmentation files and report invalid labels."""
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

