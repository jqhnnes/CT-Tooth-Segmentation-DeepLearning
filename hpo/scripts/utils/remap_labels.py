#!/usr/bin/env python3
"""
Setzt alle Labelwerte > max_label auf max_label (Default: 3) für die Masken
in nnUNet_raw/<Dataset>/labelsTr.

Beispiel:
    python hpo/remap_labels.py --dataset Dataset001_GroundTruth --max_label 3
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remappe Maskenlabels auf einen Maximalwert."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset001_GroundTruth",
        help="Dataset-Name unter nnUNet_raw.",
    )
    parser.add_argument(
        "--nnunet_raw",
        default="data/nnUNet_raw",
        help="Pfad zum nnUNet_raw-Ordner (Default: data/nnUNet_raw).",
    )
    parser.add_argument(
        "--max_label",
        type=int,
        default=3,
        help="Alle Labelwerte größer als dieser Wert werden darauf gekappt.",
    )
    return parser.parse_args()


def remap_labels(dataset_dir: Path, max_label: int):
    labels_dir = dataset_dir / "labelsTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr nicht gefunden unter '{labels_dir}'.")

    seg_files = sorted(labels_dir.glob("*.nii*"))
    iterator = tqdm(seg_files, desc="Remapping", unit="file") if tqdm else seg_files
    for seg_path in iterator:
        img = nib.load(str(seg_path))
        data = img.get_fdata().astype(np.int16)
        invalid = data > max_label
        if invalid.any():
            data[invalid] = max_label
            nib.save(nib.Nifti1Image(data, img.affine, img.header), str(seg_path))
            print(f"[Remapped] {seg_path.name}")


def main():
    args = parse_args()
    dataset_dir = Path(args.nnunet_raw) / args.dataset
    remap_labels(dataset_dir, args.max_label)


if __name__ == "__main__":
    main()

