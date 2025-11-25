#!/usr/bin/env python3
"""
Utility script to repair wrong spacing metadata for a specific nnU-Net case and
regenerate its preprocessed files across all HPO trials.

Typical usage (fix AW062 spacing + reprocess it everywhere):

    python hpo/fix_case_spacing_and_reprocess.py \
        --case_id AW062-C0005656 \
        --spacing 0.04 0.04 0.04
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

# Ensure project root on sys.path so we can import scripts.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nnunet_env import load_env  # noqa: E402

load_env()

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixes spacing metadata for a raw nnU-Net case and regenerates its "
            "preprocessed files in all (or selected) HPO trials."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="Dataset001_GroundTruth",
        help="nnU-Net dataset name (default: Dataset001_GroundTruth).",
    )
    parser.add_argument(
        "--case_id",
        required=True,
        help="Case identifier without channel suffix, e.g. AW062-C0005656.",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        default=(0.04, 0.04, 0.04),
        help="Target voxel spacing in mm for x/y/z (default: 0.04 0.04 0.04).",
    )
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration whose preprocessed files should be rebuilt.",
    )
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Optional list of trial names (e.g. trial_0 trial_5). Default: all trials.",
    )
    parser.add_argument(
        "--skip_raw_fix",
        action="store_true",
        help="Skip rewriting the raw NIfTI files (only reprocess preprocessed data).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep .bak copies of the original raw NIfTI files (recommended).",
    )
    return parser.parse_args()


def list_trial_dirs(base_dir: Path, selection: Iterable[str] | None) -> list[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Trial base directory '{base_dir}' not found. Run preprocessing first."
        )
    all_trials = sorted(
        [
            entry
            for entry in base_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("trial_")
        ],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not selection:
        return all_trials
    available = {t.name: t for t in all_trials}
    missing = sorted(set(selection) - set(available))
    if missing:
        raise FileNotFoundError(
            f"Trials not found under '{base_dir}': {', '.join(missing)}"
        )
    return [available[name] for name in selection]


def ensure_dirs_exist(paths: Sequence[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def build_new_affine(old_affine: np.ndarray, new_spacing: Sequence[float]) -> np.ndarray:
    new_affine = old_affine.copy()
    rotation = old_affine[:3, :3]
    updated = np.zeros_like(rotation)
    for axis, spacing in enumerate(new_spacing):
        col = rotation[:, axis]
        norm = np.linalg.norm(col)
        direction = col / norm if norm > 0 else np.eye(3)[:, axis]
        updated[:, axis] = direction * spacing
    new_affine[:3, :3] = updated
    return new_affine


def rewrite_case_spacing(
    image_path: Path,
    new_spacing: Sequence[float],
    keep_backup: bool,
):
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img = nib.load(str(image_path))
    data = np.asanyarray(img.dataobj)
    header = img.header.copy()
    new_affine = build_new_affine(img.affine, new_spacing)
    header.set_zooms(tuple(new_spacing))

    new_img = nib.Nifti1Image(data, new_affine, header=header)
    new_img.set_sform(new_affine)
    new_img.set_qform(new_affine)

    if keep_backup:
        backup_path = image_path.with_name(image_path.name + ".bak")
        if not backup_path.exists():
            shutil.copy2(image_path, backup_path)
    nib.save(new_img, str(image_path))


def rewrite_raw_files(
    dataset_dir: Path,
    case_id: str,
    new_spacing: Sequence[float],
    keep_backup: bool,
):
    dataset_json = load_json(dataset_dir / "dataset.json")
    file_ending = dataset_json["file_ending"]
    image_file = dataset_dir / "imagesTr" / f"{case_id}_0000{file_ending}"
    label_file = dataset_dir / "labelsTr" / f"{case_id}{file_ending}"
    ensure_dirs_exist([image_file, label_file])

    print(f"[RAW] Updating spacing for {case_id} -> {tuple(new_spacing)} mm")
    rewrite_case_spacing(image_file, new_spacing, keep_backup)
    rewrite_case_spacing(label_file, new_spacing, keep_backup)


def reprocess_case_for_trial(
    trial_dir: Path,
    dataset_name: str,
    case_id: str,
    configuration: str,
    raw_dataset_dir: Path,
) -> Path:
    dataset_dir = trial_dir / dataset_name
    plan_path = dataset_dir / "nnUNetPlans.json"
    dataset_json_path = dataset_dir / "dataset.json"
    config_dir = None

    ensure_dirs_exist([plan_path, dataset_json_path])
    dataset_json = load_json(dataset_json_path)
    file_ending = dataset_json["file_ending"]

    plans_manager = PlansManager(str(plan_path))
    config_manager = plans_manager.get_configuration(configuration)
    preprocessor_cls = config_manager.preprocessor_class
    preprocessor = preprocessor_cls(verbose=False)

    config_dir = dataset_dir / config_manager.data_identifier
    config_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs
    for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
        stale_file = config_dir / f"{case_id}{suffix}"
        if stale_file.exists():
            stale_file.unlink()

    channel_keys = sorted(dataset_json["channel_names"].keys(), key=lambda k: int(k))
    images = []
    for key in channel_keys:
        channel_idx = int(key)
        img_path = raw_dataset_dir / "imagesTr" / f"{case_id}_{channel_idx:04d}{file_ending}"
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        images.append(str(img_path))
    label_file = raw_dataset_dir / "labelsTr" / f"{case_id}{file_ending}"
    if not label_file.exists():
        raise FileNotFoundError(label_file)

    output_prefix = config_dir / case_id
    print(f"[{trial_dir.name}] Regenerating {case_id} in {config_manager.data_identifier}")
    preprocessor.run_case_save(
        str(output_prefix),
        images,
        str(label_file),
        plans_manager,
        config_manager,
        dataset_json,
    )
    return config_dir


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    case_id = args.case_id
    spacing = tuple(args.spacing)

    raw_root = Path(os.environ["nnUNet_raw"]).resolve()
    raw_dataset_dir = raw_root / dataset_name
    if not raw_dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found under nnUNet_raw ({raw_root})."
        )

    if not args.skip_raw_fix:
        rewrite_raw_files(raw_dataset_dir, case_id, spacing, keep_backup=args.backup)
    else:
        print("[RAW] Skipped rewriting raw files (per --skip_raw_fix).")

    trial_base = Path("hpo") / "preprocessing_output" / dataset_name
    trial_dirs = list_trial_dirs(trial_base, args.trials)
    if not trial_dirs:
        print(f"Keine Trials unter {trial_base} gefunden.")
        return

    for trial_dir in trial_dirs:
        try:
            reprocess_case_for_trial(
                trial_dir,
                dataset_name,
                case_id,
                args.configuration,
                raw_dataset_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {trial_dir.name}: Fehler beim Reprocessen ({exc}).")
            raise

    print("\nFertig – alle ausgewählten Trials besitzen nun aktualisierte Dateien.")


if __name__ == "__main__":
    main()

