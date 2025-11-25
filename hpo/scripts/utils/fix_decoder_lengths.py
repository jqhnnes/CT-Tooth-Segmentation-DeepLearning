#!/usr/bin/env python3
"""
Prüft alle Trial-Plan-Dateien und kürzt `n_conv_per_stage_decoder` auf
n_stages-1 Einträge (also die Anzahl der Encoder-Layer minus eins).

Beispiel:
    python hpo/fix_decoder_lengths.py --dataset Dataset001_GroundTruth
"""
import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix n_conv_per_stage_decoder in all trial plans."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset001_GroundTruth",
        help="Dataset name under hpo/preprocessing_output/.",
    )
    parser.add_argument(
        "--plans_file",
        default="/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/hpo/config/nnUNetPlans_template.json",
        help="Path to plan template file (info/default only, optional).",
    )
    return parser.parse_args()


def find_plan_files(dataset_dir: Path) -> List[Path]:
    plan_files = []
    for trial_dir in dataset_dir.iterdir():
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue
        dataset_subdir = trial_dir / dataset_dir.name
        if not dataset_subdir.exists():
            continue
        plan_path = dataset_subdir / "nnUNetPlans.json"
        if plan_path.exists():
            plan_files.append(plan_path)
    return plan_files


def fix_decoder_lengths(plan_path: Path) -> bool:
    with open(plan_path, "r") as f:
        plan = json.load(f)

    modified = False
    for cfg in plan.get("configurations", {}).values():
        arch = cfg.get("architecture", {}).get("arch_kwargs", {})
        encoder = cfg.get("n_conv_per_stage") or arch.get("n_conv_per_stage")
        decoder = cfg.get("n_conv_per_stage_decoder") or arch.get(
            "n_conv_per_stage_decoder"
        )
        if not (isinstance(encoder, list) and isinstance(decoder, list)):
            continue
        expected_len = max(0, len(encoder) - 1)
        if len(decoder) != expected_len:
            new_decoder = decoder[:expected_len] if expected_len > 0 else []
            if "n_conv_per_stage_decoder" in cfg:
                cfg["n_conv_per_stage_decoder"] = new_decoder
            if "n_conv_per_stage_decoder" in arch:
                arch["n_conv_per_stage_decoder"] = new_decoder
            modified = True

    if modified:
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
    return modified


def main():
    args = parse_args()
    dataset_dir = Path("hpo") / "preprocessing_output" / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    plan_files = find_plan_files(dataset_dir)
    if not plan_files:
        print("No plan files found.")
        return

    fixed = 0
    for plan_path in plan_files:
        if fix_decoder_lengths(plan_path):
            print(f"Fix applied: {plan_path}")
            fixed += 1

    print(f"Decoder lengths corrected in {fixed} files.")


if __name__ == "__main__":
    main()

