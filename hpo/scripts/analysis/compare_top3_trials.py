#!/usr/bin/env python3
"""
Compare a set of trials on a test set.

This script generates predictions and evaluates them for selected HPO trials,
allowing for direct comparison of their performance.

Examples:
    python hpo/scripts/analysis/compare_top3_trials.py --testset labelsTs
    python hpo/scripts/analysis/compare_top3_trials.py --trials trial_15 trial_16 trial_17
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nnunet_env import load_env

load_env()

DEFAULT_TRIALS = ['trial_8', 'trial_1', 'trial_3']
DATASET_NAME = "Dataset001_GroundTruth"
DATASET_ID = 1
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer"
PLANS_NAME = "nnUNetPlans"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the top 3 trials on a test set."
    )
    parser.add_argument(
        "--testset",
        default="labelsTs",
        help="Test set folder (labelsTs or labelsVal). Default: labelsTs",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds to evaluate. Default: 0",
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        default=DEFAULT_TRIALS,
        help=f"Trials to compare. Default: {', '.join(DEFAULT_TRIALS)}",
    )
    parser.add_argument(
        "--output_dir",
        default="hpo/analysis",
        help="Output directory for comparison results. Default: hpo/analysis",
    )
    return parser.parse_args()


def ensure_env_vars():
    """Ensure all required nnUNet environment variables are set."""
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}"
        )


def predict_with_trial(trial_name: str, testset: str, fold: str, env: dict):
    """Generate predictions for a trial on the test set."""
    print(f"\n[{trial_name}] Generating predictions for {testset}...")
    
    # Set nnUNet_results to the archived directory
    results_dir = (
        Path("hpo")
        / "training_output"
        / trial_name
        / "nnUNet_results"
    ).resolve()
    
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} has no archived results at {results_dir}"
        )
    
    trial_env = env.copy()
    trial_env["nnUNet_results"] = str(results_dir)
    
    # Set nnUNet_preprocessed to the trial preprocessing directory
    preprocessed_dir = (
        Path("hpo")
        / "preprocessing_output"
        / DATASET_NAME
        / trial_name
        / DATASET_NAME
    ).resolve()
    
    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} has no preprocessing at {preprocessed_dir}"
        )
    
    trial_env["nnUNet_preprocessed"] = str(preprocessed_dir.parent)
    
    # Prediction directory
    pred_dir = (
        results_dir
        / DATASET_NAME
        / f"{TRAINER}__{PLANS_NAME}__{CONFIGURATION}"
        / f"fold_{fold}"
        / testset
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if predictions already exist
    if list(pred_dir.glob("*.nii.gz")):
        print(f"[{trial_name}] Predictions already exist at {pred_dir}")
        return pred_dir
    
    # nnUNetv2_predict call
    cmd = [
        "nnUNetv2_predict",
        "-i", str(Path(os.environ["nnUNet_raw"]) / DATASET_NAME / testset.replace("labels", "images")),
        "-o", str(pred_dir),
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-tr", TRAINER,
        "-p", PLANS_NAME,
        "-f", fold,
        "--disable_tta",  # Faster without TTA
    ]
    
    print(f"[{trial_name}] -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=trial_env)
    
    return pred_dir


def evaluate_predictions(trial_name: str, pred_dir: Path, testset: str, output_dir: Path, env: dict):
    """Evaluate predictions for a trial."""
    print(f"\n[{trial_name}] Evaluating predictions...")
    
    gt_dir = Path(os.environ["nnUNet_raw"]) / DATASET_NAME / testset
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    
    # Load dataset.json and plans.json from trial
    trial_dir = Path("hpo") / "preprocessing_output" / DATASET_NAME / trial_name
    dataset_json = trial_dir / DATASET_NAME / "dataset.json"
    plans_json = trial_dir / DATASET_NAME / f"{PLANS_NAME}.json"
    
    if not dataset_json.exists() or not plans_json.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} missing dataset.json or {PLANS_NAME}.json"
        )
    
    summary_file = output_dir / f"{trial_name}_{testset}_summary.json"
    
    cmd = [
        "nnUNetv2_evaluate_folder",
        str(gt_dir),
        str(pred_dir),
        "-djfile", str(dataset_json),
        "-pfile", str(plans_json),
        "-o", str(summary_file),
        "--chill",
    ]
    
    print(f"[{trial_name}] -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    
    return summary_file


def main():
    args = parse_args()
    ensure_env_vars()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    
    print("=" * 80)
    print("TRIAL COMPARISON")
    print("=" * 80)
    print(f"Trials: {', '.join(args.trials)}")
    print(f"Test set: {args.testset}")
    print(f"Folds: {', '.join(args.folds)}")
    print(f"Output: {output_dir}")
    print()
    
    results = {}
    
    for trial_name in args.trials:
        results[trial_name] = {}
        
        for fold in args.folds:
            try:
                # Generate predictions
                pred_dir = predict_with_trial(trial_name, args.testset, fold, env)
                
                # Evaluate predictions
                summary_file = evaluate_predictions(
                    trial_name, pred_dir, args.testset, output_dir, env
                )
                
                results[trial_name][fold] = {
                    'pred_dir': str(pred_dir),
                    'summary_file': str(summary_file)
                }
                
            except Exception as e:
                print(f"[ERROR] {trial_name} Fold {fold} failed: {e}")
                results[trial_name][fold] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nTo view results:")
    print(f"  python hpo/scripts/analysis/analyze_comparison.py --results_dir {output_dir}")
    
    return results


if __name__ == "__main__":
    main()

