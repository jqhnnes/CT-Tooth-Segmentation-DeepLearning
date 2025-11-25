#!/usr/bin/env python3
"""
Vergleicht die Top 3 Trials auf einem Testset.

Verwendung:
    python hpo/compare_top3_trials.py --testset labelsTs
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Projekt-Root in den Python-Pfad aufnehmen
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nnunet_env import load_env

load_env()

TOP_3_TRIALS = ['trial_8', 'trial_1', 'trial_3']
DATASET_NAME = "Dataset001_GroundTruth"
DATASET_ID = 1
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer"
PLANS_NAME = "nnUNetPlans"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleicht die Top 3 Trials auf einem Testset."
    )
    parser.add_argument(
        "--testset",
        default="labelsTs",
        help="Testset-Ordner (labelsTs oder labelsVal). Default: labelsTs",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds zu evaluieren. Default: 0",
    )
    parser.add_argument(
        "--output_dir",
        default="hpo/top3_comparison",
        help="Ausgabe-Ordner für Vergleichsergebnisse. Default: hpo/top3_comparison",
    )
    return parser.parse_args()


def ensure_env_vars():
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(
            f"Fehlende Umgebungsvariablen: {', '.join(missing)}"
        )


def predict_with_trial(trial_name: str, testset: str, fold: str, env: dict):
    """Erstellt Vorhersagen für einen Trial auf dem Testset."""
    print(f"\n[{trial_name}] Erstelle Vorhersagen für {testset}...")
    
    # Setze nnUNet_results auf den archivierten Ordner
    results_dir = (
        Path("hpo")
        / "training_output"
        / trial_name
        / "nnUNet_results"
    ).resolve()
    
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} hat keine archivierten Ergebnisse unter {results_dir}"
        )
    
    trial_env = env.copy()
    trial_env["nnUNet_results"] = str(results_dir)
    
    # Setze nnUNet_preprocessed auf den Trial-Preprocessing-Ordner
    preprocessed_dir = (
        Path("hpo")
        / "preprocessing_output"
        / DATASET_NAME
        / trial_name
        / DATASET_NAME
    ).resolve()
    
    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} hat kein Preprocessing unter {preprocessed_dir}"
        )
    
    trial_env["nnUNet_preprocessed"] = str(preprocessed_dir.parent)
    
    # Vorhersage-Ordner
    pred_dir = (
        results_dir
        / DATASET_NAME
        / f"{TRAINER}__{PLANS_NAME}__{CONFIGURATION}"
        / f"fold_{fold}"
        / testset
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Prüfe ob bereits Vorhersagen existieren
    if list(pred_dir.glob("*.nii.gz")):
        print(f"[{trial_name}] Vorhersagen existieren bereits in {pred_dir}")
        return pred_dir
    
    # nnUNetv2_predict Aufruf
    cmd = [
        "nnUNetv2_predict",
        "-i", str(Path(os.environ["nnUNet_raw"]) / DATASET_NAME / testset.replace("labels", "images")),
        "-o", str(pred_dir),
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-tr", TRAINER,
        "-p", PLANS_NAME,
        "-f", fold,
        "--disable_tta",  # Schneller ohne TTA
    ]
    
    print(f"[{trial_name}] -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=trial_env)
    
    return pred_dir


def evaluate_predictions(trial_name: str, pred_dir: Path, testset: str, output_dir: Path, env: dict):
    """Evaluiert die Vorhersagen eines Trials."""
    print(f"\n[{trial_name}] Evaluiere Vorhersagen...")
    
    gt_dir = Path(os.environ["nnUNet_raw"]) / DATASET_NAME / testset
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth Ordner nicht gefunden: {gt_dir}")
    
    # Lade dataset.json und plans.json vom Trial
    trial_dir = Path("hpo") / "preprocessing_output" / DATASET_NAME / trial_name
    dataset_json = trial_dir / DATASET_NAME / "dataset.json"
    plans_json = trial_dir / DATASET_NAME / f"{PLANS_NAME}.json"
    
    if not dataset_json.exists() or not plans_json.exists():
        raise FileNotFoundError(
            f"Trial {trial_name} fehlt dataset.json oder {PLANS_NAME}.json"
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
    print("VERGLEICH DER TOP 3 TRIALS")
    print("=" * 80)
    print(f"Trials: {', '.join(TOP_3_TRIALS)}")
    print(f"Testset: {args.testset}")
    print(f"Folds: {', '.join(args.folds)}")
    print(f"Output: {output_dir}")
    print()
    
    results = {}
    
    for trial_name in TOP_3_TRIALS:
        results[trial_name] = {}
        
        for fold in args.folds:
            try:
                # Erstelle Vorhersagen
                pred_dir = predict_with_trial(trial_name, args.testset, fold, env)
                
                # Evaluiere Vorhersagen
                summary_file = evaluate_predictions(
                    trial_name, pred_dir, args.testset, output_dir, env
                )
                
                results[trial_name][fold] = {
                    'pred_dir': str(pred_dir),
                    'summary_file': str(summary_file)
                }
                
            except Exception as e:
                print(f"[ERROR] {trial_name} Fold {fold} fehlgeschlagen: {e}")
                results[trial_name][fold] = {'error': str(e)}
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"\nErgebnisse gespeichert in: {output_dir}")
    print("\nZum Anzeigen der Ergebnisse:")
    print("  python hpo/analyze_comparison.py --results_dir", output_dir)
    
    return results


if __name__ == "__main__":
    main()

