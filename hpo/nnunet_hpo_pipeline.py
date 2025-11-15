import os
import subprocess
import optuna

# ---- Umgebungsvariablen setzen ----
os.environ['nnUNet_raw_data_base'] = '/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_preprocessed'
os.environ['RESULTS_FOLDER'] = '/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_results'

# ---- Pfade ----
dataset_name = "Dataset001_GroundTruth"
input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], dataset_name)
output_folder = os.path.join(os.path.dirname(input_folder), "preprocessing_output")

# ---- Prüfen, ob Input existiert ----
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder '{input_folder}' existiert nicht.")
if not os.path.exists(os.path.join(input_folder, "imagesTr")) or not os.path.exists(os.path.join(input_folder, "labelsTr")):
    raise FileNotFoundError(f"imagesTr oder labelsTr fehlen im Input-Ordner '{input_folder}'.")

# ---- Objective-Funktion für Optuna ----
def objective(trial):
    patch_x = trial.suggest_categorical('patch_x', [64, 128, 160])
    patch_y = trial.suggest_categorical('patch_y', [64, 128, 160])
    patch_z = trial.suggest_categorical('patch_z', [64, 128, 160])
    batch_size = trial.suggest_categorical('batch_size', [2, 4])

    # Plan-Datei für nnUNet
    plan_file = f"hpo/nnUNetPlans_temp_{trial.number}.json"

    cmd = [
        "nnUNetv2_preprocess",
        "--plans", plan_file,
        "--input", input_folder,
        "--output", output_folder,
        "--num_processes", str(os.cpu_count())
    ]

    try:
        print(f"[Trial {trial.number}] Running preprocess with patch {patch_x},{patch_y},{patch_z} and batch {batch_size}")
        subprocess.run(cmd, check=True)
        # Dummy-Rückgabewert, kann später Accuracy/Loss aus Preprocessed-Daten sein
        return patch_x + patch_y + patch_z + batch_size
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] Preprocessing failed: {e}")
        return None

# ---- Optuna Study starten ----
study = optuna.create_study(direction="maximize")

try:
    study.optimize(objective, n_trials=5)
except Exception as e:
    print(f"Study optimization aborted: {e}")

# ---- Beste Ergebnisse anzeigen ----
try:
    if study.best_trial:
        print("Best trial params:", study.best_trial.params)
except Exception:
    print("Keine abgeschlossenen Trials vorhanden.")
