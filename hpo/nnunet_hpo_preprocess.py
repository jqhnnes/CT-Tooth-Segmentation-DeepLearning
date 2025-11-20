#!/usr/bin/env python3
import argparse
import os
import sys
import json
import shutil
import subprocess
import optuna
import re
from copy import deepcopy

# stelle sicher, dass das Projekt-Root im Python-Pfad liegt,
# damit `scripts` und andere Module gefunden werden
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# falls du ein custom load_env hast (wie in deinem Projekt), lade es:
from scripts.nnunet_env import load_env

load_env()

# ---- Pfade / Konfiguration ----
dataset_name = "Dataset001_GroundTruth"
# Extrahiere Dataset-ID aus dem Namen (z.B. "Dataset001_GroundTruth" -> 1)
match = re.search(r"Dataset(\d+)", dataset_name)
if not match:
    raise ValueError(f"Konnte keine Dataset-ID aus '{dataset_name}' extrahieren. Erwartetes Format: 'DatasetXXX_...'")
dataset_id = int(match.group(1))

nnunet_raw = os.environ.get("nnUNet_raw")
nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
nnunet_results = os.environ.get("nnUNet_results")

if not all([nnunet_raw, nnunet_preprocessed, nnunet_results]):
    raise EnvironmentError(
        "nnUNet erfordert die Umgebungsvariablen 'nnUNet_raw', "
        "'nnUNet_preprocessed' und 'nnUNet_results'. Bitte setze sie "
        "z.B. in scripts/nnunet_env.sh."
    )

input_folder = os.path.join(nnunet_raw, dataset_name)
hpo_dir = "hpo"
# Base-Ordner für HPO-Preprocessing-Outputs innerhalb von hpo/
hpo_preprocessing_base = os.path.join(hpo_dir, "preprocessing_output")
template_plan_path = os.path.join(hpo_dir, "nnUNetPlans_template.json")
dataset_output_dir = os.path.join(hpo_preprocessing_base, dataset_name)

os.makedirs(hpo_dir, exist_ok=True)
os.makedirs(hpo_preprocessing_base, exist_ok=True)
os.makedirs(dataset_output_dir, exist_ok=True)

# ---- Prüfungen ----
if not os.path.exists(input_folder):
    raise FileNotFoundError(
        f"Input folder '{input_folder}' existiert nicht. Bitte nnUNet_raw korrekt setzen."
    )
if not os.path.exists(
    os.path.join(input_folder, "imagesTr")
) or not os.path.exists(os.path.join(input_folder, "labelsTr")):
    raise FileNotFoundError(
        f"imagesTr oder labelsTr fehlen im Input-Ordner '{input_folder}'."
    )

if not os.path.exists(template_plan_path):
    # keine Template vorhanden — Hinweis geben (siehe weiter unten wie du eine Template erzeugst)
    raise FileNotFoundError(
        f"Template-Plan '{template_plan_path}' nicht gefunden.\n"
        "Erzeuge eine Plan-Vorlage, z.B. durch einmaliges Ausführen des nnUNetv2 planning steps (siehe README von nnUNetv2) "
        "oder lege manuell eine JSON mit den bisherigen Plan-Feldern an."
    )


# ---- Hilfsfunktionen ----
def detect_next_trial_index(base_dir):
    max_idx = -1
    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            match = re.match(r"trial_(\d+)", entry)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def reserve_trial_slot(base_dir):
    """
    Gibt den nächsten freien Trial-Namen zurück und legt den Ordner direkt an,
    damit parallel laufende Prozesse sich nicht in die Quere kommen.
    """
    while True:
        trial_idx = detect_next_trial_index(base_dir)
        trial_name = f"trial_{trial_idx}"
        trial_dir = os.path.join(base_dir, trial_name)
        try:
            os.makedirs(trial_dir)
            return trial_idx, trial_name, trial_dir
        except FileExistsError:
            # Zwischen detect() und makedirs() wurde der Ordner angelegt -> noch einmal versuchen
            continue

def replace_hpo_parameters(obj, patch_tuple, batch_size, features_per_stage=None, 
                          n_conv_per_stage=None, batch_dice=None, use_mask_for_norm=None):
    """
    Traversiert ein dict/list und ersetzt HPO-Parameter:
    - patch_size: Patch-Größe
    - batch_size: Batch-Größe
    - features_per_stage: Anzahl Features pro Stage
    - n_conv_per_stage: Anzahl Convolutions pro Stage
    - batch_dice: Batch-Dice Loss Flag
    - use_mask_for_norm: Mask für Normalisierung
    Diese Funktion verändert obj in-place.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            # Patch size
            if (
                ("patch" in lk or "patch_size" in lk)
                and isinstance(v, (list, tuple))
                and len(v) == 3
                and all(isinstance(x, int) for x in v)
            ):
                obj[k] = [
                    int(patch_tuple[0]),
                    int(patch_tuple[1]),
                    int(patch_tuple[2]),
                ]
            # Batch size
            elif "batch_size" in lk and isinstance(v, int):
                obj[k] = int(batch_size)
            # Features per stage
            elif "features_per_stage" in lk and isinstance(v, list) and features_per_stage is not None:
                obj[k] = features_per_stage
            # n_conv_per_stage (encoder)
            elif "n_conv_per_stage" in lk and isinstance(v, list) and n_conv_per_stage is not None:
                obj[k] = n_conv_per_stage
            # n_conv_per_stage_decoder
            elif "n_conv_per_stage_decoder" in lk and isinstance(v, list) and n_conv_per_stage is not None:
                # Decoder hat typischerweise n_stages-1 Convs
                obj[k] = n_conv_per_stage[:-1] if len(n_conv_per_stage) > 1 else n_conv_per_stage
            # batch_dice
            elif "batch_dice" in lk and batch_dice is not None:
                obj[k] = batch_dice
            # use_mask_for_norm
            elif "use_mask_for_norm" in lk and isinstance(v, list) and use_mask_for_norm is not None:
                obj[k] = [use_mask_for_norm]
            else:
                replace_hpo_parameters(v, patch_tuple, batch_size, features_per_stage,
                                    n_conv_per_stage, batch_dice, use_mask_for_norm)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], (dict, list)):
                replace_hpo_parameters(obj[i], patch_tuple, batch_size, features_per_stage,
                                      n_conv_per_stage, batch_dice, use_mask_for_norm)
    # primitive types werden nicht weiter traversiert


# ---- Objective für Optuna ----
def objective(trial):
    # ===== PARAMETERRAUM =====
    # Patch-Größen (wichtig für Memory/Performance Trade-off)
    patch_x = trial.suggest_categorical("patch_x", [64, 128, 160])
    patch_y = trial.suggest_categorical("patch_y", [64, 128, 160])
    patch_z = trial.suggest_categorical("patch_z", [64, 128, 160])
    patch = (patch_x, patch_y, patch_z)
    
    # Batch-Größe
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    
    # Network-Kapazität: Features pro Stage
    # Optionen: kleinere (weniger Memory) vs größere (mehr Kapazität)
    features_base = trial.suggest_categorical("features_base", [24, 32, 48])
    features_per_stage = [
        features_base,
        features_base * 2,
        features_base * 4,
        features_base * 8,
        features_base * 10,
        features_base * 10
    ]
    
    # Anzahl Convolutions pro Stage (mehr = tieferes Netzwerk)
    n_conv_per_stage = trial.suggest_categorical("n_conv_per_stage", [2, 3])
    n_conv_list = [n_conv_per_stage] * 6  # 6 stages
    
    # Batch-Dice Loss (kann Performance beeinflussen)
    batch_dice = trial.suggest_categorical("batch_dice", [False, True])
    
    # Normalisierung mit Mask (kann bei CT-Daten helfen)
    use_mask_for_norm = trial.suggest_categorical("use_mask_for_norm", [False, True])

    trial_idx, trial_name, trial_output_dir = reserve_trial_slot(dataset_output_dir)
    trial_plan_path = os.path.join(hpo_dir, f"nnUNetPlans_temp_{trial_name}.json")
    
    # Temporärer Ordner für nnUNet (nnUNet erwartet: nnUNet_preprocessed/DatasetXXX/...)
    # Wir setzen nnUNet_preprocessed auf dataset_output_dir, damit nnUNet dort DatasetXXX/ erstellt
    # Danach verschieben wir alles nach trial_X/
    temp_preprocessed_base = dataset_output_dir
    temp_preprocessed_dir = os.path.join(temp_preprocessed_base, dataset_name)
    os.makedirs(temp_preprocessed_dir, exist_ok=True)

    # 1) lade template, modifiziere und speichere Trial-plan
    with open(template_plan_path, "r") as f:
        plan = json.load(f)

    plan_mod = deepcopy(plan)
    replace_hpo_parameters(plan_mod, patch, batch_size, features_per_stage,
                          n_conv_list, batch_dice, use_mask_for_norm)

    # Speichere die modifizierte Plans-Datei im hpo-Ordner (für Nachvollziehbarkeit)
    with open(trial_plan_path, "w") as f:
        json.dump(plan_mod, f, indent=2)
    
    # 2) Kopiere die Plans-Datei an den temporären Ort, wo nnUNet sie erwartet
    # nnUNet sucht nach: nnUNet_preprocessed/DatasetXXX/nnUNetPlans.json
    plans_in_temp = os.path.join(temp_preprocessed_dir, "nnUNetPlans.json")
    shutil.copy2(trial_plan_path, plans_in_temp)

    # 3) Kopiere dataset.json, weil nnUNetv2_preprocess sie im nnUNet_preprocessed-Folder erwartet
    dataset_json_src = os.path.join(input_folder, "dataset.json")
    if not os.path.exists(dataset_json_src):
        raise FileNotFoundError(
            f"dataset.json nicht gefunden unter '{dataset_json_src}'. "
            "Bitte stelle sicher, dass dein nnUNet_raw Dataset komplett ist."
        )
    dataset_json_temp = os.path.join(temp_preprocessed_dir, "dataset.json")
    shutil.copy2(dataset_json_src, dataset_json_temp)
    
    # 4) Aufruf nnUNetv2_preprocess mit dem Trial-plan
    # Setze temporär nnUNet_preprocessed auf dataset_output_dir
    # nnUNet erstellt dann: preprocessing_output/DatasetXXX/DatasetXXX/...
    env = os.environ.copy()
    env["nnUNet_preprocessed"] = os.path.abspath(temp_preprocessed_base)
    
    # nnUNet verwendet -plans_name (ohne .json) und sucht die Datei im Dataset-Ordner
    cmd = [
        "nnUNetv2_preprocess",
        "-d",
        str(dataset_id),
        "-plans_name",
        "nnUNetPlans",  # Name ohne .json - nnUNet sucht nach nnUNetPlans.json
        "-c",
        "3d_fullres",
        "--num_processes",
        str(max(1, min(8, os.cpu_count() or 1))),
    ]

    print(f"\n[{trial_name}] ===== HPO-Parameter =====")
    print(f"  Patch: {patch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Features base: {features_base} -> {features_per_stage}")
    print(f"  Convs per stage: {n_conv_per_stage}")
    print(f"  Batch Dice: {batch_dice}")
    print(f"  Use mask for norm: {use_mask_for_norm}")
    print(f"  Output: {trial_output_dir}")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)
    
    # 5) Verschiebe alle Dateien vom temporären Ordner nach trial_X/
    # nnUNet hat alles in temp_preprocessed_dir erstellt, wir wollen es in trial_output_dir
    if os.path.exists(temp_preprocessed_dir):
        # Verschiebe alle Inhalte von DatasetXXX/ nach trial_X/
        for item in os.listdir(temp_preprocessed_dir):
            src = os.path.join(temp_preprocessed_dir, item)
            dst = os.path.join(trial_output_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        # Lösche den leeren DatasetXXX-Ordner
        try:
            os.rmdir(temp_preprocessed_dir)
        except OSError:
            pass  # Ordner nicht leer oder existiert nicht

    # === HIER: Platzhalter-Metrik ===
    # Preprocessing liefert keine Performance-Metrik. Du musst diesen Teil ersetzen,
    # wenn du echte Validationsergebnisse willst (z.B. trainiere kurz ein Modell und evaluiere).
    #
    # WICHTIG: Der aktuelle Proxy-Score ist nur ein Platzhalter!
    # Er maximiert einfach die Summe der Parameter, was NICHT sinnvoll ist.
    # Für echte HPO musst du hier eine echte Metrik verwenden (z.B. Dice-Score nach Training).
    proxy_score = patch_x + patch_y + patch_z + batch_size + features_base + n_conv_per_stage
    print(f"[{trial_name}] Proxy score: {proxy_score} (NUR PLATZHALTER!)")
    return proxy_score


# ---- Study starten ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starte nnUNet HPO Preprocessing-Trials.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Anzahl neuer Trials, die in diesem Lauf gestartet werden (Default: 10).",
    )
    args = parser.parse_args()

    next_idx = detect_next_trial_index(dataset_output_dir)
    print(
        f"Starte Optuna-Run mit {args.n_trials} neuen Trials. "
        f"Nächster verfügbarer Ordner: trial_{next_idx}."
    )

    # HINWEIS: Mit erweiterten Parametern ist der Suchraum größer!
    # Gesamt: ~27 * 2 * 3 * 2 * 2 * 2 = ~1,296 mögliche Kombinationen
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    if study.trials:
        print("Beste Trial-Parameter:", study.best_trial.params)
        print("Beste Trial-Wert:", study.best_value)
    else:
        print("Keine abgeschlossenen Trials vorhanden.")
