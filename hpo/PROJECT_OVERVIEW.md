## Project Overview

Dieses Dokument fasst das methodische Vorgehen für die Hyperparameter-Optimierung (HPO) und die Trainings-/Inference-Pipeline eines nnU-Net-v2-Modells für 3D-Zahnsegmentierung zusammen. Es dient als Grundlage für eine wissenschaftliche Beschreibung (z. B. Bachelorarbeit) und verweist auf die relevanten Skripte im Ordner `hpo/`.

### Zielsetzung
- Höhere Detailtreue und feinere Kanten durch aggressivere Raumauflösung (Spacing) und größerer Modellkapazität.
- Systematisches Durchsuchen des Hyperparameterraums trotz enger VRAM-Budgets.
- Reproduzierbare Pipeline von der Datenaufbereitung bis zur Auswertung inkl. TTA und Postprocessing.

## Pipeline in Phasen

### 1) Planung und Datenaufbereitung (HPO/Preprocessing)
- Skript: `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`
- Aufgabe: Optuna erzeugt Plan-Varianten, passt Spacing/Patch/Batch/Features dynamisch an und führt `nnUNetv2_preprocess` pro Trial aus. Ergebnisse liegen unter `hpo/preprocessing_output/<Dataset>/trial_X/`.
- Umgebungsvariablen (werden durch `scripts/nnunet_env.sh` gesetzt): `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.
- Ausführung: `python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials N`
- Logging pro Trial: `params.json` (gewählte Hyperparameter), `error.log` (Fehlerfälle).

### 2) Training (ggf. inkl. Evaluation)
- Skript: `hpo/scripts/training/nnunet_train_eval_pipeline.py`
- Aufgabe: Staging der jeweils vorverarbeiteten Trial-Daten, Start von `nnUNetv2_train`, optional `nnUNetv2_evaluate_folder`, GPU-Logging.
- Folds: frei wählbar (Standard: fold 0). Wiederaufnahme: identischen Trainingsbefehl erneut ausführen (nnU-Net setzt automatisch fort).
- Archivierung: Nach jedem Trial werden Ergebnisse nach `hpo/training_output/trial_X/nnUNet_results/...` verschoben.
- Wichtige Flags: `--trials`, `--folds`, `--skip_evaluation`, `--only_evaluate`, `--stop_on_error`, `--eval_timeout`.

### 3) Inferenz, TTA und Postprocessing
- Skript: `hpo/scripts/postprocessing/nnunet_tta_postprocess.py`
- Aufgabe: TTA-Prediction (`nnUNetv2_predict`, TTA standardmäßig aktiv), optional `nnUNetv2_find_best_configuration` (mit korrekten Folds), Anwendung von `postprocessing.pkl`, optional Evaluation.
- Wichtige Pfade je Trial: `nnUNet_results/.../fold_<f>/labelsTs_tta` → `labelsTs_tta_pp`.
- Fallbacks: erkennt vorhandene `postprocessing.pkl` in Modell- oder `crossval_results_folds_0`-Ordner, um redundante Suche zu vermeiden.

### 4) Bewertung und Ranking
- Skript: `hpo/scripts/postprocessing/evaluate_tta_pp.py`
- Aufgabe: Evaluieren vorhandener `labelsTs_tta_pp` pro Trial/Fold mit `nnUNetv2_evaluate_folder`, Schreiben von `hpo/analysis/<trial>_labelsTs_tta_pp_summary.json`, Ranking nach Dice; zeigt ΔDice vs. Baseline/TTA-only falls verfügbar.
- Flags: `--trials`, `--folds`, `--labels_ts`, `--force`.

### 5) Analyse und Visualisierung
- `hpo/scripts/analysis/summarize_trials.py`: Aggregiert Spacing/Patch/Batch/Features und Dice (baseline bzw. TTA+PP) zu `hpo/analysis/trials_summary.json`.
- `hpo/scripts/analysis/plot_trials_summary.py`: Visualisiert Dice vs. Spacing, koloriert nach `features_base`; Ausgabe `hpo/analysis/plots/trials_dice_vs_spacing.png`.

## Evolution des Suchraums (Optuna)

| Phase | Ziel | Spacing-Kandidaten (Beispiele) | Patchgrößen | Batch | features_base | Motivation |
|-------|------|--------------------------------|-------------|-------|---------------|------------|
| Früh | Stabilität, erste Feinkorn-Tests | 0.25–0.20 mm, erste Versuche <0.20 mm | moderat | 2–4 | 24–32 | Crash-freie Baselines sammeln, VRAM-reserviert |
| Mittel | Fokus auf <0.20 mm | 0.15, 0.12, 0.10 mm | kleiner/angepasst | 1–2 | 32–48 | Feinere Auflösung bei reduzierter Batch/Features, OOM minimieren |
| Spät | Aggressiv feinkörnig | 0.10, 0.095, 0.08 mm | größer, kontextreich | 1 | 48–64 | Mehr Kontext trotz feinem Spacing, nahe VRAM-Grenze |
| Aktuell (High-End) | Maximale Detailtiefe | 0.075 mm → Patch (192×256×128), fb=64; 0.080 mm → Patch (224×256×128), fb=72 | 1 | 64 / 72 | Feinste Spacings mit großer Patchfläche; bewusst VRAM-nah für höhere Detailauflösung |

Weitere Anpassungen:
- Kombinationsraum als statische Liste (`combo_choices`), um Optuna-Fehler „CategoricalDistribution does not support dynamic value space“ zu vermeiden.
- Dynamische `features_per_stage`-Skalierung abhängig von `features_base`, um Kapazität auszureizen und gleichzeitig VRAM zu schonen.
- `batch_dice` wird pro Trial variiert (True/False).
- Speichern der Trial-Parameter (`params.json`) und Fehlermeldungen (`error.log`) zur Nachverfolgung.

## Wichtige Betriebsdetails
- Umgebungsvariablen immer pro Trial korrekt setzen, insbesondere bei separatem Inferenz-/PP-Lauf:
  - `nnUNet_preprocessed=/.../hpo/preprocessing_output/<Dataset>/trial_X`
  - `nnUNet_results=/.../hpo/training_output/trial_X/nnUNet_results`
  - `nnUNet_raw` auf den Datensatz (z. B. `data/nnUNet_raw/Dataset001_GroundTruth`)
- TTA ist bei `nnUNetv2_predict` standardmäßig aktiv (`--disable_tta` schaltet ab).
- `nnUNetv2_find_best_configuration` benötigt die trainierten Folds; bei Einzel-Fold unbedingt `-f 0` setzen (in `nnunet_tta_postprocess.py` implementiert).
- Postprocessing sucht `postprocessing.pkl` zuerst in `model_root`, dann in `crossval_results_folds_0`, um unnötige Re-Läufe zu vermeiden.
- Training kann durch erneutes Ausführen desselben Befehls fortgesetzt werden (kein `--continue_training` nötig).

## Typische Fehlerquellen und Fixes
- Fehlende `_0000`-Suffixe in `imagesTs` → Symlinks mit Suffix anlegen.
- Falsche Umgebungsvariablen (verweisen auf globale statt trial-spezifische Pfade) → vor Inferenz/PP korrekt setzen.
- Optuna-Categorical-Fehler → statischer Kombinationsraum (siehe oben).
- `nnUNetv2_find_best_configuration` erwartet standardmäßig 5 Folds → bei Einzel-Fold explizit `-f 0`.

## Aktueller Stand (Beispiel)
- Beste bekannte Einzel-Fold-Run (trial_43) liegt im Bereich ~0.78 Dice nach TTA+PP (foreground_mean), basierend auf vorhandenen Summaries. Weitere Verbesserungen werden über feinere Spacings und erhöhte `features_base` angestrebt.

## Hinweise für die Dokumentation
- Beschreibe die Methodik entlang der Phasen Planung → Umsetzung → Evaluation → Reflexion.
- Hebe den Trade-off zwischen Detailtiefe (feines Spacing, große Patches, hohe `features_base`) und VRAM-Grenzen hervor.
- Dokumentiere, wie Postprocessing (Connected Components / `postprocessing.pkl`) und TTA zur Qualitätssteigerung beitragen.
- Verweise auf die Analyse-Skripte zur Reproduzierbarkeit von Rankings und Abbildungen.

