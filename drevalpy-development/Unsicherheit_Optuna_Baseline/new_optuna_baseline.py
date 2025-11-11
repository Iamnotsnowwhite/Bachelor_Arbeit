# run_optuna_optimization.py

import optuna
from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Stellen Sie sicher, dass das drevalpy-Modul verfügbar ist
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PFADE (unverändert) --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"

# Erstellen Sie ein neues Stammverzeichnis für die Optuna-Ausgaben
OPTUNA_OUT_DIR = BASE_DIR / "Optu_results" / DATASET_NAME / "optuna_study"
OPTUNA_CKPT_DIR= BASE_DIR / "checkpoints" / DATASET_NAME / "OPTUNA"

# ----------------------- HILFSFUNKTIONEN (unverändert) -----------------------------
def check_paths():
    """Überprüft die Existenz der benötigten Eingabedateien."""
    ge_hint = DATA_ROOT / DATASET_NAME / "gene_expression.csv"
    fp_dir  = DATA_ROOT / DATASET_NAME / "drug_fingerprints"
    print("\n[PATH CHECK]")
    print(" gene expr (hint):", ge_hint, ge_hint.exists())
    print(" fp dir    (hint):", fp_dir,  fp_dir.exists())
    print(" response       :", RESPONSE_FILE, RESPONSE_FILE.exists(), "\n")
    if not RESPONSE_FILE.exists():
        print("[ERROR] Response file not found."); sys.exit(1)

def clean_view(fd: FeatureDataset, view: str, clip_min=None, clip_max=None) -> Dict[str, int]:
    """Bereinigt Merkmalsdaten (behandelt NaN/Inf, beschneidet Werte) in einer FeatureDataset-Ansicht."""
    stats = {"nan": 0, "posinf": 0, "neginf": 0, "clipped": 0}
    for _id in fd.identifiers:
        v = fd.features[_id][view].astype(np.float64, copy=False)
        stats["nan"]    += int(np.isnan(v).sum())
        stats["posinf"] += int(np.isposinf(v).sum())
        stats["neginf"] += int(np.isneginf(v).sum())
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_min is not None and clip_max is not None:
            before = v.copy()
            v = np.clip(v, clip_min, clip_max)
            stats["clipped"] += int((before != v).sum())
        fd.features[_id][view] = v.astype(np.float32, copy=False)
    return stats

def metrics_all(y_true, y_pred) -> Dict[str, float]:
    """Berechnet Standard-Regressionsmetriken (MSE, RMSE, MAE, R2, Pearson-Korrelation)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    pr  = 0.0 if (y_true.std()<1e-12 or y_pred.std()<1e-12) else float(np.corrcoef(y_true, y_pred)[0,1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}

# ----------------------------- MODIFIZIERTE DATENAUFBEREITUNG -----------------------------
def prepare_train_test(
    response: DrugResponseDataset,
    celline_features: FeatureDataset,
    target_cells: List[str],
) -> Tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Teilt den Datensatz nach Zelllinien NUR in Trainings- und Testsets auf.
    Das Trainingsset enthält alle Zelllinien, die NICHT in 'target_cells' sind.
    Das Testset enthält NUR die Zelllinien in 'target_cells'.
    """
    # Zellen, die NICHT im Testset sind, bilden das Trainingsset
    training_cells = [c for c in celline_features.identifiers if c not in target_cells]
    # Zellen, die im Testset sind
    test_cells = [c for c in celline_features.identifiers if c in target_cells]

    # Testset erstellen
    test = response.copy()
    test.reduce_to(cell_line_ids=test_cells)

    # Trainingsset erstellen (alle Daten außer dem Testset)
    train_subset = response.copy()
    train_subset.reduce_to(cell_line_ids=training_cells)

    print(f"Datensplit: {len(train_subset)} Trainingspunkte, {len(test)} Testpunkte.")
    
    # Wir geben nur noch train und test zurück
    return train_subset, test

# ----------------------------- OPTUNA-ZIELFUNKTION (ANGEPASST) -----------------------------
def objective(trial: optuna.trial.Trial, resp: DrugResponseDataset, cell_fd: FeatureDataset, drug_fd: FeatureDataset, target_cell_lines: List[str]) -> float:
    """
    Die Zielfunktion für Optuna. Sie definiert, trainiert und bewertet ein Modell für einen Durchlauf (Trial).
    """
    # --- 1. Hyperparameter für diesen Durchlauf vorschlagen ---
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "max_epochs": 100,  # Erhöhe die Epochenzahl für die HPO, EarlyStopping kümmert sich darum
        "aleatoric": True,
        "logvar_init": -2.0, "logvar_min": -5.0, "logvar_max": 2.0,
        "mc_T": 50
    }

    # Netzwerkstruktur dynamisch definieren
    n_layers = trial.suggest_int("n_layers", 1, 4)
    units = []
    for i in range(n_layers):
        # Anzahl der Neuronen pro Schicht vorschlagen, nachfolgende Schichten sind meist kleiner
        p_units = trial.suggest_int(f"n_units_l{i}", 64, 512)
        units.append(p_units)
    hparams["units_per_layer"] = units

    # Erstelle ein eindeutiges Ausgabeverzeichnis für diesen Durchlauf
    TRIAL_DIR = OPTUNA_OUT_DIR / f"trial_{trial.number}"
    TRIAL_CKPT_DIR = OPTUNA_CKPT_DIR / f"trial_{trial.number}"
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    TRIAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Daten vorbereiten (MODIFIZIERT) ---
    # Rufe die neue Funktion auf, die nur Train und Test zurückgibt
    train, test = prepare_train_test(
        response=resp,
        celline_features=cell_fd,
        target_cells=target_cell_lines,
    )

    # --- 3. Modell trainieren (MODIFIZIERT) ---
    # Mögliche Optuna Pruning-Ausnahmen abfangen
    try:
        model = SimpleNeuralNetwork()
        model.build_model(dict(hparams))
        model.train(
            output=train,
            # output_earlystopping=val_ds, # <-- DIESE ZEILE WURDE ENTFERNT
            # Das Modell verwendet jetzt einen internen Split von 'output' für Early Stopping
            cell_line_input=cell_fd,
            drug_input=drug_fd,
            model_checkpoint_dir=str(TRIAL_CKPT_DIR),
        )
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        print(f"Trial {trial.number} ist mit Fehler fehlgeschlagen: {e}")
        # Wenn ein Fehler auftritt (z.B. CUDA out of memory), einen sehr schlechten Wert zurückgeben
        return float('inf')


    # --- 4. Evaluierung (unverändert) ---
    # Die Evaluierung findet weiterhin auf dem 'test'-Set statt, was korrekt ist.
    print(f"\n=== [TRIAL {trial.number}] EVALUIERUNG ===")
    mean, _, _ = model.predict_uncertainty_by_ids(
        cell_line_ids=test.cell_line_ids,
        drug_ids=test.drug_ids,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        T=hparams["mc_T"],
        keep_bn_eval=True
    )

    tgt_metrics = metrics_all(test.response, mean)
    print(f"Trial {trial.number} Metriken auf Ziel-Zelllinien: {tgt_metrics}")

    # --- 5. Ergebnisse speichern (unverändert) ---
    with open(TRIAL_DIR / "trial_results.json", "w") as f:
        json.dump({
            "trial_number": trial.number,
            "value_rmse": tgt_metrics["RMSE"],
            "hyperparams": hparams,
            "metrics": tgt_metrics,
        }, f, indent=2)

    # --- 6. Die zu minimierende Metrik zurückgeben (unverändert) ---
    return tgt_metrics["RMSE"]


# ----------------------------- HAUPTPROGRAMM (unverändert) -----------------------------
if __name__ == "__main__":
    # --- 0. Einrichtung ---
    OPTUNA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    check_paths()
    N_TRIALS = 100 # Die Anzahl der Durchläufe, die Sie ausführen möchten

    # --- 1. Daten einmalig laden und vorbereiten ---
    print("[INFO] Lade Daten einmalig für alle Durchläufe...")
    tmp = SimpleNeuralNetwork()
    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] Daten geladen. Gesamtzahl Zeilen = {len(resp)}")

    # --- 2. Ziel-Zelllinien einmalig auswählen ---
    NUM_TARGETS = 10
    all_cell_lines = np.unique(resp.cell_line_ids)
    TARGET_CELL_LINES = np.random.choice(
        a=all_cell_lines,
        size=NUM_TARGETS,
        replace=False
    ).tolist() # tolist() für einfache JSON-Serialisierung
    print(f"\n[INFO] Verwende feste Ziel-Zelllinien für alle {N_TRIALS} Durchläufe: {TARGET_CELL_LINES}")

    # --- 3. Optuna-Studie erstellen und ausführen ---
    # Sie können einen Speicher konfigurieren, um die Studie zu speichern und fortzusetzen
    # z.B.: storage="sqlite:///db.sqlite3"
    study = optuna.create_study(direction="minimize", study_name=f"{DATASET_NAME}_baseline_ale_hpo")

    # Verwenden Sie eine Lambda-Funktion, um zusätzliche Argumente (die vorgeladenen Daten) zu übergeben
    study.optimize(
        lambda trial: objective(trial, resp, cell_fd, drug_fd, TARGET_CELL_LINES),
        n_trials=N_TRIALS
    )

    # --- 4. Endergebnisse ausgeben ---
    print("\n\n" + "="*50)
    print(" OPTIMIERUNG ABGESCHLOSSEN ".center(50, "="))
    print("="*50)
    print(f"Anzahl abgeschlossener Durchläufe: {len(study.trials)}")

    best_trial = study.best_trial
    print("\n🏆 Bester Durchlauf:")
    print(f"  Wert (RMSE): {best_trial.value:.4f}")

    print("\n  Optimale Hyperparameter:")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

    # Die besten Parameter speichern
    with open(OPTUNA_OUT_DIR / "best_params.json", "w") as f:
        json.dump(best_trial.params, f, indent=2)

    print(f"\nBeste Parameter gespeichert in: {OPTUNA_OUT_DIR / 'best_params.json'}")
    print(f"Ergebnisse der einzelnen Durchläufe gespeichert in: {OPTUNA_OUT_DIR}")