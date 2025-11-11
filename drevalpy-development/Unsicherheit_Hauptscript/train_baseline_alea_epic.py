"""
Train a baseline model where training/validation data excludes target cells; 
    - No CV
    - but do training with heads 

- Loads the GDSC2 dataset (responses, gene expression, and drug fingerprints).
- Cleans features (handles NaN/Inf, clips extreme values).
- Uses drevalpy’s SimpleNeuralNetwork with aleatoric uncertainty enabled.
- Leaves out one target cell line completely (not used in training or validation).
- Splits the remaining data into train and validation sets.
- Trains the model on train set, uses validation for early stopping.
- Evaluates performance on validation and the held-out target cell line.
- Saves model, metrics, and results for later comparison
"""

from pathlib import Path
import os, sys, json, yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"

OUT_DIR        = BASE_DIR / "results" / DATASET_NAME
# Directory to store the baseline model artifacts and prediction CSVs for Active Learning
BASELINE_DIR   = OUT_DIR / "test_baseline_models_ale_epic" / "exclude_target_ale_epic"
CKPT_DIR       = BASE_DIR / "checkpoints" / DATASET_NAME / "BASELINE"

# ----------------------- CONFIG ------------------------
# SEED = 42
# Target cell lines to be completely held out from the training data (for evaluation)
# TARGET_CELL_LINES = ["GA-10","SW1783","NCI-H2803","ETK-1","NCI-H2869"]   
# ----------------------- UTILS -----------------------------
def load_hparams() -> dict:
    """Loads and formats hyperparameters, setting defaults for uncertainty (aleatoric, mc_T)."""
    default = {
        "dropout_prob": 0.3,
        "units_per_layer": [256, 128, 32],
        "max_epochs": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "input_dim_gex": None,
        "input_dim_fp": None,
        "aleatoric": True,    # Enables aleatoric head and Gaussian NLL loss
        # Uncertainty inference settings
        "logvar_init": -2.0,
        "logvar_min":  -5.0,
        "logvar_max":   2.0,
        "mc_T": 30,           # Number of Monte Carlo Dropout samples
    }
    return default

#helpfuc
def check_paths():
    """Verifies existence of required input data files."""
    ge_hint = DATA_ROOT / DATASET_NAME / "gene_expression.csv"
    fp_dir  = DATA_ROOT / DATASET_NAME / "drug_fingerprints"
    print("\n[PATH CHECK]")
    print(" gene expr (hint):", ge_hint, ge_hint.exists())
    print(" fp dir    (hint):", fp_dir,  fp_dir.exists())
    print(" response       :", RESPONSE_FILE, RESPONSE_FILE.exists(), "\n")
    if not RESPONSE_FILE.exists():
        print("[ERROR] Response file not found."); sys.exit(1)

#helpfuc
def clean_view(fd: FeatureDataset, view: str, clip_min=None, clip_max=None) -> Dict[str, int]:
    """Cleans feature data (handles NaN/Inf, clips values) in a FeatureDataset view."""
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
    """Calculates standard regression metrics (MSE, RMSE, MAE, R2, Pearson correlation)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    pr  = 0.0 if (y_true.std()<1e-12 or y_pred.std()<1e-12) else float(np.corrcoef(y_true, y_pred)[0,1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}

def prepare_train_val_test(
    response: DrugResponseDataset,
    celline_features: FeatureDataset,
    target_cells: List[str],
    val_fraction: float,
    seed: int,
) -> Tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset, List[str], List[str]]:
    """
    Split the dataset by cell lines into training, validation, and test sets.

    Logic:
        1. Test set  = target cell lines (completely held out, not used in training or validation)
        2. Train set = 80% of non-target cell lines
        3. Validation set   = 20% of non-target cell lines

    Returns:
        train_subset : DrugResponseDataset  → training data (non-target cells)
        val_subset   : DrugResponseDataset  → validation data (non-target cells)
        test      : DrugResponseDataset  → test data (target cells)
        training_cells  : List[str]            → list of training cell line IDs
        validation_cells : List[str]            → list of validation cell line IDs
    """
    # Separate non-target and target cell lines
    training_cells = [c for c in celline_features.identifiers if c not in target_cells]
    test_cells = [c for c in celline_features.identifiers if c in target_cells]

    # Create the test dataset (target cell lines only)
    test = response.copy()
    test.reduce_to(cell_line_ids=test_cells)

    # Create dataset containing only non-target cell lines (for train/val split)
    non_target = response.copy()
    non_target.reduce_to(cell_line_ids=training_cells)

    # Split non-target cell lines into training and validation sets
    rng = np.random.default_rng(seed)
    all_cells = np.unique(non_target.cell_line_ids)
    n_val = max(1, int(len(all_cells) * val_fraction))
    validation_cells = set(rng.choice(all_cells, size=n_val, replace=False))
    target_cells = [c for c in all_cells if c not in validation_cells]

    # Create training and validation datasets
    train_subset = non_target.copy(); 
    train_subset.reduce_to(cell_line_ids=target_cells)
    val_subset   = non_target.copy(); 
    val_subset.reduce_to(cell_line_ids=list(validation_cells))

    return train_subset, val_subset, test, training_cells, list(validation_cells)

# ----------------------------- MAIN -----------------------------
def main():
    # Ensure output and checkpoint directories exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # --- 1. Load Data and Clean Features ---
    hparams = load_hparams()
    tmp = SimpleNeuralNetwork()
    tmp.build_model(hparams)

    # Load features (gene expression and drug fingerprints)
    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    # Load response data (LN_IC50)
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] total rows = {len(resp)}")

    # --- 2. test Data ---
    NUM_TARGETS = 2
    all_cell_lines = np.unique(resp.cell_line_ids)
    TARGET_CELL_LINES = np.random.choice(
        a=all_cell_lines,
        size=NUM_TARGETS,
        replace=False 
    )

    train, val_ds, test, training_cells, val_cells = prepare_train_val_test(
    response=resp,
    val_fraction=0.2,
    celline_features=cell_fd,
    target_cells=TARGET_CELL_LINES,
    seed=42,
)

    # --- 3. Train Baseline Model ---
    model = SimpleNeuralNetwork()
    model.build_model(dict(hparams))
    model.train(
        output=train,
        output_earlystopping=val_ds,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        model_checkpoint_dir=str(CKPT_DIR),
    )

    # --- 4. Evaluation (MC Uncertainty) ---
    print("\n=== EVALUATION (Target/Test: MC mean as point prediction; save uncertainties) ===")
    all_metrics_rows = []

    T_mc = hparams.get("mc_T", 30)
    all_metrics_rows = []

    mean, sigma_epi, sigma_ale = model.predict_uncertainty_by_ids(
        cell_line_ids=test.cell_line_ids,
        drug_ids=test.drug_ids,
        cell_line_input=cell_fd, 
        drug_input=drug_fd,
        T=T_mc,
        keep_bn_eval=True
    )

    if sigma_ale is not None:
        sigma_ale = sigma_ale 
        # Total uncertainty = sqrt(Epistemic^2 + Aleatoric^2)
        sigma_tot = np.sqrt(sigma_epi**2 + sigma_ale**2)
    else:
        sigma_tot = sigma_epi 

    # 4c. Calculate Metrics and Save Results
    tgt_metrics = metrics_all(test.response, mean)
    num_test_cells = len(np.unique(test.cell_line_ids))
    print(f"Target/Test (Over {num_test_cells} cells) Metrics:", tgt_metrics)

    # Create DataFrame containing true values, predictions, and all uncertainty metrics
    out_df = pd.DataFrame({
        "cell_line": test.cell_line_ids,
        "drug_id":   test.drug_ids,
        "y_true":    test.response,
        "y_pred":    mean,
        "sigma_epi": sigma_epi,
        "sigma_ale": sigma_ale if sigma_ale is not None else np.nan,
        "sigma_tot": sigma_tot,
        "mc_T":      T_mc,
    })
    out_csv = BASELINE_DIR / f"target_preds_with_uncert.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[SAVE] results: predictions → {out_csv}")

    # --- 5. Save Model and Consolidated Metrics ---
    model.save(str(BASELINE_DIR))

    with open(BASELINE_DIR / "baseline_results.json", "w") as f:
        json.dump({
            "target_cells": np.unique(test.cell_line_ids).tolist(),
            "train_size": len(train),
            "target_size": len(test),
            "hyperparams": hparams,
            "per_target_metrics": all_metrics_rows,
        }, f, indent=2)

    metrics_csv = BASELINE_DIR / "baseline_metrics.csv"
    pd.DataFrame(all_metrics_rows).to_csv(metrics_csv, index=False)
    print("\n BASELINE TRAINING COMPLETED")
    print("target_celllines:", TARGET_CELL_LINES)
    print("  - model dir :", BASELINE_DIR)
    print("  - metrics   :", metrics_csv)
    print("  - per-target CSVs saved under :", BASELINE_DIR)

if __name__ == "__main__":
    main()