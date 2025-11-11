"""
Train a baseline model where training/validation data excludes target cells; 
    - No CV
    - without heads 

- Loads the GDSC2 dataset (responses, gene expression, and drug fingerprints).
- Cleans features (handles NaN/Inf, clips extreme values).
- Uses drevalpy‚ SimpleNeuralNetwork with aleatoric uncertainty enabled.
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
import random

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
GENE_DIR       = DATA_ROOT / "GDSC2/gene_expression.csv"

OUT_DIR        = BASE_DIR / "results" / DATASET_NAME
# Directory to store the standard baseline model artifacts (explicitly excludes uncertainty heads)
BASELINE_DIR   = OUT_DIR / "baseline_models" / "exclude_target_no_heads" 
CKPT_DIR       = BASE_DIR / "checkpoints" / DATASET_NAME / "BASELINE"

# ----------------------- CONFIG ------------------------
SEED = 42
#["GA-10","SW1783","NCI-H2803","ETK-1","NCI-H2869","M14","HCT","SaOS-2","SiSo","IGR-1","OAW42","OVISE","NCI-H1623","RXF","NCI-H1395","QGP-1","WRO","NCI-H1734","Panc","EW-8","LNZTA3WT4","TE-10"]   
# ----------------------- UTILS -----------------------------

def load_hparams() -> dict:
    """Loads and formats hyperparameters, explicitly setting aleatoric=False for standard MSE baseline."""
    default = {
        "dropout_prob": 0.3,
        "units_per_layer": [1024, 256, 128, 32],
        "max_epochs": 100,
        "batch_size": 512,
        "lr": 1e-3,
        #"weight_decay": 1e-4,
        "input_dim_gex": None,
        "input_dim_fp": None,
        "aleatoric": False,    # ‚òÖ KEY FIX: Disable aleatoric uncertainty head (Standard MSE Loss)
        # Uncertainty parameters are kept but irrelevant for this standard training setup
        #"logvar_init": -2.0,  
        #"logvar_min":  -5.0,
        #"logvar_max":   2.0,
        #"mc_T": 1,           # MC samples set to 1 for standard point prediction efficiency
    }
    return default

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

def prepare_datasets(
    resp: DrugResponseDataset,
    celline_features: FeatureDataset,
    target_cells: List[str],
) -> Tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Splits the full dataset: 
    1. train_ds: all non-target cell lines (for training the baseline).
    2. test: holds out 100% of samples for each target cell line (for evaluation).
    """
    training_cells = [c for c in celline_features.identifiers if c not in target_cells]
    test_cells = [c for c in celline_features.identifiers if c in target_cells]
    
    train = resp.copy()
    test = resp.copy()

    train.reduce_to(cell_line_ids=training_cells)
    test.reduce_to(cell_line_ids=test_cells)

    return train, test

# ----------------------------- MAIN -----------------------------
def main():
    np.random.seed(SEED)
    # Ensure output and checkpoint directories exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True) # Using the new 'no_heads' directory
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # --- 1. Load Data and Clean Features ---
    hparams = load_hparams()
    model = SimpleNeuralNetwork()
    model.build_model(hparams)

    # Load features (gene expression and drug fingerprints)
    cell_fd = model.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = model.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    # Load response data (LN_IC50)
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] total rows = {len(resp)}")

    # --- 2. Data Splitting ---
    NUM_TARGETS = 20
    all_cell_lines = np.unique(resp.cell_line_ids)
    TARGET_CELL_LINES = np.random.choice(
        a=all_cell_lines,
        size=NUM_TARGETS,
        replace=False 
    )

    train, test = prepare_datasets(resp, cell_fd, TARGET_CELL_LINES)

    # --- 3. Train Baseline Model (Standard MSE Loss) ---
    model.train(
        output=train,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        model_checkpoint_dir=str(CKPT_DIR),
    )

    # a. Get Point Prediction (using the unified wrapper with T=1) co
    # Since aleatoric=False, sigma_ale_std will be None. We only care about mu_bar_std.
    preds= model.predict(
        cell_line_ids=test.cell_line_ids,
        drug_ids=test.drug_ids,
        cell_line_input=cell_fd, 
        drug_input=drug_fd
    )

    # Save predictions (only saving the necessary columns for the standard baseline)
    out_df = pd.DataFrame({
        "cell_line": test.cell_line_ids,
        "drug_id":   test.drug_ids,
        "y_true":    test.response,
        "y_pred":    preds,
    })
    # Save results to the 'no_heads' directory
    out_csv = BASELINE_DIR / f"target_preds_no_uncert.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[SAVE]: predictions ‚Ü{out_csv}")

    # --- 5. Save Model and Consolidated Metrics ---
    model.save(str(BASELINE_DIR))
    print (metrics_all(test.response,preds))

if __name__ == "__main__":
    main()
