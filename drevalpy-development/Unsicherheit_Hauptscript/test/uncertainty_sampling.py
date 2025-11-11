from pathlib import Path
import os, sys, json, yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Literal,Optional
import random

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS & CONFIG --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data" 
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"

OUT_DIR        = BASE_DIR / "results" / DATASET_NAME
BASELINE_DIR   = OUT_DIR / "baseline_models_ale_epic" / "exclude_target_ale_epic"
CKPT_DIR       = BASE_DIR / "checkpoints" / DATASET_NAME / "BASELINE"

# Active Learning loop parameters
N_BUDGET = 20       # Wie viel ich jedes mal bei jeder Cycle hinzufüge
# We hold out one cell line for the target task
# 20 celllines 
RANDOM_SEED = 42    
MAX_AL_CYCLES =  2
FINETUNE_EPOCHS = 12
STRATEGY = os.environ['AL_STRATEGY']

#STRATEGY = 'aleatoric' 

# Control the prediction range ('all' = all drugs; 'remaining' = only drugs not used for fine-tuning)
# 控制预测范围（'all'=在全部药物上预测；'remaining'=仅在未用于微调的药物上预测）
PREDICT_SCOPE: Literal['all', 'remaining'] = 'all'

# ----------------------- UTILS -----------------------------
def load_hparams() -> dict:
    """Loads and formats hyperparameters, setting defaults for uncertainty (aleatoric, mc_T)."""
    default = {
        "dropout_prob": 0.3,
        "units_per_layer": [256, 128, 32],
        "max_epochs": 80,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "input_dim_gex": None,
        "input_dim_fp": None,
        "aleatoric": True,    # Enables aleatoric head and Gaussian NLL loss
        "logvar_init": -2.0,
        "logvar_min":  -5.0,
        "logvar_max":   2.0,
        "mc_T": 50,           # Number of Monte Carlo Dropout samples
    }
    return default

def check_paths():
    """Verifies existence of required input data files."""
    ge_hint = DATA_ROOT / DATASET_NAME / "gene_expression.csv"
    fp_dir  = DATA_ROOT / DATASET_NAME / "drug_fingerprints"
    print("\n[PATH CHECK]")
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
    Splits the full dataset using drevalpy objects: 
    1. train: all non-target cell lines (train_set).
    2. test: 100% of samples for each target cell line (full Query Pool).
    """
    training_cells = [c for c in celline_features.identifiers if c not in target_cells]
    test_cells = [c for c in celline_features.identifiers if c in target_cells]
    
    train_set = resp.copy()
    test = resp.copy()

    train_set.reduce_to(cell_line_ids=training_cells)
    test.reduce_to(cell_line_ids=test_cells)

    return train_set, test

# --- Sampling strategies ---

def select_drugs_by_strategy(
    Response: DrugResponseDataset, 
    scores_df: pd.DataFrame,
    N: int, 
    strategy: Literal['epistemic', 'aleatoric', 'random']
) -> np.ndarray: 
    """Selects N drug-cell line combinations (samples) from the pool."""

    if strategy == 'epistemic':
        if 'sigma_epi' not in scores_df.columns:
            raise ValueError("there is no sigma_epi in the Dataframe!")
        selected_samples = scores_df.sort_values(by='sigma_epi', ascending=False).head(N)
        query_drugs = selected_samples['drug_id'].values
        print(f"-> Strategy: Epistemic - Selecting the top {N} samples by sigma_epi.",query_drugs)
    elif strategy == 'aleatoric':
        if'sigma_ale' not in scores_df.columns:
            raise ValueError("there is no sigma_ale in the Dataframe!")
        selected_samples = scores_df.sort_values(by='sigma_ale', ascending=False).head(N)
        query_drugs = selected_samples['drug_id'].values
        print(f"-> Strategy: Aleatoric - Selecting the top {N} samples by sigma_ale.",query_drugs)
    elif strategy == 'random':
        available_drugs = scores_df['drug_id'].unique()
        if len(available_drugs) == 0:
             print("-> Strategy: Random - No drugs left in pool.")
             return np.array([])
        select_n = min(N, len(available_drugs)) 
        query_drugs = np.random.choice(a=available_drugs, size=select_n, replace= False)
    else:
        raise ValueError("Unknown sampling strategy. Please use 'epistemic', 'aleatoric', or 'random'.")

    return query_drugs

def load_helper():
    hyperparameters = {
        "dropout_prob": 0.3, 
        "units_per_layer": [512, 128, 64, 16]
    }
    temp = SimpleNeuralNetwork()
    temp.build_model(hyperparameters) # Use 100 epochs (or whatever is in hparams)

    try:
        # --- 4. Load Data and Clean Features ---
        # Load features (gene expression and drug fingerprints)
        cell_fd = temp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
        drug_fd = temp.load_drug_features(str(DATA_ROOT), DATASET_NAME)
        
        # Load response data (LN_IC50)
        resp = DrugResponseDataset.from_csv(
            input_file=str(RESPONSE_FILE),
            dataset_name=DATASET_NAME,
            measure="LN_IC50_curvecurator",
        )
        resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
        resp.remove_nan_responses()
        print(f"[INFO] total rows = {len(resp)}")
        return cell_fd, drug_fd, resp
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load drevalpy objects: {e}"); sys.exit(1)


# ----------------------------- MAIN -----------------------------
def main():
    # --- 0. Setup and Configuration ---
    # Ensure output and checkpoint directories exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # --- 1. Read Environment Variables ---
    try:
        STRATEGY = os.environ['AL_STRATEGY']
        # Read the list of target cell lines and convert back to a Python list
        ALL_TARGET_CELL_LINES_LIST = os.environ['ALL_TARGET_CELL_LINES_LIST'].split(',') 
    except KeyError as e:
        print(f"[ERROR] Missing required environment variable: {e}")
        sys.exit(1)

    # The TARGET_CELL_LINE variable exists only during AL runs
    TARGET_CELL_LINE = os.environ.get('TARGET_CELL_LINE')

    # --- 2. Define Global Paths ---
    # Path to save/load the *one* global baseline model
    GLOBAL_BASELINE_MODEL_DIR = BASELINE_DIR / "GLOBAL_MODEL" 
    # Path to the *one* CSV file with baseline predictions for all target cells
    # Use the same path as defined in final_evaluation.py
    GLOBAL_BASELINE_PREDS_CSV = BASELINE_DIR / "global_baseline_preds.csv" 

    # --- 3. Load Hparams ---
    hparams = load_hparams()
    cell_fd, drug_fd, resp = load_helper()

    # --- 2. Data Split (Baseline Step) ---
    #all_cell_lines = np.unique(resp.cell_line_ids)
    
    # Randomly select TARGET_CELL_LINES for the current run
    # TARGET_CELL_LINES_CURRENT = np.random.choice(
    #      a=all_cell_lines,
    #      size=NUM_TARGETS,
    #      replace=False 
    #  )

    #TARGET_CELL_LINES_CURRENT = ["NCI-H322M"]

    train_set, test_all_targets = prepare_datasets(resp, cell_fd, ALL_TARGET_CELL_LINES_LIST)

    print(f"\n[INFO] Global Setup: Held-out Target Cell(s): {ALL_TARGET_CELL_LINES_LIST}") 
    print(f"[INFO] Global train_set (Train) size: {len(train_set)} samples")
    print(f"[INFO] Global Test (Target Pool) size: {len(test_all_targets)} samples")
    
    T_mc = hparams.get("mc_T", 50)

    # --- 6. Check Run Mode (Training vs. AL) ---

    if STRATEGY == 'baseline_only':
        # --- 6A. This is the "global baseline training" run ---
        
        print(f"\n[INFO] Running GLOBAL BASELINE TRAINING...")
        model = SimpleNeuralNetwork()
        model.build_model(hparams) # Use 100 epochs (or whatever is in hparams)

        print("\n=== START GLOBAL BASELINE MODEL TRAINING ===")
        model.train(
            output=train_set, # Training set = All non-target cells
            cell_line_input=cell_fd,
            drug_input=drug_fd,
            model_checkpoint_dir=str(CKPT_DIR),
        )
        print("=== GLOBAL BASELINE MODEL TRAINING COMPLETED ===")

        # --- 7A. Baseline prediction for *all* target cells ---
        print(f"\n[INFO] Generating predictions for ALL {len(ALL_TARGET_CELL_LINES_LIST)} target cells...")
        if len(test_all_targets) == 0:
             print("[ERROR] Cannot predict baseline: test_all_targets set is empty! Check prepare_datasets and feature files.")
             sys.exit(1)
             
        mean, sigma_epi, sigma_ale = model.predict_uncertainty_by_ids(
            cell_line_ids=test_all_targets.cell_line_ids, 
            drug_ids=test_all_targets.drug_ids,
            cell_line_input=cell_fd, 
            drug_input=drug_fd,
            T=T_mc,
            keep_bn_eval=True 
        )

        sigma_ale_val = sigma_ale if sigma_ale is not None else np.nan
        if sigma_ale is not None and len(sigma_epi) == len(sigma_ale):
            sigma_tot = np.sqrt(sigma_epi**2 + sigma_ale**2)
        else:
            sigma_tot = sigma_epi 

        output_baseline_df = pd.DataFrame({
            "cell_line": test_all_targets.cell_line_ids,
            "drug_id":   test_all_targets.drug_ids,
            "y_true":    test_all_targets.response, 
            "y_pred":    mean,
            "sigma_epi": sigma_epi,
            "sigma_ale": sigma_ale_val,
            "sigma_tot": sigma_tot,
            "mc_T":      T_mc,
        })
        
        # Save to the *one* global CSV file
        # Ensure the parent directory exists
        GLOBAL_BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True) 
        model.save(str(GLOBAL_BASELINE_MODEL_DIR)) 
        print(f"[SAVE] Global baseline model saved to {GLOBAL_BASELINE_MODEL_DIR}")

        GLOBAL_BASELINE_PREDS_CSV.parent.mkdir(parents=True, exist_ok=True) 
        output_baseline_df.to_csv(GLOBAL_BASELINE_PREDS_CSV, index=False)
        print(f"[SAVE] Global baseline predictions saved to {GLOBAL_BASELINE_PREDS_CSV}")
        print("\n--- Baseline Run Finished ---")
        sys.exit(0) 

    else:
        # --- 6B. This is an "Active Learning" (AL) run ---
        
        # 1. Prepare data for this run
        TARGET_CELL_LINES_CURRENT = [TARGET_CELL_LINE]
        
        # 'test' contains data only for this one cell
        test = test_all_targets.copy() # Start with all target cell data
        test.reduce_to(cell_line_ids=TARGET_CELL_LINES_CURRENT) # Reduce to the current cell

        # 'cellline_train' starts with the *global* training set
        cellline_train = train_set.copy() 
        
        # 2. Load global baseline predictions (only for this cell)
        try:
            df_global_base = pd.read_csv(GLOBAL_BASELINE_PREDS_CSV)
        except FileNotFoundError:
            print(f"[ERROR] Global baseline file not found: {GLOBAL_BASELINE_PREDS_CSV}")
            sys.exit(1)
            
        output_baseline_df = df_global_base[df_global_base['cell_line'] == TARGET_CELL_LINE].copy()
    
# ===================================================================
# ------------------------- AL loop begins --------------------------
# ===================================================================
    all_results_list = [output_baseline_df]
    strategy_folder = OUT_DIR / f"{STRATEGY}_AL_Result"

    # Initialize a DataFrame to hold the scores for the sampling pool.
    # It starts with the baseline scores.
    pool_for_sampling_df = output_baseline_df.copy()

    acquired_pairs = set()
    al_metrics_records = []

    for cellline in TARGET_CELL_LINES_CURRENT:
        target = test.copy()
        target.reduce_to(cell_line_ids=[cellline])
    
        if PREDICT_SCOPE == 'all':
            pred_target = test.copy()                # all drugs for this cellline 
            pred_target.reduce_to(cell_line_ids=[cellline]) 
        else:
            pred_target = target # 仅剩余池

        cellline_train = train_set.copy()

        # model = SimpleNeuralNetwork.load(GLOBAL_BASELINE_MODEL_DIR)
        # model.hyperparameters["max_epochs"] = FINETUNE_EPOCHS
        # model.hyperparameters["lr"]=1e-5
        model = SimpleNeuralNetwork()
        model.build_model(hparams) # Use 100 epochs (or whatever is in hparams)

        pool_for_sampling_this_cell = pool_for_sampling_df[pool_for_sampling_df['cell_line'] == cellline].copy()

        print("rows:", len(pool_for_sampling_this_cell),
        "unique drugs:", pool_for_sampling_this_cell['drug_id'].nunique())

        for cycle in range(MAX_AL_CYCLES):
            drugs_to_add = select_drugs_by_strategy(                
                Response=target,
                scores_df=pool_for_sampling_this_cell,
                N=min(N_BUDGET, len(pool_for_sampling_this_cell)),  # >>> 池不足时自动缩小 
                strategy=STRATEGY
            )
            print (len(drugs_to_add))

            # 把本轮选中的样本标记为“已用于微调” 
            for d in np.asarray(drugs_to_add).ravel():
                acquired_pairs.add((str(cellline), str(d)))

            target._remove_drugs(drugs_to_add)
            
            temp = target.copy()
            temp.reduce_to(drug_ids=drugs_to_add)
            cellline_train.add_rows(temp)
            
            model.train(
                output=cellline_train,
                cell_line_input=cell_fd,
                drug_input=drug_fd,
                warm_start_path= GLOBAL_BASELINE_MODEL_DIR
            )
            
            mean, sigma_epi, sigma_ale = model.predict_uncertainty_by_ids(
                 cell_line_ids=pred_target.cell_line_ids,
                 drug_ids=pred_target.drug_ids,
                 cell_line_input=cell_fd, 
                 drug_input=drug_fd,
                 T=T_mc,
                 keep_bn_eval=True
            )

            # Calculate total uncertainty
            sigma_ale_val = sigma_ale if sigma_ale is not None else np.nan
            if sigma_ale is not None and len(sigma_epi) == len(sigma_ale):
                sigma_tot = np.sqrt(sigma_epi**2 + sigma_ale**2)
            else:
                sigma_tot = sigma_epi

            # new input data for training 
            output_Al_df = pd.DataFrame({
                "cell_line": pred_target.cell_line_ids,
                "drug_id":   pred_target.drug_ids,
                "y_true":    pred_target.response, # True values for later label acquisition (simulation)
                "y_pred":    mean,
                "sigma_epi": sigma_epi,
                "sigma_ale": sigma_ale_val,
                "sigma_tot": sigma_tot,
                "mc_T":      [T_mc]*len(pred_target),  
                "cycle":     [int(cycle+1)]*len(pred_target),   
                "Known_Drug": "TBD"*len(pred_target),
                #(cell_line, drug) pairs that have been used for 
                #fine-tuning will be shown as True, and the rest as False              
            })


            # mark (cell_line, drug) if fine tune
            cl_ser = pd.Series(output_Al_df["cell_line"].astype(str))
            dg_ser = pd.Series(output_Al_df["drug_id"].astype(str))
            output_Al_df["Known_Drug"] = [
                (cl, dg) in acquired_pairs for cl, dg in zip(cl_ser, dg_ser)
            ]

            #Update the pool of scores for the NEXT cycle's selection.
            #The new predictions become the scores for the next round.
            #We also need to filter it down to only the remaining unlabeled samples.
            remaining_ids = pd.DataFrame({'cell_line': target.cell_line_ids, 'drug_id': target.drug_ids})
            pool_for_sampling_this_cell = pd.merge(output_Al_df, remaining_ids, on=['cell_line', 'drug_id'], how='inner')

            # save results for every cycle 
            cycle_results_folder = strategy_folder / "cycles"
            cycle_results_folder.mkdir(parents=True, exist_ok=True)
            
            cycle_filename = f"{cellline}_{STRATEGY}_cycle_{cycle+1}_results.csv"
            cycle_output_path = cycle_results_folder / cycle_filename
            
            output_Al_df.to_csv(cycle_output_path, index=False)
            print(f"saved cycle {cycle} in dir: {cycle_output_path}")
            all_results_list.append(output_Al_df)
    
    #save final results
    final_df = pd.concat(all_results_list, ignore_index=True)
    strategy_folder.mkdir(parents=True, exist_ok=True)
    output_path  = strategy_folder / f"final_{cellline}_{STRATEGY}_active_learning_result.csv"
    final_df.to_csv(output_path, index=False)
    print(f"All results saved (Baseline + Fine-Tuning) -> {output_path}")

if __name__ == "__main__":
    main()