from pathlib import Path
import json
import argparse  
import numpy as np
import pandas as pd
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# --- All your utility functions (metrics_all, prepare_datasets, etc.) ---
# (I've included the necessary ones below)
def metrics_all(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    # ... (add other metrics if you need them)
    return {"MSE": mse, "RMSE": rmse}

def prepare_datasets(resp, celline_features, target_cells) -> tuple:
    training_cells = [c for c in celline_features.identifiers if c not in target_cells]
    # For HPO, we need a validation set. Let's split the training cells.
    np.random.shuffle(training_cells)
    train_split_idx = int(len(training_cells) * 0.9) # 90/10 split
    train_cells = training_cells[:train_split_idx]
    val_cells = training_cells[train_split_idx:]
    
    train_set = resp.copy()
    val_set = resp.copy()
    test_set = resp.copy()

    train_set.reduce_to(cell_line_ids=train_cells)
    val_set.reduce_to(cell_line_ids=val_cells)
    test_set.reduce_to(cell_line_ids=target_cells)

    return train_set, val_set, test_set

# ======================== OPTUNA OBJECTIVE FUNCTION ========================
def objective(trial: optuna.Trial, train_data, val_data, cell_fd, drug_fd) -> float:
    """Defines a single trial for Optuna."""
    try:
        # --- 1. Define Hyperparameter Search Space ---
        hparams = {
            "max_epochs": 100, 
            "aleatoric": True,
            "mc_T": 50,
            # Tunable Hyperparameters
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout_prob": trial.suggest_float("dropout_prob", 0.2, 0.6 , step=0.1),
            "batch_size": trial.suggest_categorical("batch_size", [256]),
        }
        n_layers = trial.suggest_int("n_layers", 2, 4)
        first_layer_units = trial.suggest_categorical("first_layer_units", [128, 256, 512, 1024])
        layers = [first_layer_units]
        for i in range(n_layers - 1):
            next_units = max(layers[-1] // 2, 32)
            layers.append(next_units)
        hparams["units_per_layer"] = layers

        # --- 2. Setup Pruning Callback ---
        # Monitor validation loss and prune if it's not promising.
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        # --- 3. Build and Train the Model ---
        model = SimpleNeuralNetwork()
        model.build_model(dict(hparams))
        
        model.train(
            output=train_data,
            cell_line_input=cell_fd,
            drug_input=drug_fd,
        )
        
        # --- 4. Evaluate and Return Metric ---
        # Evaluate on the validation set to get the score for Optuna
        mean, _, _ = model.predict_uncertainty_by_ids(
            cell_line_ids=val_data.cell_line_ids,
            drug_ids=val_data.drug_ids,
            cell_line_input=cell_fd, 
            drug_input=drug_fd,
            T=hparams["mc_T"],
        )
        metrics = metrics_all(val_data.response, mean)
        validation_rmse = metrics["RMSE"]

    except Exception as e:
        # If a trial fails, report it as pruned and continue
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return validation_rmse

# ======================== MAIN DRIVER SCRIPT ========================
def main(args):
    # --- Define Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    STRATEGY_NAME = "baseline"
    DATASET_NAME = "GDSC2"
    DATA_ROOT = BASE_DIR / "data"
    RESPONSE_FILE = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
    OUT_DIR = BASE_DIR / "optuna_result"/ STRATEGY_NAME

    # --- 1. ONE-TIME SETUP: Load and Prepare Data ---
    print("--- Loading and preparing data once... ---")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SimpleNeuralNetwork()
    tmp.build_model({"aleatoric": True})
    
    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)
    resp = DrugResponseDataset.from_csv(input_file=str(RESPONSE_FILE), dataset_name=DATASET_NAME, measure="LN_IC50_curvecurator")
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    
    # Use a fixed target cell line for all trials for fair comparison
    np.random.seed(42)
    TARGET_CELL_LINES = np.random.choice(np.unique(resp.cell_line_ids), size= 40, replace=False)
    
    # We need a validation set for Optuna, so the split logic is slightly different
    train_data, val_data, test_data = prepare_datasets(resp, cell_fd, TARGET_CELL_LINES)
    print(f"Data loaded. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Held-out target cell line(s): {TARGET_CELL_LINES}")

    # --- 2. OPTUNA STUDY SETUP ---
    study_name = "gdsc2_snn_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner= None,
        load_if_exists=True
    )

    # --- 3. RUN OPTIMIZATION ---
    objective_with_data = lambda trial: objective(trial, train_data, val_data, cell_fd, drug_fd)
    study.optimize(objective_with_data, n_trials=args.n_trials)

    # --- 4. SHOW AND SAVE RESULTS ---
    print("\n\n" + "="*50)
    print(" OPTIMIZATION FINISHED ".center(50, "="))
    print("="*50)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Validation RMSE): {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params to a file
    best_params_file = OUT_DIR / f"{study_name}_best_params.json"
    with open(best_params_file, "w") as f:
        json.dump(trial.params, f, indent=4)
    print(f"\nBest parameters saved to: {best_params_file}")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100, # Anzahl von Trials
        help="The number of optimization trials to run."
    )
    args = parser.parse_args()
    
    main(args)