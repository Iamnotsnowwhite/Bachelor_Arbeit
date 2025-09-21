"""
Train a baseline model (aleatoric-enabled) where training/validation data excludes target cells;
Target cells are completely held out for subsequent personalized active learning and evaluation.
"""

from pathlib import Path
import os, sys, json, yaml, random
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
HYPERPARAM_YML = BASE_DIR / "drevalpy" / "models" / "SimpleNeuralNetwork" / "hyperparameters.yaml"

OUT_DIR            = BASE_DIR / "results" / DATASET_NAME
PREV_BASELINE_DIR  = OUT_DIR / "baseline_models" / "exclude_target"         # 旧 MSE baseline（若有则 warm-start）
BASELINE_DIR       = OUT_DIR / "baseline_models_ale" / "exclude_target"     # 新 aleatoric baseline 输出
CKPT_DIR           = BASE_DIR / "checkpoints" / DATASET_NAME / "BASELINE_ALE"

# ----------------------- CONFIG ------------------------
SEED = 42
TARGET_CELL_LINE = "UACC-257"

# ----------------------- UTILS -----------------------------
def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def load_hparams() -> dict:
    if HYPERPARAM_YML.exists():
        y = yaml.safe_load(open(HYPERPARAM_YML, "r"))["SimpleNeuralNetwork"]
        return {
            "dropout_prob":    y["dropout_prob"][0],
            "units_per_layer": y["units_per_layer"][0],
            "max_epochs":      50,
            "batch_size":      32,
            "patience":        5,
            "lr":              1e-3,
            "weight_decay":    1e-4,
            "input_dim_gex":   None,
            "input_dim_fp":    None,
            "aleatoric":       True,   # ★ 打开 aleatoric
        }
    return {
        "dropout_prob": 0.3,
        "units_per_layer": [32, 16, 8, 4],
        "max_epochs": 50,
        "batch_size": 32,
        "patience": 5,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "input_dim_gex": None,
        "input_dim_fp": None,
        "aleatoric": True,            # ★
    }

def check_paths():
    ge_hint = DATA_ROOT / DATASET_NAME / "gene_expression.csv"
    fp_dir  = DATA_ROOT / DATASET_NAME / "drug_fingerprints"
    print("\n[PATH CHECK]")
    print(" gene expr (hint):", ge_hint, ge_hint.exists())
    print(" fp dir    (hint):", fp_dir,  fp_dir.exists())
    print(" response       :", RESPONSE_FILE, RESPONSE_FILE.exists(), "\n")
    if not RESPONSE_FILE.exists():
        print("[ERROR] Response file not found."); sys.exit(1)

def clean_view(fd: FeatureDataset, view: str, clip_min=None, clip_max=None) -> Dict[str, int]:
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
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    pr  = 0.0 if (y_true.std()<1e-12 or y_pred.std()<1e-12) else float(np.corrcoef(y_true, y_pred)[0,1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}

def subset_by_idx(ds: DrugResponseDataset, idx: np.ndarray, name: str) -> DrugResponseDataset:
    return DrugResponseDataset(
        response=ds.response[idx],
        cell_line_ids=ds.cell_line_ids[idx],
        drug_ids=ds.drug_ids[idx],
        tissues=ds.tissue[idx] if ds.tissue is not None else None,
        predictions=None,
        dataset_name=name,
    )

def prepare_datasets(resp: DrugResponseDataset, target_cell: str, seed:int=SEED
                    ) -> Tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset]:
    non_target_idx = np.where(resp.cell_line_ids != target_cell)[0]
    target_idx     = np.where(resp.cell_line_ids == target_cell)[0]
    if len(non_target_idx) == 0:
        raise RuntimeError(f"No non-target data after excluding {target_cell}.")
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(non_target_idx, test_size=0.2, random_state=seed, shuffle=True)
    return (
        subset_by_idx(resp, tr_idx, name="baseline_train"),
        subset_by_idx(resp, va_idx, name="baseline_val"),
        subset_by_idx(resp, target_idx, name=f"holdout_{target_cell}")
    )

# ----------------------------- MAIN -----------------------------
def main():
    set_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # --- Features ---
    hparams = load_hparams()
    tmp = SimpleNeuralNetwork(); tmp.build_model(hparams)
    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    ge_stats = clean_view(cell_fd, "gene_expression", clip_min=-50.0, clip_max=50.0)
    fp_stats = clean_view(drug_fd, "fingerprints",    clip_min=0.0,   clip_max=1.0)
    print("[CLEAN] gene_expression:", ge_stats)
    print("[CLEAN] fingerprints   :", fp_stats)

    # --- Response data ---
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] total rows = {len(resp)}")
    print(f"[INFO] target '{TARGET_CELL_LINE}' rows = {np.sum(resp.cell_line_ids==TARGET_CELL_LINE)}")

    # --- Splits ---
    train_ds, val_ds, target_ds = prepare_datasets(resp, TARGET_CELL_LINE)
    print(f"[INFO] train(non-target) = {len(train_ds)}, val = {len(val_ds)}, target = {len(target_ds)}")

    # --- Train (aleatoric on) ---
    print(f"\n=== TRAIN BASELINE (ALEATORIC=ON, exclude: {TARGET_CELL_LINE}) ===")
    model = SimpleNeuralNetwork()
    model.build_model(dict(hparams))

    warm_path = str(PREV_BASELINE_DIR) if (PREV_BASELINE_DIR / "model.pt").exists() else None
    if warm_path:
        print(f"[INFO] Will warm-start from: {warm_path}")
    else:
        print("[INFO] No previous baseline to warm-start. Training from scratch.")

    model.train(
        output=train_ds,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        output_earlystopping=val_ds,
        model_checkpoint_dir=str(CKPT_DIR),
        warm_start_path=warm_path,  # ★ 交给 train() 内部在构建完网络后加载
    )

    # --- Evaluation ---
    print("\n=== EVALUATION ===")
    val_pred = model.predict(val_ds.cell_line_ids, val_ds.drug_ids, cell_fd, drug_fd)
    val_metrics = metrics_all(val_ds.response, val_pred)
    print("Validation (non-target) Metrics:", val_metrics)

    if len(target_ds) > 0:
        tgt_pred = model.predict(target_ds.cell_line_ids, target_ds.drug_ids, cell_fd, drug_fd)
        tgt_metrics = metrics_all(target_ds.response, tgt_pred)
        print(f"Target ({TARGET_CELL_LINE}) Metrics:", tgt_metrics)
    else:
        tgt_metrics = {}
        print(f"No rows for target cell {TARGET_CELL_LINE}")

    # --- Save ---
    model.save(str(BASELINE_DIR))
    with open(BASELINE_DIR / "baseline_results.json", "w") as f:
        json.dump({
            "target_cell": TARGET_CELL_LINE,
            "val_metrics": val_metrics,
            "target_metrics": tgt_metrics,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "target_size": len(target_ds),
            "hyperparams": hparams,
            "warm_start_from": warm_path,
        }, f, indent=2)

    pd.DataFrame([
        {"split":"validation", **val_metrics},
        {"split":f"target:{TARGET_CELL_LINE}", **({k:v for k,v in tgt_metrics.items()} if tgt_metrics else {})}
    ]).to_csv(BASELINE_DIR / "baseline_metrics.csv", index=False)

    print("\n BASELINE (ALEATORIC) TRAINING COMPLETED")
    print("  - model dir :", BASELINE_DIR)
    print("  - metrics   :", BASELINE_DIR / "baseline_metrics.csv")
    print("  - json      :", BASELINE_DIR / "baseline_results.json")

if __name__ == "__main__":
    main()
