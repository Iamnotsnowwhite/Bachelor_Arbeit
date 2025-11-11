"""
-  this script uses CV - LCO

- Loads the GDSC2 dataset (responses, gene expression, and drug fingerprints).
- Cleans features (handles NaN/Inf, clips extreme values).
- Uses drevalpy’s SimpleNeuralNetwork with aleatoric uncertainty enabled.
- Leaves out one target cell line completely (not used in training or validation).
- Splits the remaining data into train and validation sets.
- Trains the model on train set, uses validation for early stopping.
- Evaluates performance on validation and the held-out target cell line.
- Saves model, metrics, and results for later comparison
"""

from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork


# ───────────── Configuration ───────────── #

# 数据与路径
BASE_DIR        = Path(__file__).resolve().parent
DATASET_NAME    = "GDSC2"
DATA_ROOT       = BASE_DIR / "data"
RESPONSE_FILE   = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
HYPERPARAM_YML  = BASE_DIR / "drevalpy" / "models" / "SimpleNeuralNetwork" / "hyperparameters.yaml"

# 输出与检查点
OUT_DIR         = BASE_DIR / "results" / DATASET_NAME
#PREV_BASELINE   = OUT_DIR / "baseline_models"      / "exclude_target"      # 旧 MSE 基线（若存在，用于 warm-start）
CUR_BASELINE    = OUT_DIR / "new_baseline_models_ale"  / "new_exclude_target"      # 本次 aleatoric 基线输出
CKPT_DIR        = BASE_DIR / "checkpoints" / DATASET_NAME / "BASELINE_ALE"

# 实验设置（固定协议）
SEED            = 42
TARGET_CELL     = "Kelly"
#"Kelly"
#"SiSo"
#"WM115" 0.68
#"HeLa" 0.66
#"LOX-IMVI" 0.68
#"KYSE-450" 0.75
#"NCI-H290" 0.68
#"UACC-257" 0.64

# nur für CV -LCO
#VAL_RATIO       = 0.2       # 训练集再切验证比例（固定）
#N_FOLDS         = 5         # LCO 折数（固定）


# ───────────── Utilities ───────────── #

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_hparams() -> Dict:
    """从 hyperparameters.yaml 读取网络结构与 Dropout，其它用默认值；显式开启 aleatoric。"""
    defaults = dict(
        dropout_prob=0.3,
        units_per_layer=[32, 16, 8, 4],
        max_epochs=50,
        batch_size=32,
        patience=5,
        lr=1e-3,
        weight_decay=1e-4,
        input_dim_gex=None,
        input_dim_fp=None,
        aleatoric= False,
    )
    if HYPERPARAM_YML.exists():
        y = yaml.safe_load(open(HYPERPARAM_YML, "r"))["SimpleNeuralNetwork"]
        return {
            **defaults,
            "dropout_prob":    y["dropout_prob"][0],
            "units_per_layer": y["units_per_layer"][0],
        }
    return defaults

def clean_view(fd: FeatureDataset, view: str,
               clip_min: Optional[float] = None,
               clip_max: Optional[float] = None) -> Dict[str, int]:
    """NaN/Inf → 0，可选裁剪；返回统计信息。"""
    stats = {"nan": 0, "posinf": 0, "neginf": 0, "clipped": 0}
    for _id in fd.identifiers:
        v = fd.features[_id][view].astype(np.float64, copy=False)
        stats["nan"]    += int(np.isnan(v).sum())
        stats["posinf"] += int(np.isposinf(v).sum())
        stats["neginf"] += int(np.isneginf(v).sum())
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if (clip_min is not None) and (clip_max is not None):
            before = v.copy()
            v = np.clip(v, clip_min, clip_max)
            stats["clipped"] += int((before != v).sum())
        fd.features[_id][view] = v.astype(np.float32, copy=False)
    return stats

def metrics_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    pr  = 0.0 if (y_true.std()<1e-12 or y_pred.std()<1e-12) else float(np.corrcoef(y_true, y_pred)[0,1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}

def subset_by_mask(ds: DrugResponseDataset, mask: np.ndarray, name: str) -> DrugResponseDataset:
    idx = np.where(mask)[0]
    return DrugResponseDataset(
        response=ds.response[idx],
        cell_line_ids=ds.cell_line_ids[idx],
        drug_ids=ds.drug_ids[idx],
        tissues=ds.tissue[idx] if ds.tissue is not None else None,
        predictions=None,
        dataset_name=name,
    )

def pick_lco_fold_with_target(
    resp: DrugResponseDataset,
    target_cell: str,
    n_folds: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset]:
    """
    使用内置 LCO（按 cell_line 分组）生成 n_folds 折，
    选中“测试集包含目标细胞系”的那一折：
      • 训练/验证：该折的 train / validation（天然不含目标细胞）
      • 测试：该折 test 中仅筛出目标细胞样本
    """
    cv_splits = resp.split_dataset(
        n_cv_splits=n_folds,
        mode="LCO",
        split_validation=True,
        split_early_stopping=False,
        validation_ratio=val_ratio,
        random_state=seed,
    )

    chosen = None
    for fold in cv_splits:
        if target_cell in np.unique(fold["test"].cell_line_ids):
            chosen = fold
            break
    if chosen is None:
        raise RuntimeError(f"[LCO] No fold whose test set contains target cell '{target_cell}'.")

    train_ds = chosen["train"]
    #val_ds   = chosen.get("validation", chosen["train"])  # 极端容错
    test_ds  = chosen["test"]

    mask_tgt = (test_ds.cell_line_ids == target_cell)
    tgt_ds   = subset_by_mask(test_ds, mask_tgt, name=f"holdout_{target_cell}")

    return train_ds, val_ds, tgt_ds

def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    keys = ["RMSE", "MAE", "R2", "Pearson"]
    msg = "  ".join([f"{k}={metrics.get(k, float('nan')):.3f}" for k in keys])
    print(f"{title}: {msg}")


# ───────────── Pipeline ───────────── #

def main() -> None:
    # 0) init
    set_seeds(SEED)
    for p in [OUT_DIR, CUR_BASELINE, CKPT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) 特征
    hparams = load_hparams()
    tmp = SimpleNeuralNetwork(); tmp.build_model(hparams)
    cell_fd: FeatureDataset = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd: FeatureDataset = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    ge_stats = clean_view(cell_fd, "gene_expression", clip_min=-50.0, clip_max=50.0)
    fp_stats = clean_view(drug_fd, "fingerprints",    clip_min=0.0,   clip_max=1.0)
    print("[CLEAN] gene_expression:", ge_stats)
    print("[CLEAN] fingerprints   :", fp_stats)

    # 2) 响应数据
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()

    n_cells = len(np.unique(resp.cell_line_ids))
    n_drugs = len(np.unique(resp.drug_ids))
    print(f"[INFO] rows={len(resp)}  unique_cells={n_cells}  unique_drugs={n_drugs}")
    print(f"[INFO] target '{TARGET_CELL}' rows = {int(np.sum(resp.cell_line_ids==TARGET_CELL))}")

    # 3) LCO（固定 n_folds=5，val_ratio=0.2）
    train_ds, val_ds, target_ds = pick_lco_fold_with_target(
        resp=resp,
        target_cell=TARGET_CELL,
        n_folds=N_FOLDS,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    print(f"[SPLIT] train(non-target)={len(train_ds)}  val(non-target)={len(val_ds)}  target(test)={len(target_ds)}")

    # 4) 训练（Aleatoric=ON；可 warm-start）
    print(f"\n=== TRAIN BASELINE (ALEATORIC=ON, exclude: {TARGET_CELL}) ===")
    model = SimpleNeuralNetwork()
    model.build_model(dict(hparams))

    #warm_path = str(PREV_BASELINE) if (PREV_BASELINE / "model.pt").exists() else None
    #print(f"[INIT] warm_start_from: {warm_path if warm_path else 'None (scratch)'}")

    model.train(
        output=train_ds,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        output_earlystopping=val_ds,
        model_checkpoint_dir=str(CKPT_DIR),
        #warm_start_path=warm_path,
    )

    # 5) 评估
    print("\n=== EVALUATION ===")
    val_pred = model.predict(val_ds.cell_line_ids, val_ds.drug_ids, cell_fd, drug_fd)
    val_metrics = metrics_all(val_ds.response, val_pred)
    print_metrics("Validation (non-target)", val_metrics)

    if len(target_ds) > 0:
        tgt_pred = model.predict(target_ds.cell_line_ids, target_ds.drug_ids, cell_fd, drug_fd)
        tgt_metrics = metrics_all(target_ds.response, tgt_pred)
        print_metrics(f"Target ({TARGET_CELL})", tgt_metrics)
    else:
        tgt_metrics = {}
        print(f"[WARN] Chosen fold has no rows for target cell '{TARGET_CELL}' after filtering.")

    # 6) 落盘
    model.save(str(CUR_BASELINE))

    (CUR_BASELINE / "new_baseline_results.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "dataset": DATASET_NAME,
                "target_cell": TARGET_CELL,
                "lco": {"n_cv_splits": N_FOLDS, "val_ratio": VAL_RATIO},
                "splits": {
                    "train_size": len(train_ds),
                    "val_size": len(val_ds),
                    "target_size": len(target_ds),
                },
                "val_metrics": val_metrics,
                "target_metrics": tgt_metrics,
                "hyperparams": hparams,
                #_start_from": warm_path,
                "used_builtin_lco": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"split": "validation_non_target", **val_metrics},
            {"split": f"target_holdout:{TARGET_CELL}", **({k: v for k, v in tgt_metrics.items()} if tgt_metrics else {})},
        ]
    ).to_csv(CUR_BASELINE / "new_baseline_metrics.csv", index=False)

    print("\n[DONE] Single-target LCO baseline completed.")
    print(f"  • model dir : {CUR_BASELINE}")
    print(f"  • metrics   : {CUR_BASELINE / 'new_baseline_metrics.csv'}")
    print(f"  • summary   : {CUR_BASELINE / 'new_baseline_results.json'}")

if __name__ == "__main__":
    main()