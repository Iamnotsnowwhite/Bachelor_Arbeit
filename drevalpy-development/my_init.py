#!/usr/bin/env python3
# run_pal_random.py
"""
Personalized Active Learning (PAL) — Random sampling only (no aleatoric/epistemic).
- LCO CV：目标细胞只出现在 test；候选池来自 test（模拟对新细胞做实验）。
- 每轮：基线训练集 = 其它细胞(train) + 该细胞已选样本(L)；在未选样本(U)上评估；从 U 随机选 QUERY_SIZE 加入 L。
- 输出：逐轮指标、聚合学习曲线、Markdown 报告。
"""

from pathlib import Path
import os, sys, json, yaml
import numpy as np
import pandas as pd
from typing import Dict, List

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"   # 里有 data/GDSC2/gene_expression.csv、drug_fingerprints/、GDSC2.csv
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
HYPERPARAM_YML = BASE_DIR / "drevalpy" / "models" / "SimpleNeuralNetwork" / "hyperparameters.yaml"

OUT_DIR        = BASE_DIR / "results" / DATASET_NAME
PAL_DIR        = OUT_DIR / "pal_random"
CKPT_DIR       = BASE_DIR / "checkpoints" / DATASET_NAME / "PAL_RANDOM"

# ----------------------- PAL CONFIG ------------------------
CV_MODE   = "LCO"     # Leave-Cell-Out：保证“个性化”效果（目标细胞仅在 test）
N_SPLITS  = 5
SEED      = 42

INIT_SIZE  = 16       # 初始标注的药物对数量（目标细胞内部）
QUERY_SIZE = 8        # 每轮新增
N_ROUNDS   = 10       # 主动学习轮数

STRATEGY_NAME = "Random"   # 仅随机采样
TARGET_CELLS_LIMIT = None  # 可设整数限制每折目标细胞数量（调试用）

# ----------------------- UTILS -----------------------------
def load_hparams() -> dict:
    """统一的 SNN 超参数（保持与其他策略可比）。"""
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
    """轻量清洗：替换 NaN/Inf、可选裁剪、转 float32。"""
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
    """常用指标：MSE, RMSE, MAE, R2, Pearson。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    pr  = 0.0 if (y_true.std()<1e-12 or y_pred.std()<1e-12) else float(np.corrcoef(y_true, y_pred)[0,1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}

def subset_dr(ds: DrugResponseDataset, idx: np.ndarray, name: str) -> DrugResponseDataset:
    """从 DrugResponseDataset 按行索引抽子集。"""
    return DrugResponseDataset(
        response=ds.response[idx],
        cell_line_ids=ds.cell_line_ids[idx],
        drug_ids=ds.drug_ids[idx],
        tissues=ds.tissue[idx] if ds.tissue is not None else None,
        predictions=None,
        dataset_name=name,
    )

def filter_by_cell(ds: DrugResponseDataset, cell: str) -> np.ndarray:
    """返回属于某个 cell 的行索引数组。"""
    return np.where(ds.cell_line_ids == cell)[0]

def train_snn(hparams: dict, train_ds: DrugResponseDataset,
              cell_fd: FeatureDataset, drug_fd: FeatureDataset,
              ckpt_dir: Path, seed: int) -> SimpleNeuralNetwork:
    """从头训练一个 SNN；每轮都重新训练以保证可比性。"""
    np.random.seed(seed)
    m = SimpleNeuralNetwork()
    m.build_model(dict(hparams))
    m.train(
        output=train_ds,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        output_earlystopping=None,
        model_checkpoint_dir=str(ckpt_dir),
    )
    return m

def predict_snn(model: SimpleNeuralNetwork, ds: DrugResponseDataset,
                cell_fd: FeatureDataset, drug_fd: FeatureDataset) -> np.ndarray:
    """对给定样本集做预测，返回 shape=[n]。"""
    return model.predict(
        cell_line_ids=ds.cell_line_ids,
        drug_ids=ds.drug_ids,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
    ).reshape(-1)

# ----------------------- PAL for ONE CELL (Random only) -----------------------
def pal_random_for_cell(cell_id: str,
                        split: dict,
                        hparams: dict,
                        cell_fd: FeatureDataset,
                        drug_fd: FeatureDataset,
                        ckpt_base: Path,
                        seed: int) -> List[Dict]:
    """
    针对单个目标细胞的 PAL 随机采样循环。
    - base_train = 其它细胞（来自 train）
    - pool/候选集 = 目标细胞在 test 里的所有 (cell, drug) 对（LCO 下目标只在 test）
    - 每轮：随机从 U 选 QUERY_SIZE 加入 L，重训，评估 U。
    """
    rng = np.random.default_rng(seed)

    # 1) 其它细胞作为基础训练集（来自 train）
    train_all = split["train"]
    base_idx  = np.where(train_all.cell_line_ids != cell_id)[0]
    base_train = subset_dr(train_all, base_idx, "BASE_TRAIN")

    # 2) 目标细胞的候选池（来自 test！）
    test_all = split["test"]
    tgt_idx  = filter_by_cell(test_all, cell_id)
    if len(tgt_idx) == 0:
        return []  # 安全兜底：该折里没有这个细胞

    tgt_all = subset_dr(test_all, tgt_idx, f"{cell_id}_ALL")  # 池子
    pool_idx = np.arange(len(tgt_all))

    # 3) 初始化 L / U
    init_k  = min(INIT_SIZE, len(pool_idx))
    L_local = rng.choice(pool_idx, size=init_k, replace=False)
    U_local = np.setdiff1d(pool_idx, L_local, assume_unique=False)

    records = []

    # 4) 主循环：包含 r=0 的“冷启动评估”
    for r in range(N_ROUNDS + 1):
        # 4.1 当前 L 子集
        L_ds = subset_dr(tgt_all, L_local, f"{cell_id}_L{r}")

        # 4.2 合并训练集
        merged_train = base_train.copy()
        if len(L_ds) > 0:
            merged_train.add_rows(L_ds)

        # 4.3 训练
        model_r = train_snn(
            hparams=hparams,
            train_ds=merged_train,
            cell_fd=cell_fd,
            drug_fd=drug_fd,
            ckpt_dir=ckpt_base / f"round_{r}",
            seed=seed + r,
        )

        # 4.4 在 U 上评估
        U_ds = subset_dr(tgt_all, U_local, f"{cell_id}_U{r}")
        if len(U_ds) > 0:
            y_true = U_ds.response
            y_pred = predict_snn(model_r, U_ds, cell_fd, drug_fd)
            mets = metrics_all(y_true, y_pred)
        else:
            mets = {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Pearson": np.nan}

        mets.update({"cell_id": cell_id, "round": r, "labeled_size": int(len(L_local))})
        records.append(mets)

        # 4.5 结束条件
        if r == N_ROUNDS or len(U_local) == 0:
            break

        # 4.6 随机选下一批
        k = min(QUERY_SIZE, len(U_local))
        new_local = rng.choice(U_local, size=k, replace=False)
        L_local = np.union1d(L_local, new_local)
        U_local = np.setdiff1d(U_local, new_local, assume_unique=False)

    return records

# ----------------------------- MAIN -----------------------------
def main():
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # --- 加载特征（drevalpy 自带 loader） ---
    tmp = SimpleNeuralNetwork()
    hparams = load_hparams()
    tmp.build_model(hparams)

    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    ge_stats = clean_view(cell_fd, "gene_expression", clip_min=-50.0, clip_max=50.0)
    fp_stats = clean_view(drug_fd, "fingerprints",    clip_min=0.0,   clip_max=1.0)
    print("[CLEAN] gene_expression:", ge_stats)
    print("[CLEAN] fingerprints   :", fp_stats)

    # --- 加载响应并对齐 ---
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] rows after filtering = {len(resp)}")
    if len(resp) == 0:
        print("[ERROR] empty dataset after filtering."); sys.exit(1)

    # --- LCO 拆分（drevalpy 内置） ---
    splits = resp.split_dataset(
        n_cv_splits=N_SPLITS,
        mode=CV_MODE,
        split_validation=True,
        split_early_stopping=True,
        validation_ratio=0.1,
        random_state=SEED,
    )

    # --- 逐折逐细胞执行 PAL（随机） ---
    all_records = []
    print(f"\n=== STRATEGY: {STRATEGY_NAME} ===")
    for fi, split in enumerate(splits):
        test_cells = np.unique(split["test"].cell_line_ids)
        if TARGET_CELLS_LIMIT is not None:
            test_cells = test_cells[:TARGET_CELLS_LIMIT]
        print(f"  Fold {fi+1}/{len(splits)}: {len(test_cells)} target cells")

        for cid in test_cells:
            print(f"    - Target cell: {cid}")
            recs = pal_random_for_cell(
                cell_id=cid,
                split=split,
                hparams=hparams,
                cell_fd=cell_fd.copy(),
                drug_fd=drug_fd.copy(),
                ckpt_base=CKPT_DIR / f"fold{fi}" / cid,
                seed=SEED + fi,
            )
            for r in recs:
                r.update({"strategy": STRATEGY_NAME, "fold": fi})
            all_records.extend(recs)

    # --- 导出 ---
    df = pd.DataFrame(all_records)
    if df.empty:
        print("[WARN] No PAL records generated. Check data splits.")
        (PAL_DIR / "pal_metrics.csv").write_text("")
        (PAL_DIR / "pal_summary.csv").write_text("")
        (PAL_DIR / "pal_report.md").write_text("# PAL (Random) Report\n\nNo records generated.\n")
        return

    df.to_csv(PAL_DIR / "pal_metrics.csv", index=False)

    summary = df.groupby(["strategy", "round"]).agg(
        labeled_size=("labeled_size", "mean"),
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        Pearson_mean=("Pearson", "mean"), Pearson_std=("Pearson", "std"),
    ).reset_index()
    summary.to_csv(PAL_DIR / "pal_summary.csv", index=False)

    with open(PAL_DIR / "pal_report.md", "w") as f:
        f.write(f"# Personalized Active Learning (Random) – {DATASET_NAME}\n\n")
        f.write(f"- CV mode: **{CV_MODE}**, folds: **{N_SPLITS}**\n")
        f.write(f"- Strategy: **{STRATEGY_NAME}**\n")
        f.write(f"- AL params: INIT={INIT_SIZE}, QUERY={QUERY_SIZE}, ROUNDS={N_ROUNDS}\n\n")
        f.write("## Aggregated Learning Curves\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Sample Records (head)\n\n")
        f.write(df.head(30).to_markdown(index=False))
        f.write("\n")

    with open(PAL_DIR / "pal_run_config.json", "w") as f:
        json.dump({
            "dataset": DATASET_NAME,
            "cv_mode": CV_MODE,
            "n_splits": N_SPLITS,
            "seed": SEED,
            "al": {
                "init_size": INIT_SIZE,
                "query_size": QUERY_SIZE,
                "rounds": N_ROUNDS,
                "strategy": STRATEGY_NAME,
                "personalized": True
            },
            "paths": {
                "data_root": str(DATA_ROOT),
                "response_file": str(RESPONSE_FILE),
                "out_dir": str(OUT_DIR),
                "pal_dir": str(PAL_DIR),
            },
            "hyperparams_SNN": load_hparams(),
        }, f, indent=2)

    print("\n PAL (Random) finished.")
    print("   - metrics:", PAL_DIR / "pal_metrics.csv")
    print("   - summary:", PAL_DIR / "pal_summary.csv")
    print("   - report :", PAL_DIR / "pal_report.md")

if __name__ == "__main__":
    main()
