"""
个性化主动学习 (Personalized Active Learning, PAL) for Drug Response Prediction (GDSC2).

核心思想：
- 使用 LCO (Leave-Cell-Out) 交叉验证，把一批细胞划到 test 中。
- 对 test 中的“每一个目标细胞”，仅在该细胞 药物的组合上做主动学习（模拟逐步做实验）。
- 训练集始终由两部分组成：
  (1) 其他细胞的所有已知样本（提供“先验/总体知识”）；
  (2) 目标细胞当前已选出的少量样本（逐轮扩充）。
- 每一轮主动学习：选择新的药物对加入到(2)，重新训练，再在目标细胞“尚未选中的药物”上评估，
  得到个性化学习曲线。

输出：
  results/GDSC2/pal/pal_metrics.csv    （逐折、逐细胞、逐轮的记录）
  results/GDSC2/pal/pal_summary.csv    （跨折/跨细胞聚合的学习曲线）
  results/GDSC2/pal/pal_report.md      （Markdown 报告）
"""

from pathlib import Path
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

# -------------------------- PATHS --------------------------
BASE_DIR       = Path(__file__).resolve().parent
DATASET_NAME   = "GDSC2"
DATA_ROOT      = BASE_DIR / "data"   # 存放 gene_expression.csv, drug_fingerprints, GDSC2.csv
RESPONSE_FILE  = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
HYPERPARAM_YML = BASE_DIR / "drevalpy" / "models" / "SimpleNeuralNetwork" / "hyperparameters.yaml"

OUT_DIR        = BASE_DIR / "results" / DATASET_NAME
PAL_DIR        = OUT_DIR / "pal"
CKPT_DIR       = BASE_DIR / "checkpoints" / DATASET_NAME


# ----------------------- PAL CONFIG ------------------------
CV_MODE     = "LCO"   # Leave-Cell-Out，保证“个性化”效果
N_SPLITS    = 5       # 5 折交叉验证
SEED        = 42      # 固定随机数种子，保证可复现

INIT_SIZE   = 16      # 初始已知的药物对数量（目标细胞内部）
QUERY_SIZE  = 8       # 每轮新增数量
N_ROUNDS    = 10      # AL 迭代轮数
EPI_ENSEMBLE = 3      # Epistemic ensemble 大小（模型数）；越大不确定性估计越稳但更慢

STRATEGIES  = ["Random", "EPI-EnsVar"]   # 采样策略：随机 / 集成方差（近似 EPI）
TARGET_CELLS_LIMIT = None   # 限制每折测试的目标细胞数；None 表示用该折所有 test 细胞


# ----------------------- HELPERS（前半部分） -----------------------
def load_hparams() -> dict:
    """加载超参数。如果存在 hyperparameters.yaml，就取里面的；否则用默认。"""
    if HYPERPARAM_YML.exists():
        with open(HYPERPARAM_YML, "r") as f:
            y = yaml.safe_load(f)["SimpleNeuralNetwork"]
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
    # 默认参数（确保脚本能跑起来）
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
    """检查 gene expression / fingerprints / response 文件是否存在。"""
    ge_hint = DATA_ROOT / DATASET_NAME / "gene_expression.csv"
    fp_dir  = DATA_ROOT / DATASET_NAME / "drug_fingerprints"
    print("\n[PATH CHECK]")
    print(" gene expr (hint):", ge_hint, ge_hint.exists())
    print(" fp dir    (hint):", fp_dir,  fp_dir.exists())
    print(" response       :", RESPONSE_FILE, RESPONSE_FILE.exists(), "\n")
    if not RESPONSE_FILE.exists():
        print("[ERROR] Response file not found."); sys.exit(1)


def clean_view(fd: FeatureDataset, view: str,
               clip_min=None, clip_max=None) -> Dict[str, int]:
    """
    清洗特征数据：替换 NaN / Inf，裁剪极端值，转 float32。
    返回清洗统计信息，便于你在日志里检查数据质量。
    """
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
    """计算多种评估指标 (MSE, RMSE, MAE, R², Pearson)。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2  = float(1 - ss_res / (ss_tot + 1e-12))
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        pr = 0.0
    else:
        pr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pr}


# ----------------------- HELPERS（新增：数据子集与训练） -----------------------
def subset_dr(ds: DrugResponseDataset, idx: np.ndarray, name: str) -> DrugResponseDataset:
    """
    从 DrugResponseDataset 中抽取一个“行子集”，返回一个新的数据集对象。
    - 典型用途：构建“目标细胞的已标注集合 L”、“未标注集合 U”等。
    """
    return DrugResponseDataset(
        response=ds.response[idx],
        cell_line_ids=ds.cell_line_ids[idx],
        drug_ids=ds.drug_ids[idx],
        tissues=ds.tissue[idx] if ds.tissue is not None else None,
        predictions=None,
        dataset_name=name,
    )


def filter_by_cell(ds: DrugResponseDataset, cell: str) -> np.ndarray:
    """
    返回给定数据集中，属于某个具体 cell_line 的“行索引数组”。
    - 典型用途：定位某个目标细胞在 train/test 中的所有药物对。
    """
    return np.where(ds.cell_line_ids == cell)[0]


def train_snn(hparams: dict, train_ds: DrugResponseDataset,
              cell_fd: FeatureDataset, drug_fd: FeatureDataset,
              ckpt_dir: Path, seed: int) -> SimpleNeuralNetwork:
    """
    训练一个 SNN 模型（从头训练）。
    - 为什么要“从头训练”？为了避免上一轮的权重/状态泄漏到下一轮，确保每轮 AL 可比。
    - ckpt_dir：用于保存中间检查点（按需）。
    """
    np.random.seed(seed)
    m = SimpleNeuralNetwork()
    m.build_model(dict(hparams))  # 用拷贝防止被内部写回修改
    m.train(
        output=train_ds,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
        output_earlystopping=None,          # 如需 ES，可在 split 中提供 early_stopping
        model_checkpoint_dir=str(ckpt_dir),
    )
    return m


def predict_snn(model: SimpleNeuralNetwork, ds: DrugResponseDataset,
                cell_fd: FeatureDataset, drug_fd: FeatureDataset) -> np.ndarray:
    """
    用训练好的 SNN 在给定样本集上做预测，返回一维 numpy 数组（长度=样本数）。
    """
    return model.predict(
        cell_line_ids=ds.cell_line_ids,
        drug_ids=ds.drug_ids,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
    ).reshape(-1)


# ----------------------- HELPERS（新增：EPI 排序函数） -----------------------
def epi_rank_by_ensemble(hparams: dict,
                         base_train: DrugResponseDataset,
                         labeled_target: DrugResponseDataset,
                         pool_target: DrugResponseDataset,
                         cell_fd: FeatureDataset,
                         drug_fd: FeatureDataset,
                         ckpt_dir: Path,
                         ensemble: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    用“小型深度集成”估计 epistemic 不确定性：
      - 训练 E 个模型（不同随机种子），训练集 = base_train (其他细胞) + labeled_target (当前目标细胞已标注)。
      - 对 pool_target 预测，得到 E×|pool| 的预测矩阵。
      - 沿集成维度求方差，作为 epistemic 不确定性度量。
      - 返回“按方差降序”的 pool 内部索引（越靠前越不确定，优先选）。
    """
    ens_preds = []
    seeds = rng.integers(1, 10_000, size=ensemble)

    # 合并训练集（其他细胞的全部 + 目标细胞已标注）
    merged_train = base_train.copy()
    if len(labeled_target) > 0:
        merged_train.add_rows(labeled_target)

    for i, s in enumerate(seeds):
        m = train_snn(hparams, merged_train, cell_fd, drug_fd, ckpt_dir / f"ens_{i}", seed=int(s))
        p = predict_snn(m, pool_target, cell_fd, drug_fd)
        ens_preds.append(p)

    ens_preds = np.stack(ens_preds, axis=0)  # [E, |pool|]
    epi_var = ens_preds.var(axis=0)          # [|pool|]
    return np.argsort(-epi_var)              # 方差从大到小的排序索引（pool 内部位置）


# ----------------------- 主流程（单细胞的 PAL 循环） -----------------------
def personalized_al_for_cell(cell_id: str,
                             split: dict,
                             hparams: dict,
                             cell_fd_full: FeatureDataset,
                             drug_fd_full: FeatureDataset,
                             ckpt_base: Path,
                             seed: int) -> List[Dict]:
    """
    针对“一个目标细胞”执行完整的个性化主动学习循环，返回每一轮的评估记录列表。

    定义：
    - base_train：该折 train 中“其他细胞”的所有样本（提供总体先验）。
    - tgt_all：该折 train 中“目标细胞”的所有样本（作为 AL 的候选池）。
    - L_local：目标细胞“已标注”的药物对（本轮训练会使用）。
    - U_local：目标细胞“未标注”的药物对（本轮评估 & 采样从这里进行）。
    - 评估：每轮在 U_local 上评估（衡量“还没实验的药物”的个性化泛化性能）。
    """
    rng = np.random.default_rng(seed)

    # 1) 拆分出“其他细胞”的训练集（base_train）
    train_all = split["train"]
    base_mask = (train_all.cell_line_ids != cell_id)
    base_idx  = np.where(base_mask)[0]
    base_train = subset_dr(train_all, base_idx, name="BASE_TRAIN")

    # 2) 目标细胞的池（来自该折 train 中属于该 cell 的所有样本）
    tgt_idx = filter_by_cell(train_all, cell_id)
    if len(tgt_idx) == 0:
        # 某些折可能没有这个细胞的训练样本（极少见）；直接跳过
        return []

    tgt_all = subset_dr(train_all, tgt_idx, name=f"{cell_id}_ALL")
    pool_idx = np.arange(len(tgt_all))  # 目标细胞内部的“行索引”

    # 3) 初始化已标注 L 与未标注 U（都在目标细胞内部进行）
    init_k = min(INIT_SIZE, len(pool_idx))
    L_local = rng.choice(pool_idx, size=init_k, replace=False)
    U_local = np.setdiff1d(pool_idx, L_local, assume_unique=False)

    records = []

    # 4) 主动学习主循环：包含 r=0 的“冷启动性能”，以及 N_ROUNDS 轮迭代
    for r in range(N_ROUNDS + 1):
        # 4.1 构建当前“目标细胞已标注”子集 L_ds
        L_ds = subset_dr(tgt_all, L_local, name=f"{cell_id}_L{r}")

        # 4.2 合并训练集：base_train（其他细胞） + L_ds（目标细胞当前已标注）
        merged_train = base_train.copy()
        if len(L_ds) > 0:
            merged_train.add_rows(L_ds)

        # 4.3 训练当前轮的模型（从头训练）
        model_r = train_snn(
            hparams=hparams,
            train_ds=merged_train,
            cell_fd=cell_fd_full,
            drug_fd=drug_fd_full,
            ckpt_dir=ckpt_base / f"round_{r}",
            seed=seed + r,
        )

        # 4.4 评估：在“目标细胞的未标注 U 上”评估泛化
        U_ds = subset_dr(tgt_all, U_local, name=f"{cell_id}_U{r}")
        if len(U_ds) > 0:
            y_true = U_ds.response
            y_pred = predict_snn(model_r, U_ds, cell_fd_full, drug_fd_full)
            mets = metrics_all(y_true, y_pred)
        else:
            # 若 U 已空，指标记为 NaN（代表无未标注可评估）
            mets = {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Pearson": np.nan}

        # 4.5 记录本轮结果
        mets.update({
            "cell_id": cell_id,
            "round": r,
            "labeled_size": int(len(L_local)),
        })
        records.append(mets)

        # 4.6 是否终止（轮数到达 / 池已空）
        if r == N_ROUNDS or len(U_local) == 0:
            break

        # 4.7 选下一批（在 U 内选择），更新 L/U
        #     注意：Random 与 EPI-EnsVar 都只在目标细胞内部做，不影响其他细胞。
        if STRATEGY == "Random":
            k = min(QUERY_SIZE, len(U_local))
            new_local = rng.choice(U_local, size=k, replace=False)

        elif STRATEGY == "EPI-EnsVar":
            # 在 U 子集上根据 EPI 排序，选择方差最大的前 k 个
            U_ds_for_rank = subset_dr(tgt_all, U_local, name=f"{cell_id}_POOL")
            order_in_U = epi_rank_by_ensemble(
                hparams=hparams,
                base_train=base_train,       # 其他细胞
                labeled_target=L_ds,         # 目标细胞当前已标注
                pool_target=U_ds_for_rank,   # 目标细胞候选池（未标注）
                cell_fd=cell_fd_full,
                drug_fd=drug_fd_full,
                ckpt_dir=ckpt_base / f"epi_round_{r}",
                ensemble=EPI_ENSEMBLE,
                rng=rng,
            )
            top = order_in_U[:min(QUERY_SIZE, len(order_in_U))]
            new_local = U_local[top]

        else:
            raise ValueError(f"Unknown strategy: {STRATEGY}")

        # 加入新标注，收缩池
        L_local = np.union1d(L_local, new_local)
        U_local = np.setdiff1d(U_local, new_local, assume_unique=False)

    return records


# ----------------------------- MAIN -----------------------------
def main():
    """
    程序入口：
    1) 读取/检查路径 & 数据；
    2) LCO 交叉验证拆分；
    3) 对每个策略、每一折、每个目标细胞执行 PAL；
    4) 导出学习曲线（逐轮）与聚合报告。
    """
    # 目录准备
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    # ---- 加载特征（用 SNN 的内置 loader，自动处理 arcsinh+scaler 等）
    tmp = SimpleNeuralNetwork()
    hparams = load_hparams()
    tmp.build_model(hparams)

    cell_fd = tmp.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = tmp.load_drug_features(str(DATA_ROOT), DATASET_NAME)

    # 轻量清洗（可按需调整范围）
    ge_stats = clean_view(cell_fd, "gene_expression", clip_min=-50.0, clip_max=50.0)
    fp_stats = clean_view(drug_fd, "fingerprints",    clip_min=0.0,   clip_max=1.0)
    print("[CLEAN] gene_expression:", ge_stats)
    print("[CLEAN] fingerprints   :", fp_stats)

    # ---- 加载响应 & 对齐 ID（非常关键：避免特征与响应错位）
    resp = DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
        tissue_column=None,  # 若 CSV 含有 'tissue' 列，可填入列名以支持 LTO
    )
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    print(f"[INFO] rows after filtering = {len(resp)}")
    if len(resp) == 0:
        print("[ERROR] empty dataset after filtering."); sys.exit(1)

    # ---- LCO 交叉验证拆分（个性化最合适）
    splits = resp.split_dataset(
        n_cv_splits=N_SPLITS,
        mode=CV_MODE,
        split_validation=True,     # 可选：提供 validation/early_stopping（SNN 会用）
        split_early_stopping=True,
        validation_ratio=0.1,
        random_state=SEED,
    )

    all_records = []

    # ---- 外层循环：策略 → 折 → 目标细胞
    for STRATEGY in STRATEGIES:
        print(f"\n=== STRATEGY: {STRATEGY} ===")
        for fi, split in enumerate(splits):
            # 该折 test 中有哪些细胞 → 它们就是本折的“目标个体”
            test_cells = np.unique(split["test"].cell_line_ids)
            if TARGET_CELLS_LIMIT is not None:
                test_cells = test_cells[:TARGET_CELLS_LIMIT]
            print(f"  Fold {fi+1}/{len(splits)}: {len(test_cells)} target cells")

            for cid in test_cells:
                print(f"    - Target cell: {cid}")
                recs = personalized_al_for_cell(
                    cell_id=cid,
                    split=split,
                    hparams=hparams,
                    cell_fd_full=cell_fd.copy(),   # 传 copy，避免某些操作意外修改原对象
                    drug_fd_full=drug_fd.copy(),
                    ckpt_base=CKPT_DIR / "PAL" / STRATEGY / f"fold{fi}" / cid,
                    seed=SEED + fi,                # 让每折有不同随机性
                )
                for r in recs:
                    r.update({"strategy": STRATEGY, "fold": fi})
                all_records.extend(recs)

    # ---- 导出逐轮记录
    df = pd.DataFrame(all_records)
    df.to_csv(PAL_DIR / "pal_metrics.csv", index=False)

    # ---- 跨折/跨细胞聚合：得到平均学习曲线（更平滑、可写论文主图）
    summary = df.groupby(["strategy", "round"]).agg(
        labeled_size=("labeled_size", "mean"),
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        Pearson_mean=("Pearson", "mean"), Pearson_std=("Pearson", "std"),
    ).reset_index()
    summary.to_csv(PAL_DIR / "pal_summary.csv", index=False)

    # ---- Markdown 报告（可直接放到论文附录）
    with open(PAL_DIR / "pal_report.md", "w") as f:
        f.write(f"# Personalized Active Learning (PAL) – {DATASET_NAME}\n\n")
        f.write(f"- CV mode: **{CV_MODE}**, folds: **{N_SPLITS}**\n")
        f.write(f"- Strategies: {', '.join(STRATEGIES)}\n")
        f.write(f"- AL params: INIT={INIT_SIZE}, QUERY={QUERY_SIZE}, ROUNDS={N_ROUNDS}, ENSEMBLE={EPI_ENSEMBLE}\n\n")
        f.write("## Aggregated Learning Curves\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Sample Records (head)\n\n")
        f.write(df.head(30).to_markdown(index=False))
        f.write("\n")

    # ---- 运行配置归档（便于复现实验）
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
                "ensemble": EPI_ENSEMBLE,
                "strategies": STRATEGIES,
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

    print("\n PAL finished.")
    print("   - metrics:", PAL_DIR / "pal_metrics.csv")
    print("   - summary:", PAL_DIR / "pal_summary.csv")
    print("   - report :", PAL_DIR / "pal_report.md")


if __name__ == "__main__":
    main()
