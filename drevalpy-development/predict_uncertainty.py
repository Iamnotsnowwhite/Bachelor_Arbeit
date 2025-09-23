"""
Target Cell Prediction + MC Dropout (Epistemic) + Mean-Variance (Aleatoric)
- Uses SimpleNeuralNetwork.get_concatenated_features(...) to build X
- Calls FeedForwardNetwork.predict_mc_meanvar(T) if available; else falls back to predict_mc(T)
- Computes var_epi (MC over means), var_ale (MC-avg exp(log_var)), var_total, std_total, and 95% CI
- Reports Spearman correlation between |error| and each uncertainty (epi / ale / total)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

EVAL_MODE = "target"  # 可选 "target" 或 "val"

# --------- Paths and Configuration ----------
BASE_DIR      = Path(__file__).resolve().parent
DATASET_NAME  = "GDSC2"
DATA_ROOT     = BASE_DIR / "data"
RESPONSE_FILE = DATA_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"

# 根目录（results/<dataset>）
RESULTS_DIR   = BASE_DIR / "results" / DATASET_NAME

# 不确定性输出目录
UNCERT_DIR    = RESULTS_DIR / "uncertainty"
OUT_CSV       = UNCERT_DIR / "mc_dropout+mean_variance_target.csv"

# 基线模型目录（优先用带 aleatoric 的；否则回退到旧的 MSE 基线）
ALE_DIR       = RESULTS_DIR / "baseline_models_ale" / "exclude_target"
OLD_DIR       = RESULTS_DIR / "baseline_models"     / "exclude_target"
BASELINE_DIR = ALE_DIR if (ALE_DIR / "model.pt").exists() else OLD_DIR
print("[LOAD] Using baseline dir:", BASELINE_DIR)

# Output
OUT_DIR       = BASE_DIR / "results" / DATASET_NAME / "uncertainty"
OUT_CSV       = OUT_DIR / "SW1783_mc_dropout+mean_variance_target.csv"

# Target cell line
TARGET_CELL_LINE = "SW1783"
#"UACC-257"

# MC Dropout Configuration
T               = 50      # Number of MC samples
SEED            = 42
FORCE_DROPOUT_P = None    # Force set all nn.Dropout p to 0.5; set to None to keep original
EPS_VAR         = 1e-12   # Numerical guard for variance
Z_95            = 1.96    # 95% CI multiplier

def load_features(sn: SimpleNeuralNetwork) -> tuple[FeatureDataset, FeatureDataset]:
    cell_fd = sn.load_cell_line_features(str(DATA_ROOT), DATASET_NAME)
    drug_fd = sn.load_drug_features(str(DATA_ROOT), DATASET_NAME)
    return cell_fd, drug_fd

def load_responses() -> DrugResponseDataset:
    return DrugResponseDataset.from_csv(
        input_file=str(RESPONSE_FILE),
        dataset_name=DATASET_NAME,
        measure="LN_IC50_curvecurator",
    )

def subset_by_idx(ds: DrugResponseDataset, idx: np.ndarray, name: str) -> DrugResponseDataset:
    return DrugResponseDataset(
        response=ds.response[idx],
        cell_line_ids=ds.cell_line_ids[idx],
        drug_ids=ds.drug_ids[idx],
        tissues=ds.tissue[idx] if ds.tissue is not None else None,
        predictions=None,
        dataset_name=name,
    )

def get_target_ds(resp: DrugResponseDataset, cell_fd: FeatureDataset, drug_fd: FeatureDataset) -> DrugResponseDataset:
    resp = resp.copy()
    resp.reduce_to(cell_line_ids=cell_fd.identifiers, drug_ids=drug_fd.identifiers)
    resp.remove_nan_responses()
    idx = np.where(resp.cell_line_ids == TARGET_CELL_LINE)[0]
    if len(idx) == 0:
        raise RuntimeError(f"No rows for target cell {TARGET_CELL_LINE}.")
    return subset_by_idx(resp, idx, f"holdout_{TARGET_CELL_LINE}")

def set_dropout_p_only(module: torch.nn.Module, p: float | None):
    if p is None:
        return
    n = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(p); n += 1
    print(f"[INFO] Set Dropout.p = {p} for {n} Dropout layers")

def describe_dropout(module: torch.nn.Module):
    ps = [m.p for m in module.modules() if isinstance(m, torch.nn.Dropout)]
    if ps:
        ps = np.array(ps, float)
        print(f"[CHECK] #Dropout layers = {len(ps)}, p(min/mean/max) = {ps.min():.3f}/{ps.mean():.3f}/{ps.max():.3f}")
    else:
        print("[WARN] No Dropout layers found! MC Dropout will not produce randomness.")

@torch.no_grad()
def build_X(
    model: SimpleNeuralNetwork,
    ds: DrugResponseDataset,
    cell_fd: FeatureDataset,
    drug_fd: FeatureDataset,
) -> np.ndarray:
    """Use project concatenation to ensure training/inference consistency."""
    x = model.get_concatenated_features(
        cell_line_view="gene_expression",
        drug_view="fingerprints",
        cell_line_ids_output=ds.cell_line_ids,
        drug_ids_output=ds.drug_ids,
        cell_line_input=cell_fd,
        drug_input=drug_fd,
    )  # -> np.ndarray [N, D]
    return x

@torch.no_grad()
def mc_dropout_mean_and_vars(
    model: SimpleNeuralNetwork,
    x_np: np.ndarray,
    T: int = 50,
    seed: int = 42,
    eps: float = 1e-12,
):
    """
    Preferred: FeedForwardNetwork.predict_mc_meanvar -> returns (MU[T,N], LOGVAR[T,N])
    Fallback : predict_mc -> returns MU[T,N]; LOGVAR=None (aleatoric=0)
    Returns:
        mu         (N,)
        var_epi    (N,)
        var_ale    (N,)
        var_total  (N,)
        std_total  (N,)
    """
    np.random.seed(seed)
    torch.manual_seed(int(seed))

    has_meanvar = hasattr(model.model, "predict_mc_meanvar")
    if has_meanvar:
        MU, LOGVAR = model.model.predict_mc_meanvar(x_np, T=T, keep_bn_eval=True)
    else:
        MU = model.model.predict_mc(x_np, T=T, keep_bn_eval=True)
        LOGVAR = None

    mu = MU.mean(axis=0)                    # (N,)
    var_epi = MU.var(axis=0, ddof=0)        # (N,)

    if LOGVAR is not None:
        # clip for numerical stability, then average σ^2 over MC
        var_ale = np.exp(np.clip(LOGVAR, -10.0, 10.0)).mean(axis=0)
    else:
        var_ale = np.zeros_like(var_epi)

    var_total = np.maximum(var_epi + var_ale, 0.0) + eps
    std_total = np.sqrt(var_total)
    return mu, var_epi, var_ale, var_total, std_total

def main():
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load baseline model (restores hyperparameters/Scaler/weights)
    model = SimpleNeuralNetwork.load(str(BASELINE_DIR))

    # 1.1 Optional: Amplify Dropout probability (for clearer diagnostics)
    set_dropout_p_only(model.model, FORCE_DROPOUT_P)
    describe_dropout(model.model)

    # 2) Load features and target data
    cell_fd, drug_fd = load_features(model)
    resp = load_responses()
    target_ds = get_target_ds(resp, cell_fd, drug_fd)
    print(f"[INFO] Target cell {TARGET_CELL_LINE}: {len(target_ds)} drug pairs")

    # 3) Build X and run MC Dropout + Mean-Variance
    X_target = build_X(model, target_ds, cell_fd, drug_fd)
    y_mean, var_epi, var_ale, var_total, std_total = mc_dropout_mean_and_vars(
        model=model,
        x_np=X_target,
        T=T,
        seed=SEED,
        eps=EPS_VAR,
    )

    # 4) Diagnostics: Spearman correlation between |error| and uncertainties
    err_abs = np.abs(target_ds.response - y_mean)

    def _safe_spear(a, b, name):
        try:
            r, p = spearmanr(a, b)
            print(f"[DIAG] Spearman(|error|, {name}) = {r:.3f}  (p={p:.2e})")
        except Exception as e:
            print(f"[DIAG] Spearman({name}) failed: {e}")

    _safe_spear(err_abs, np.sqrt(var_epi + EPS_VAR), "epi_std")
    if np.any(var_ale > 0):
        _safe_spear(err_abs, np.sqrt(var_ale + EPS_VAR), "ale_std")
    _safe_spear(err_abs, std_total, "total_std")

    # 5) Confidence intervals (approx. Gaussian)
    ci_low  = y_mean - Z_95 * std_total
    ci_high = y_mean + Z_95 * std_total

    # 6) Export
    def _z(v):
        v = np.asarray(v)
        return (v - v.mean()) / (v.std() + 1e-12)

    df = pd.DataFrame({
        "cell_line": target_ds.cell_line_ids,
        "drug_id":   target_ds.drug_ids,
        "y_true":    target_ds.response,
        "y_pred":    y_mean,
        "var_epi":   var_epi,
        "var_ale":   var_ale,
        "var_total": var_total,
        "std_total": std_total,
        "ci95_low":  ci_low,
        "ci95_high": ci_high,
        "uncert_z":  _z(std_total),
        "abs_error": err_abs,
        "T":         T,
        "seed":      SEED,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved uncertainties to: {OUT_CSV}")
    print(df.head(10))

if __name__ == "__main__":
    main()
