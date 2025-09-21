# Quantifying the Impact of Aleatoric and Epistemic Uncertainty on Drug Response Prediction in Active Learning Settings

## Abstract

The inherent biological heterogeneity of cancer leads to substantial variability in individual drug response, rendering conventional global prediction models often inadequate for personalized medicine. This work addresses the critical need for precise individualized drug efficacy predictions by leveraging cancer cell line data from the GDSC2 dataset. Given the substantial costs associated with experimental drug response measurement, we employ active learning to strategically select the most informative data points for model improvement. I systematically compare two uncertainty-based sampling strategies: aleatoric uncertainty estimation using mean-variance networks and epistemic uncertainty quantification via Monte Carlo dropout. Both approaches are evaluated against random sampling baselines using neural networks, with the ultimate goal of identifying the most effective uncertainty quantification method for maximizing model performance while minimizing data acquisition costs.

## Research Workflow

### ✅ 1. Baseline Model Training (Completed)

**Goal:** Train a generalizable model on non-target cell lines; hold the target cell line out completely.

**Implementation**

- Split (LCO): Exclude target cell line (UACC-257) from training/validation. On the non-target data, use an 80/20 train/validation split.
- Model: Simple Neural Network + Dropout + dual-head outputs (mean μ and log-variance logσ²).
- Loss: Gaussian NLL when aleatoric is enabled (fallback to MSE if disabled).
- Initialization: Optional warm start from a previous MSE baseline (only for initialization; LCO split is unchanged).
- Evaluation: Report metrics on the non-target validation set and on the fully held-out target cell line.
- Observation: Under LCO, target-cell performance is close to the non-target validation performance (e.g., R²≈0.55–0.58), indicating reasonable generalization while still leaving room for personalized adaptation via active learning + fine-tuning.

### 2. Uncertainty for the Target Cell Line (Completed)

**Objective:** Goal: For the target cell line, compute predictions and per-drug uncertainties to support active sampling.

#### A. Epistemic Uncertainty (Model Uncertainty)
- **Technique:** Monte Carlo Dropout (T = 50)
- **Approach:** Perform **T stochastic forward passes** with **Dropout active** (BatchNorm kept in eval). Collect per-pass predictions.
- **Output:**
  - **Mean prediction:** \( \mu=\mathbb{E}_t[\mu_t] \)
  - **Epistemic variance:** \( \mathrm{Var}_{\text{epi}}=\mathrm{Var}_t[\mu_t] \) → **epistemic std** \(=\sqrt{\mathrm{Var}_{\text{epi}}}\)
- **Interpretation:** Captures uncertainty due to **model parameters / limited data**. **Reducible** with more labels → ideal for **active learning** ranking.

#### B. Aleatoric Uncertainty (Data Uncertainty)
- **Technique:** **Mean–Variance (dual-head)** network trained with **Gaussian NLL** (outputs \( \mu, \log\sigma^2 \))
- **Approach:** During inference we also run MC; for each pass get \( \log\sigma_t^2 \) and compute  
  \( \mathrm{Var}_{\text{ale}}=\mathbb{E}_t\!\left[\exp(\log\sigma_t^2)\right] \)
- **Output:** **Aleatoric std** \(=\sqrt{\mathrm{Var}_{\text{ale}}}\) — per-sample noise estimate
- **Interpretation:** Captures **measurement/biological noise**. **Irreducible**; sets a floor on interval width.

#### C. Combined Uncertainty & Intervals
- **Total variance:** \( \mathrm{Var}_{\text{tot}}=\mathrm{Var}_{\text{epi}}+\mathrm{Var}_{\text{ale}} \)  
- **Total std:** \( \mathrm{Std}_{\text{tot}}=\sqrt{\mathrm{Var}_{\text{tot}}} \)  
- **95% CI:** \( \mu \pm 1.96 \cdot \mathrm{Std}_{\text{tot}} \)  
- **(Optional) Calibration:** On the **non-target validation set**, estimate a scalar \(c\) to match two-sided 95% coverage, then apply  
  \( \mathrm{Std}_{\text{tot}} \leftarrow c\cdot \mathrm{Std}_{\text{tot}} \) to target-cell intervals.

**Implementation Notes:** Dropout \(p=0.3\) for training & final inference (no forced amplification); seed = 42; T = 50.  
**CSV columns:** `cell_line, drug_id, y_true, y_pred, var_epi, var_ale, var_total, std_total, ci95_low, ci95_high, abs_error, T, seed`.

---

### Result for sample cell line: `TARGET_CELL_LINE = "UACC-257"`
- 95% CI coverage ≈ **94%**（with final inference at training dropout, no forced amplification）  
- Source split: **Epistemic ≈ 55%**, **Aleatoric ≈ 45%**  
- Epistemic shows higher dispersion → strong signal for **active sampling**

<img width="1800" height="1400" alt="图片" src="https://github.com/user-attachments/assets/a667ac26-5c19-42ad-ba46-e8fe99226616" />

### 3. Active Learning Drug Selection

**Objective:** Select the most informative drug-cell line combinations for experimental validation based on uncertainty estimates.

**Comparison Strategies:**
- Uncertainty-guided selection (epistemic vs. aleatoric)
- Random sampling baseline
- Fixed budget of N drugs per active learning cycle

### 4. Simulated Experimental Validation

**Objective:** Incorporate selected drug response measurements into training data through simulated "experimental" validation.

**Process:**
- Reveal ground truth labels for selected drug combinations
- Augment training dataset with new target cell line measurements
- Maintain rigorous separation between training and evaluation sets

### 5. Model Fine-tuning and Evaluation

**Objective:** Assess performance improvement through personalized model adaptation.

**Evaluation Metrics:**
- Prediction accuracy on target cell line drugs
- Generalization capability to unseen drug combinations
- Comparison against baseline model performance
- Resource efficiency analysis (performance gain per additional data point)
