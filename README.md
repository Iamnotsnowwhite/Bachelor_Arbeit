# Quantifying the Impact of Aleatoric and Epistemic Uncertainty on Drug Response Prediction in Active Learning Settings

## Summary

The inherent biological heterogeneity of cancer leads to substantial variability in individual drug response, rendering conventional global prediction models often inadequate for personalized medicine. This work addresses the critical need for precise individualized drug efficacy predictions by leveraging cancer cell line data from the GDSC2 dataset. Given the substantial costs associated with experimental drug response measurement, I employ active learning to strategically select the most informative data points for model improvement. I systematically compare two uncertainty-based sampling strategies: aleatoric uncertainty estimation using mean-variance networks and epistemic uncertainty quantification via Monte Carlo dropout. Both approaches are evaluated against random sampling baselines using neural networks, with the ultimate goal of identifying the most effective uncertainty quantification method for maximizing model performance while minimizing data acquisition costs.

<div align="center">
  <img width="866" height="562" alt="截屏2025-10-04 16 11 48" src="https://github.com/user-attachments/assets/1de7b1e8-270e-444f-874f-e80c260d51b4" /> </div>
  
## Research Workflow

### ✅ 1. Baseline Model Training (Completed)

**Goal:** Train a generalizable model on non-target cell lines; hold the target cell line out completely.

**Implementation**

- Split dataset:
  - test set - target cell line/ target cell lines
  - training set - Exclude target cell line/ target cell lines
- Model: Simple Neural Network + Dropout + dual-head (mean μ and log-variance logσ²).
- Loss: Gaussian NLL when aleatoric is enabled (fallback to MSE if disabled).
- Initialization: warm start from a previous baseline.
  
### ✅ 2. Uncertainty for the Target Cell Line (Completed)

**Goal:** For the target cell line, compute predictions and per-drug uncertainties to support active sampling.

### A. Epistemic Uncertainty (Model Uncertainty)
- **Technique:** Monte Carlo Dropout (T = 50)
- **Approach:** Run the model T times with dropout **on** (BatchNorm kept in eval) and collect the T predictions.
- **Output:**
  - **Mean prediction:** average of the T predictions.
  - **Epistemic variance:** variance across the T predictions.
  - **Interpretation:** Uncertainty from the **model / limited data**. It is **reducible** by adding labels.

### B. Aleatoric Uncertainty (Data Uncertainty)
- **Technique:** Mean–Variance (dual-head) network trained with **Gaussian NLL**.
- **Approach:** During inference I also run MC; for each pass I read the predicted log-variance, convert it to variance, then **average** these variances over T passes.
- **Output:**
  
<!-- - **Aleatoric variance:** average of the per-pass variances.
%%- **Aleatoric std:** square root of the aleatoric variance.%%
- **Interpretation:** Uncertainty from **measurement noise**. It is **irreducible** and sets a floor on interval width.
-->

### Implementation Notes: 
#### CSV columns(Baseline): cell_line,drug_id,y_true,y_pred,sigma_epi,sigma_ale,sigma_tot,mc_T
---

### Result for sample cell line: `TARGET_CELL_LINE = "UACC-257"`
- Source split: **Epistemic ≈ 55%**, **Aleatoric ≈ 45%**  
- Epistemic shows higher spread → strong signal for **active sampling**

<img width="1800" height="1400" alt="图片" src="https://github.com/user-attachments/assets/a667ac26-5c19-42ad-ba46-e8fe99226616" />

### ✅ 3. Active Learning Drug Selection

**Objective:** Select the most informative drug-cell line combinations for experimental validation based on uncertainty estimates.

**Comparison Strategies:**
- Uncertainty-guided selection (epistemic vs. aleatoric)
- Random sampling baseline
- Fixed budget of N drugs per active learning cycle

### ✅ 4. Simulated Experimental Validation

**Objective:** Incorporate selected drug response measurements into training data through simulated "experimental" validation.

**Process:**
- Reveal ground truth labels for selected drug combinations
- Augment training dataset with new target cell line measurements
- Maintain rigorous separation between training and evaluation sets

### 5. Model Fine-tuning and Evaluation

**Objective:** Assess performance improvement through personalized model adaptation.

**Evaluation Metrics:**

#### 1. Prediction Accuracy
（Measure the model's predictive performance on target cell line drugs.）

- **Final Performance Metrics**:
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Pearson Correlation Coefficient

- **Improvement Analysis**:
  - Percentage improvement over baseline for each metric
  - Statistical significance testing (paired t-tests)

#### 2. Generalization Capability
（Assess the model's ability to generalize to unseen data.）

- **New Drug Prediction**:
  - Accuracy on drugs not used during fine-tuning

- **Uncertainty Calibration**:
  - Correlation between predicted confidence and actual errors
  - Reliability of uncertainty estimates

- **Performance Stability**:
  - Consistency across different drug subsets
  - Robustness to data variations

#### 3. Resource Efficiency
（Evaluate the efficiency of data utilization and cost-effectiveness.）

- **Learning Speed**:
  - Number of data points required to reach target performance
  - Learning curves analysis

- **Cost-Effectiveness**:
  - Performance gain per additional data point (ΔR²/sample)
  - Optimal stopping point identification

#### 4. Uncertainty Quality
（Validate the reliability of uncertainty estimates.）

- **Method Comparison**:
  - Epistemic vs. aleatoric uncertainty reliability
  - Ranking correlation analysis
    
#### 5. Statistical Robustness
（Ensure results are statistically sound and reproducible.）

- **Multiple Runs**:
  - Average results over 5-10 independent runs
  - Variance reduction through repeated experiments
