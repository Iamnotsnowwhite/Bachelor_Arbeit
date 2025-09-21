# Quantifying the Impact of Aleatoric and Epistemic Uncertainty on Drug Response Prediction in Active Learning Settings

## Abstract

The inherent biological heterogeneity of cancer leads to substantial variability in individual drug response, rendering conventional global prediction models often inadequate for personalized medicine. This work addresses the critical need for precise individualized drug efficacy predictions by leveraging cancer cell line data from the GDSC2 dataset. Given the substantial costs associated with experimental drug response measurement, we employ active learning to strategically select the most informative data points for model improvement. We systematically compare two uncertainty-based sampling strategies: aleatoric uncertainty estimation using mean-variance networks and epistemic uncertainty quantification via Monte Carlo dropout. Both approaches are evaluated against random sampling baselines using neural networks, with the ultimate goal of identifying the most effective uncertainty quantification method for maximizing model performance while minimizing data acquisition costs.

## Research Workflow

### ✅ 1. Baseline Model Training - Completed

**Objective:** Establish a reference model trained exclusively on non-target cell lines to simulate initial generalization capability before personalized adaptation.

**Implementation:**
- Strict exclusion of target cell line data during training
- 80/20 split of non-target cell lines for training/validation
- Model architecture: Simple Neural Network with dropout regularization
- Evaluation on both validation set and completely held-out target cell line

**Key Insight:** The performance gap between validation set and target cell line demonstrates the necessity for personalized learning approaches.

### 2. Uncertainty Quantification for Target Cell Line - mayebe already completed 

**Objective:** Compute predictive uncertainties for all drug combinations on the target cell line to enable intelligent data selection.

**Methods Under Investigation:**

#### A. Epistemic Uncertainty (Model Uncertainty)
- **Technique:** Monte Carlo Dropout
- **Approach:** Multiple forward passes with activated dropout during inference
- **Output:** Standard deviation of predictions across sampling iterations
- **Interpretation:** Measures model confidence based on parameter uncertainty

#### B. Aleatoric Uncertainty (Data Uncertainty)  
- **Technique:** Mean-Variance Estimation
- **Approach:** Single forward pass with dual-output architecture (mean + variance)
- **Output:** Learned data-dependent uncertainty estimates
- **Interpretation:** Captures inherent noise in the observations

### result for sample cell line: TARGET_CELL_LINE = "UACC-257"
<img width="1825" height="1484" alt="图片" src="https://github.com/user-attachments/assets/a667ac26-5c19-42ad-ba46-e8fe99226616" />

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
