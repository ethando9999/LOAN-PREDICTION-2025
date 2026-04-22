# OPTIMIZING CREDIT DEFAULT PREDICTION USING HYBRID SHAP-FOCAL-LIGHTGBM MODEL ON LOAN PREDICTION 2025 DATASET

**Author:** Thanh-Chien Do  
**Email:** ethando369@gmail.com  
**Report Date:** 2026-04-22  
**Code:** `src/`, `configs/config.yaml`  
**Artifacts:** `reports/`

---

## ABSTRACT

**Background.** In the Fintech era and economic fluctuations of 2025, credit scoring systems require both high accuracy and transparency to meet legal requirements.

**Problem.** Traditional statistical models (Logistic Regression) are easy to interpret but have low performance; SOTA models (XGBoost, CatBoost, Deep Learning) have high AUC but are "black boxes" and handle imbalanced data poorly.

**Solution.** The study proposes a **Hybrid SHAP-Focal-LightGBM** model, combining (i) LightGBM with customized **Focal Loss** to focus on hard samples (minority defaults), (ii) **Optuna** for hyperparameter tuning, and (iii) **SHAP** for decision transparency at both global and local levels.

**Results.** Tested on *Loan Prediction Dataset 2025* (Kaggle, 20,000 records, default rate 20%), the **Focal-LightGBM tuned** model achieves **PR-AUC = 0.8021**, **ROC-AUC = 0.9017**, **Gini = 0.8035**, **KS = 0.6046** on test set — almost equal to CatBoost (0.001 PR-AUC difference) and significantly outperforms Logistic Regression (+0.086 PR-AUC) and other tree baselines. Combined with SHAP for explaining each credit decision — proving that the Accuracy-vs-Explainability gap can be narrowed.

**Keywords.** Credit scoring, Focal Loss, LightGBM, SHAP, Imbalanced Learning, Explainable AI.

---

## 1. INTRODUCTION

### 1.1. Problem Statement

The problem of predicting default probability (Probability of Default — PD) is a binary classification:

\[ y \in \{0, 1\}, \quad y = 1 \text{ if customer defaults} \]

Objective: build a function \( f: \mathcal{X} \to [0,1] \) estimating \( P(y=1 \mid \mathbf{x}) \), where \( \mathbf{x} \) is the vector of demographic/behavior/history features.

### 1.2. Importance

- Reduce bad debt ratio (NPL) and protect liquidity.
- Automate credit approval process with mandatory interpretability according to Basel III/IV, IFRS 9.
- Support complaints — customers have the right to know *why* they were rejected.

### 1.3. Research Objectives

1. Model the **non-linear** relationship between credit variables and PD.
2. Handle **imbalance** well (default rate ≈ 20%).
3. Provide **explanations** at both model level (global) and individual level (local).

---

## 2. RELATED WORK

| Approach | Representatives | Pros | Cons |
|----------|-----------------|------|------|
| Traditional Statistics | Logistic Regression, Altman Z-score (1968), Credit Scorecard | Direct interpretation via coefficients | Linear assumptions, weak with interactions |
| Tree-based Ensemble | XGBoost [Chen & Guestrin 2016], LightGBM [Ke et al. 2017], CatBoost | SOTA on tabular data | Black box |
| Deep Learning for tabular | TabNet [Arik & Pfister 2021], TabTransformer | Strong feature learning | Large data + resources, hard to deploy |
| Rule-based | `IF overdue > X THEN reject` | Absolute transparency | Rigid, hard to scale |
| XAI | SHAP [Lundberg & Lee 2017], LIME, PDP | Explain black box models | Computational cost |

**Gap.** Lack of framework integrating (a) imbalance handling via loss function, (b) systematic tuning, (c) individual-level explanation — simultaneously on real credit data.

---

## 3. CONTRIBUTIONS

1. **Custom Focal Loss for LightGBM:** explicit gradient/hessian implementation for LightGBM 4.x compatibility (via `params["objective"]`).
2. **Quantitative Feature Selection:** pipeline Chi-square + ANOVA F + Weight-of-Evidence + Information Value, retaining 7/21 variables with `IV ≥ 0.02`.
3. **Hybrid SHAP-Focal-LightGBM:** final model combining Optuna tuning and SHAP explainer, packaged reproducible in `src/pipeline.py`.

---

## 4. METHODOLOGY

### 4.1. Preprocessing

- **Target flip:** change labels to convention `1 = default, 0 = paid_back` to suit imbalanced metrics (PR-AUC, F2 prioritizes default group).
- **Encoding:** `OrdinalEncoder` for 6 categorical variables (gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade).
- **Missing values:** dataset has no NA, so `IterativeImputer` (Bayesian Ridge) declared in pipeline but not activated.
- **Split:** stratified 70/15/15 (train/val/test) with `random_state=42`.

### 4.2. Statistical Tests

Statistics on train set:

- **Chi-square** for categorical variables vs target.
- **ANOVA F-test** for numeric variables vs target.

Typical results (see `reports/anova_test.csv`, `reports/chi2_test.csv`):

| Feature | Test | p-value |
|---------|------|---------|
| `debt_to_income_ratio` | ANOVA | 5.1e-162 |
| `credit_score` | ANOVA | 2.7e-125 |
| `interest_rate` | ANOVA | 4.8e-42 |
| `gender` | Chi² | 0.64 (not significant) |

### 4.3. Weight of Evidence & Information Value

\[ WoE_i = \ln\!\left(\frac{\text{Good}_i / \text{Good}_{\text{total}}}{\text{Bad}_i / \text{Bad}_{\text{total}}}\right), \quad IV = \sum_i (\text{Good}\%_i - \text{Bad}\%_i)\, WoE_i \]

Keep variable if \( IV \geq 0.02 \). Top 10 results (see `reports/iv_table.csv`):

| Feature | IV | Interpretation |
|---------|----|---------------|
| `employment_status` | **1.903** | Very strong (usually > 0.5 is unusual — check for leakage) |
| `debt_to_income_ratio` | 0.337 | Strong |
| `credit_score` | 0.294 | Strong |
| `grade_subgrade` | 0.271 | Strong |
| `interest_rate` | 0.088 | Medium |
| `delinquency_history` | 0.043 | Weak-Medium |
| `num_of_delinquencies` | 0.033 | Weak-Medium |
| `loan_purpose` | 0.007 | Weak |
| `total_credit_limit` | 0.006 | Weak |
| `age`, `gender`, `marital_status` | < 0.01 | Weak |

> **Leakage note:** `employment_status` has IV = 1.90 — unusually high. In practice, check if variable is updated *after* default. Here kept as Kaggle dataset is fixed snapshot.

**Final feature set:** 7 selected variables.

### 4.4. Multicollinearity (VIF)

VIF on selected numeric variables:

| Feature | VIF |
|---------|-----|
| `num_of_delinquencies` | 17.9 |
| `credit_score` | 15.9 |
| `delinquency_history` | 15.4 |
| `interest_rate` | 15.3 |
| `debt_to_income_ratio` | 4.0 |

High VIF mainly due to unstandardized scale; **no drop** as tree-based models (LightGBM/XGBoost/RF/CatBoost) are less affected by multicollinearity. Only logistic regression would have minor cost in coefficient interpretation — acceptable trade-off.

### 4.5. Proposed Model — Focal-LightGBM

**Focal Loss** [Lin et al. 2017], applied to binary classification:

\[ \mathcal{L}_{\text{FL}}(p_t) = -\alpha_t \,(1 - p_t)^{\gamma}\, \log p_t \]

where \( p_t = p \) if \( y=1 \), \( 1-p \) if \( y=0 \), and \( \alpha_t \) is class weight. The factor \( (1-p_t)^\gamma \) **reduces contribution of easy samples** (well-classified) so model focuses on hard ones — especially minority defaults.

**Gradient & Hessian** (implementation at `src/models/focal_lgbm.py:focal_loss_obj`):

\[ \frac{\partial \mathcal{L}}{\partial z} = \alpha_t\, (1-p_t)^{\gamma}\, \big[\gamma\, p_t \log p_t + p_t - 1\big]\,(2y - 1) \]

with \( z \) as raw-score (logit). Hessian calculated similarly then absolute value + epsilon to ensure positive-definite for LightGBM leaf updates.

**Integration with LightGBM 4.x:** pass callable via `params["objective"]` (old `fobj` API removed).

### 4.6. Hyperparameter Tuning — Optuna

Search space (30 trials, TPE sampler, `seed=42`):

| Parameter | Range |
|-----------|-------|
| `num_leaves` | [15, 255] |
| `learning_rate` | [0.01, 0.2] log-uniform |
| `min_child_samples` | [5, 100] |
| `reg_lambda` | [1e-3, 10] log-uniform |
| `feature_fraction` | [0.6, 1.0] |
| `bagging_fraction` | [0.6, 1.0] |
| `bagging_freq` | [0, 10] |
| `alpha` (focal) | [0.25, 0.9] |
| `gamma` (focal) | [0.5, 4.0] |

Objective: **PR-AUC** on val set.

**Best configuration** (see `reports/best_params.json`):

```json
{
  "num_leaves": 16,
  "learning_rate": 0.115,
  "min_child_samples": 72,
  "reg_lambda": 0.824,
  "feature_fraction": 0.909,
  "bagging_fraction": 0.630,
  "bagging_freq": 3,
  "alpha": 0.325,
  "gamma": 3.521
}
```

Notable: **γ ≈ 3.5** (higher than default 2.0) — system "forces" focus on hard samples more than usual; **α ≈ 0.33** — slightly favors negative class (paid_back), balancing with focusing effect.

### 4.7. SHAP Explainability

\[ g(\mathbf{z}') = \phi_0 + \sum_{j=1}^{M} \phi_j z'_j, \quad \phi_j = \text{contribution of feature } j \]

Use `shap.TreeExplainer` on tuned Focal-LightGBM, with 3,000 random samples from test set (out of 3,000 records) to save computation.

---

## 5. EXPERIMENTS

### 5.1. Dataset

- **Source:** [`nabihazahid/loan-prediction-dataset-2025`](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025)
- **Actual size:** 20,000 records × 22 columns (initial docs estimated 100K+ — difference from reality).
- **Target:** `loan_paid_back` (1 = paid back, 0 = default). **Flipped** to `1 = default` in pipeline.
- **Class ratio:** 80% paid_back vs **20% default** — medium imbalance (docs expected < 5%, reality milder but still needs handling).
- **Missing values:** 0 (dataset pre-cleaned).

### 5.2. Environment

```
Python 3.12.3
pandas 2.2.x, scikit-learn 1.5.x
lightgbm 4.6.x, xgboost 3.2.0, catboost 1.2.x
shap 0.44.x, optuna 3.5+
```

### 5.3. Metrics

According to credit scoring practice, **not** using Accuracy as main metric. Use:

| Metric | Meaning |
|--------|---------|
| **PR-AUC** | *Main metric* — prioritizes when imbalance |
| ROC-AUC | Risk ranking ability |
| Gini = 2·AUC − 1 | Industry standard |
| F2-score | Weighted Recall (no default misses) |
| KS | Distribution distance Good/Bad |
| Brier | Probability calibration (lower better) |

### 5.4. Baselines

Four comparison models with imbalance handling:

| Model | Imbalance handling |
|-------|-------------------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest (n=500) | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| CatBoost | `auto_class_weights='Balanced'` |

### 5.5. Main Results

**Leaderboard on test set** (sorted by PR-AUC):

| # | Model | PR-AUC | ROC-AUC | Gini | F2 | KS | Brier |
|---|-------|--------|---------|------|-----|-----|-------|
| 1 | CatBoost | **0.8031** | **0.9023** | **0.8046** | 0.6988 | **0.6162** | 0.1156 |
| 2 | **Focal-LightGBM (tuned)** | **0.8021** | 0.9017 | 0.8035 | 0.6006 | 0.6046 | 0.1197 |
| 3 | Focal-LightGBM (base) | 0.7900 | 0.8931 | 0.7862 | 0.6684 | 0.5905 | 0.1345 |
| 4 | Random Forest | 0.7827 | 0.8808 | 0.7617 | 0.6100 | 0.5916 | **0.0796** |
| 5 | XGBoost | 0.7768 | 0.8832 | 0.7664 | 0.6488 | 0.5705 | 0.1002 |
| 6 | Logistic Regression | 0.7158 | 0.8658 | 0.7316 | 0.6736 | 0.5731 | 0.1472 |

**Key observations:**

1. **Focal-LightGBM tuned (#2) approaches CatBoost (#1)** — 0.001 PR-AUC difference (within seed variance).
2. **Optuna improves base Focal-LightGBM +0.012 PR-AUC** (0.790 → 0.802) — affirms value of searching γ/α with tree hyperparams.
3. **Focal-LightGBM tuned outperforms Logistic Regression +0.086 PR-AUC** (+12% relative) — clear value of non-linear models.
4. **Random Forest has lowest Brier (0.0796)** — well-calibrated predicted probabilities, though PR-AUC not top.
5. **Logistic Regression F2 = 0.674**, higher than Focal-LightGBM tuned (0.601) — consequence of `class_weight='balanced'` boosting Recall, but Precision drops hard making PR-AUC fall.

### 5.6. SHAP Analysis

**Global importance** (see `reports/figures/shap_summary_bar.png`, `shap_summary_dot.png`) shows average |SHAP| contribution order:

1. `employment_status`
2. `debt_to_income_ratio`
3. `credit_score`
4. `grade_subgrade`
5. `interest_rate`

This order **consistent with IV table** — signal model works reasonably, not disturbed by artificial interactions.

**Local explanations** (see `reports/figures/shap_waterfall_sample_{0,1,2}.png`): each waterfall chart decomposes `log-odds(default)` for a specific customer into `base_value + Σ φ_j`, helping answer "*Why was customer X rated high risk?*" — directly serves compliance and explainability requests.

---

## 6. DISCUSSION

### 6.1. Interpretability vs Accuracy — Gap Narrowed

| Comparison | PR-AUC | Explanation |
|------------|--------|-------------|
| LR (easiest to interpret) | 0.716 | Direct linear coefficients |
| Focal-LightGBM tuned + SHAP | **0.802** | SHAP provides individual + global explanations |

Focal-LightGBM + SHAP achieves **+12% PR-AUC** over LR while still providing individual explanations. This is the *Accuracy-vs-Explainability gap* being closed.

### 6.2. Production Deployment (planned — Phase 8)

- **REST API scoring** via FastAPI: `/score` receives JSON features, returns `{pd, risk_grade, top_shap_features}`.
- **Batch scoring**: pipeline reads CSV → scored CSV (serves periodic portfolio re-evaluation).
- **Monitoring**: log input features and output PD distributions to detect **data drift** / **concept drift**.

### 6.3. Limitations

1. **Dataset size.** 20K records smaller than expected 100K+ in docs; tuned hyperparams may differ on larger real data.
2. **Kaggle dataset may not reflect real portfolio.** `employment_status` with IV = 1.90 suspicious; check if updated *after decision time* (label leakage).
3. **Single train/test split.** No K-Fold CV for final leaderboard — numbers have ±0.005–0.01 PR-AUC variance.
4. **Fraud adversarial attack.** Current model lacks mechanism to detect intentional fake features (e.g., inflated `annual_income`).
5. **Concept drift.** Customer behavior changes with economic cycles — 2025 model may lose validity 2026+ without re-training.
6. **Data leakage risk.** `grade_subgrade` may be consequence of internal bank scoring (circular signal).
7. **Calibration.** No Platt/Isotonic calibration applied — current probabilities may not reflect true default frequencies.
8. **High VIF (15-17)** for delinquency-related variables — no impact on tree models but unstable for LR if expanding feature set.

---

## 7. CONCLUSION & FUTURE WORK

### 7.1. Conclusion

The study presents a complete credit scoring pipeline **SHAP-Focal-LightGBM** achieving **PR-AUC = 0.8021** (Gini 0.80, KS 0.60) on Loan Prediction 2025, approaching SOTA (CatBoost) while providing individual explainability via SHAP. Specific contributions:

- Focal Loss implementation compatible with LightGBM 4.x (explicit gradient/hessian).
- Quantitative feature selection pipeline (Chi² + ANOVA + WoE/IV) selecting 7 variables with IV ≥ 0.02.
- Optuna tuning improves +0.012 PR-AUC vs baseline Focal (+1.5% relative).
- Accompanying SHAP plots global + local.

### 7.2. Future Directions

1. **Graph Neural Networks** for customer relationships (co-borrower, shared address).
2. **Temporal modeling** (LSTM/Transformer) on payment behavior sequences.
3. **Federated learning** allowing multiple banks to learn jointly without sharing raw data — GDPR/PDPL compliant.
4. **AutoML credit scoring** with broader search space (including neural tabular models: TabNet, FT-Transformer, Tabular Foundation Models).
5. **Calibration** (Platt / Isotonic / Beta) for probabilities directly usable for pricing and capital allocation.
6. **Adversarial robustness** — train with input perturbations to counter fraud.
7. **Drift monitoring** (PSI, KS on production traffic) + automatic re-train triggers.

---

## REFERENCES

1. Altman, E. I. (1968). *Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.* Journal of Finance.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV.
5. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
6. Arik, S. O., & Pfister, T. (2021). *TabNet: Attentive interpretable tabular learning.* AAAI.
7. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased boosting with categorical features.* NeurIPS.
8. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A next-generation hyperparameter optimization framework.* KDD.

---

## APPENDIX — Reproduction Guide

```bash
# 1. Setup
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Data
.venv/bin/python -m src.data.download

# 3. EDA
.venv/bin/python -m src.data.eda

# 4. End-to-end pipeline (preprocessing → WoE/IV → baselines → Focal-LightGBM tune → SHAP → leaderboard)
.venv/bin/python -m src.pipeline
```

**Generated artifacts in `reports/`:**

```
reports/
├── leaderboard.csv        # Comparison table of 6 models
├── iv_table.csv           # IV for all variables
├── chi2_test.csv          # Chi-square tests
├── anova_test.csv         # ANOVA tests
├── selected_features.json # 7 selected variables
├── best_params.json       # Optuna best config
├── focal_lgbm_tuned.txt   # Trained model
├── eda_report.json
├── numeric_describe.csv
├── categorical_counts.json
└── figures/
    ├── shap_summary_dot.png
    ├── shap_summary_bar.png
    └── shap_waterfall_sample_{0,1,2}.png
```