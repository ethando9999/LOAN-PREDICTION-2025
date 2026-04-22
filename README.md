# Loan Prediction 2025 — Hybrid SHAP-Focal-LightGBM

Credit default prediction system combining **LightGBM + Focal Loss** (high accuracy on imbalanced data) with **SHAP** (transparent, auditable explanations) on the [Loan Prediction Dataset 2025](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025) from Kaggle.

**Author:** Thanh-Chien Do — ethando369@gmail.com

See [`docs.md`](docs.md) for research background, [`loan-prediction-planning.md`](loan-prediction-planning.md) for the execution roadmap, and [`loan-prediction-report.md`](loan-prediction-report.md) for the final report.

---

## Results

Test-set leaderboard (primary metric: PR-AUC):

| # | Model | PR-AUC | ROC-AUC | Gini | F2 | KS | Brier |
|---|-------|--------|---------|------|----|----|----|
| 1 | CatBoost | **0.8031** | **0.9023** | **0.8046** | 0.6988 | **0.6162** | 0.1156 |
| 2 | **Focal-LightGBM (tuned)** | **0.8021** | 0.9017 | 0.8035 | 0.6006 | 0.6046 | 0.1197 |
| 3 | Focal-LightGBM (base) | 0.7900 | 0.8931 | 0.7862 | 0.6684 | 0.5905 | 0.1345 |
| 4 | Random Forest | 0.7827 | 0.8808 | 0.7617 | 0.6100 | 0.5916 | **0.0796** |
| 5 | XGBoost | 0.7768 | 0.8832 | 0.7664 | 0.6488 | 0.5705 | 0.1002 |
| 6 | Logistic Regression | 0.7158 | 0.8658 | 0.7316 | 0.6736 | 0.5731 | 0.1472 |

**Highlights:**
- Focal-LightGBM (tuned) matches CatBoost within 0.001 PR-AUC (seed-level noise) and beats Logistic Regression by **+0.086 PR-AUC (+12%)**.
- Optuna tuning (30 trials, TPE sampler) lifts base Focal-LightGBM by **+0.012 PR-AUC** — best config has `γ=3.52`, `α=0.33`, `num_leaves=16`, `lr=0.115`.
- SHAP global ranking is consistent with IV ranking — no noisy interactions.

---

## Dataset

- **Source:** `nabihazahid/loan-prediction-dataset-2025` (Kaggle)
- **Shape:** 20,000 rows × 22 columns
- **Target:** `loan_paid_back` (flipped to `1 = default`, `0 = paid_back` in the pipeline)
- **Class balance:** 80% paid / **20% default**
- **Missing values:** 0

---

## Setup

```bash
# Create venv (using uv)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# Install Kaggle credentials
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
.venv/bin/python -m src.data.download
```

## Run

```bash
# Exploratory data analysis
.venv/bin/python -m src.data.eda

# End-to-end pipeline:
#   load → split → preprocess → stat tests → WoE/IV →
#   baselines → Focal-LightGBM → Optuna → SHAP → leaderboard
.venv/bin/python -m src.pipeline
```

Artifacts are written to `reports/` and `reports/figures/`.

---

## Project structure

```
configs/                 YAML configuration
data/
  raw/                   Kaggle CSV dump
  interim/
  processed/             Parquet splits (train/val/test)
notebooks/               Exploration notebooks
src/
  data/                  download, loader, eda
  features/              stat_tests (chi2, anova, VIF), woe_iv, preprocessing
  models/                focal_lgbm, baselines, tune_focal
  explain/               SHAP utilities
  evaluation/            metrics (PR-AUC, ROC-AUC, Gini, F2, KS, Brier)
  deployment/            (Phase 8 — REST API scoring, not implemented)
  pipeline.py            End-to-end driver
reports/                 Leaderboard, IV table, stat tests, figures
tests/
```

## Methodology (summary)

1. **Preprocessing** — stratified 70/15/15 split, ordinal encoding for 6 categorical columns, target flip.
2. **Statistical tests** — Chi-square for categoricals, ANOVA F-test for numerics (`reports/chi2_test.csv`, `reports/anova_test.csv`).
3. **Feature selection via WoE/IV** — keep features with `IV ≥ 0.02`, 7/21 retained (`reports/iv_table.csv`).
4. **VIF** — reported as diagnostic only; not used for dropping (tree models tolerant to multicollinearity).
5. **Baselines** — Logistic Regression, Random Forest, XGBoost, CatBoost, all with class-weight / scale_pos_weight balancing.
6. **Focal-LightGBM** — custom objective with explicit gradient/hessian passed through `params["objective"]` (LightGBM 4.x API).
7. **Optuna** — 30 trials on validation PR-AUC over `num_leaves`, `learning_rate`, `min_child_samples`, `reg_lambda`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `alpha`, `gamma`.
8. **SHAP** — `TreeExplainer` on 3,000 test samples; global summary (bar + beeswarm) and local waterfall plots.

## Tech stack

```
Python 3.12
pandas 2.2, numpy, pyarrow, scipy, statsmodels
scikit-learn 1.5, lightgbm 4.6, xgboost 3.2, catboost 1.2
optuna 3.5+, shap 0.44+
matplotlib, seaborn
fastapi, uvicorn (reserved for Phase 8)
pytest
```

## Status (per [`loan-prediction-planning.md`](loan-prediction-planning.md))

- [x] Phase 1 — EDA
- [x] Phase 2 — Preprocessing (stat tests + encoding)
- [x] Phase 3 — Feature engineering (WoE / IV)
- [x] Phase 4 — Baselines (LR, RF, XGBoost, CatBoost)
- [x] Phase 5 — Focal-LightGBM + Optuna tuning
- [x] Phase 6 — SHAP explainability
- [x] Phase 7 — Leaderboard & comparison
- [ ] Phase 8 — Deployment (FastAPI + Docker) *(skipped)*
- [x] Phase 9 — Final research report — see [`loan-prediction-report.md`](loan-prediction-report.md)
