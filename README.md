# Loan Prediction 2025 — Hybrid SHAP-Focal-LightGBM

Hệ thống dự đoán vỡ nợ tín dụng kết hợp **LightGBM + Focal Loss** (chính xác trên imbalance) và **SHAP** (minh bạch giải trình).

Tham khảo `docs.md` cho background nghiên cứu và `loan-prediction-planning.md` cho lộ trình triển khai.

## Setup

```bash
# Tạo venv (dùng uv)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# Cài Kaggle credentials
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Tải dataset
.venv/bin/python -m src.data.download
```

## Cấu trúc

```
configs/      # YAML config
data/         # raw / interim / processed
notebooks/    # EDA, preprocessing, modeling, SHAP, evaluation
src/
  data/       # download, loader
  features/   # stat tests (chi2, anova, VIF), WoE/IV
  models/     # focal_lgbm, baselines
  explain/    # SHAP utils
  evaluation/ # metrics (PR-AUC, KS, Gini, F2)
  deployment/ # FastAPI scoring
reports/      # figures
tests/
```
