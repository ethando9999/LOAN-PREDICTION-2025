# KẾ HOẠCH TRIỂN KHAI: TỐI ƯU HÓA DỰ ĐOÁN VỠ NỢ TÍN DỤNG BẰNG MÔ HÌNH HYBRID SHAP-FOCAL-LIGHTGBM

> Dựa trên `docs.md` — Loan Prediction Dataset 2025 (Kaggle)
> Ngày lập kế hoạch: 2026-04-20

---

## 0. TỔNG QUAN

**Mục tiêu:** Xây dựng mô hình Hybrid **SHAP-Focal-LightGBM** để dự đoán xác suất vỡ nợ (PD) trên tập dữ liệu Loan Prediction 2025, cân bằng giữa **độ chính xác** và **khả năng giải thích (XAI)**.

**Đóng góp chính:**
1. Hybrid model: LightGBM + Focal Loss (xử lý imbalance) + SHAP (giải thích)
2. Pipeline tiền xử lý có kiểm định thống kê & WoE/IV feature selection
3. So sánh với baselines: LR, RF, XGBoost, CatBoost
4. Framework triển khai (REST API / Batch scoring)

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN

```
Loan-Prediction-2025/
├── data/
│   ├── raw/                 # Dữ liệu gốc từ Kaggle
│   ├── interim/             # Sau làm sạch
│   └── processed/           # Sẵn sàng train
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_shap_explain.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data/                # load, clean
│   ├── features/            # WoE, IV, VIF
│   ├── models/              # focal_lgbm, baselines
│   ├── explain/             # SHAP utils
│   ├── evaluation/          # metrics, KS, PR-AUC
│   └── deployment/          # FastAPI, batch
├── configs/                 # Hyperparam YAML
├── reports/                 # Figures, SHAP plots
├── tests/
├── requirements.txt
└── README.md
```

---

## 2. CÁC GIAI ĐOẠN THỰC HIỆN

### GIAI ĐOẠN 1 — Chuẩn bị dữ liệu & EDA (Tuần 1)

| # | Công việc | Output |
|---|-----------|--------|
| 1.1 | Tải dataset Loan Prediction 2025 từ Kaggle (100K+ records) | `data/raw/` |
| 1.2 | EDA: phân phối biến, tỷ lệ class, missing rate | `01_eda.ipynb` |
| 1.3 | Xác nhận mức độ imbalance (kỳ vọng < 5% positive) | Báo cáo tỷ lệ |
| 1.4 | Phân loại features: demographic / behavior / history | Schema tài liệu hóa |

**Tiêu chí hoàn thành:** Có bảng thống kê mô tả đầy đủ + kết luận về imbalance.

---

### GIAI ĐOẠN 2 — Tiền xử lý dữ liệu (Tuần 2)

Theo **Section 4.1** của `docs.md`:

| # | Kỹ thuật | Thư viện |
|---|----------|----------|
| 2.1 | **Iterative Imputer** + Bayesian Ridge cho missing values | `sklearn.experimental` |
| 2.2 | Kiểm định **Chi-square** (biến categorical vs target) | `scipy.stats` |
| 2.3 | Kiểm định **ANOVA F-test** (biến numeric vs target) | `scipy.stats` |
| 2.4 | Tính **VIF**, loại biến có `VIF > 5` | `statsmodels` |
| 2.5 | Encoding categorical, scaling (nếu cần) | `sklearn` |
| 2.6 | Stratified train/val/test split (70/15/15) | `sklearn` |

**Tiêu chí hoàn thành:** Pipeline tiền xử lý tái lập được + bảng biến bị loại (có lý do).

---

### GIAI ĐOẠN 3 — Feature Engineering (Tuần 3)

Theo **Section 4.2**:

| # | Công việc | Công thức |
|---|-----------|-----------|
| 3.1 | Binning biến liên tục (Decision Tree / quantile) | — |
| 3.2 | Tính **Weight of Evidence (WoE)** cho từng bin | `WoE_i = ln(Good_i / Bad_i)` |
| 3.3 | Tính **Information Value (IV)** | `IV = Σ (Good_i − Bad_i) × WoE_i` |
| 3.4 | Giữ biến có `IV > 0.02`, ưu tiên `IV ∈ [0.1, 0.5]` | — |
| 3.5 | Tạo features tương tác / aggregation nếu cần | — |

**Tiêu chí hoàn thành:** Bảng IV toàn bộ biến + tập feature cuối cùng.

---

### GIAI ĐOẠN 4 — Xây dựng mô hình Baselines (Tuần 4)

| Model | Config | Metric so sánh |
|-------|--------|----------------|
| Logistic Regression | L2, class_weight=balanced | PR-AUC, ROC-AUC |
| Random Forest | n=500, balanced | PR-AUC, ROC-AUC |
| XGBoost | scale_pos_weight | PR-AUC, ROC-AUC |
| CatBoost | auto_class_weights | PR-AUC, ROC-AUC |

**Tiêu chí hoàn thành:** 4 baselines cùng chạy trên cùng split + leaderboard khởi đầu.

---

### GIAI ĐOẠN 5 — Mô hình đề xuất Focal-LightGBM (Tuần 5-6)

Theo **Section 4.3**:

- **Base:** LightGBM (GOSS + EFB)
- **Loss:** Focal Loss tùy chỉnh
  ```
  FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
  ```
- **Siêu tham số khởi đầu:** `γ = 2`, `α ∈ [0.25, 0.75]`

| # | Công việc |
|---|-----------|
| 5.1 | Viết custom objective `focal_loss_obj` + `focal_eval` |
| 5.2 | Tích hợp cost-sensitive (scale_pos_weight) kết hợp focal |
| 5.3 | **Optuna** tune: `num_leaves`, `learning_rate`, `γ`, `α`, `min_child_samples`, `reg_lambda` |
| 5.4 | Stratified K-Fold CV (k=5) để chọn best params |
| 5.5 | Calibration (Platt / Isotonic) cho xác suất tin cậy |

**Tiêu chí hoàn thành:** Mô hình đạt PR-AUC tối thiểu > baseline tốt nhất.

---

### GIAI ĐOẠN 6 — SHAP Explainability (Tuần 7)

Theo **Section 4.4 & 6.1**:

| # | Output |
|---|--------|
| 6.1 | `TreeExplainer` + tính SHAP values toàn bộ tập test |
| 6.2 | **Global:** SHAP Summary Plot, Bar Plot (feature importance) |
| 6.3 | **Local:** Force Plot, Waterfall Plot cho sample minh họa |
| 6.4 | Dependence Plot cho top-10 feature |
| 6.5 | Đối chiếu SHAP vs IV ranking — kiểm tra nhất quán |

**Tiêu chí hoàn thành:** Bộ biểu đồ giải thích + narrative cho 3 case (approved / rejected / borderline).

---

### GIAI ĐOẠN 7 — Đánh giá & So sánh (Tuần 8)

Theo **Section 5.3** — **KHÔNG dùng Accuracy làm metric chính**:

| Metric | Lý do |
|--------|-------|
| **PR-AUC** | Ưu tiên cho imbalance |
| **ROC-AUC** | So sánh xếp hạng |
| **Gini** = 2·AUC − 1 | Tiêu chuẩn tín dụng |
| **F2-score** | Ưu tiên Recall |
| **KS-test** | Phân biệt Good/Bad |
| **Brier Score** | Độ hiệu chỉnh xác suất |

**Output:** Bảng leaderboard + biểu đồ so sánh + kiểm định McNemar giữa các mô hình.

---

### GIAI ĐOẠN 8 — Triển khai (Tuần 9)

Theo **Section 6.2**:

| # | Công việc |
|---|-----------|
| 8.1 | **REST API** bằng FastAPI — endpoint `/score` |
| 8.2 | **Batch scoring** pipeline (CSV in → scored CSV out) |
| 8.3 | Docker image + docker-compose |
| 8.4 | Logging, monitoring (prediction drift) |
| 8.5 | Unit test + integration test (pytest) |

**Tiêu chí hoàn thành:** API chạy được trong Docker + test pass.

---

### GIAI ĐOẠN 9 — Viết báo cáo & Trình bày (Tuần 10)

- Cập nhật đầy đủ các section trong `docs.md` bằng kết quả thực tế.
- Thảo luận hạn chế: **Fraud adversarial**, **Concept drift**, **Data leakage**.
- Future work: GNN, Temporal modeling, Federated learning, AutoML.

---

## 3. RỦI RO & GIẢM THIỂU

| Rủi ro | Giảm thiểu |
|--------|------------|
| Data leakage (target leakage) | Kiểm tra tương quan feature–target; time-based split nếu có timestamp |
| Overfitting do tune quá | Nested CV, early stopping |
| Focal Loss không hội tụ | Fallback `binary_logloss` + `is_unbalance=True` |
| SHAP quá chậm trên 100K rows | Sample 5-10K cho plots; dùng `approximate=True` |
| Phụ thuộc môi trường | Cố định phiên bản trong `requirements.txt` + `conda env` |

---

## 4. NGĂN XẾP CÔNG NGHỆ (TECH STACK)

```
Python 3.11
├── Data:    pandas, numpy, pyarrow
├── ML:      scikit-learn, lightgbm, xgboost, catboost
├── Stats:   scipy, statsmodels
├── Tune:    optuna
├── XAI:     shap
├── Deploy:  fastapi, uvicorn, docker
└── Test:    pytest, pytest-cov
```

---

## 5. CHECKLIST HOÀN THÀNH

- [ ] Dataset đã tải, EDA xong
- [ ] Pipeline tiền xử lý có kiểm định thống kê
- [ ] WoE/IV feature selection
- [ ] 4 baselines chạy được
- [ ] Focal-LightGBM tune xong, vượt baselines
- [ ] SHAP global + local plots
- [ ] Bảng metric đầy đủ (PR-AUC, KS, Gini, F2)
- [ ] API/batch scoring hoạt động
- [ ] Báo cáo cuối cùng

---

## 6. BƯỚC KẾ TIẾP

1. Xác nhận kế hoạch này với supervisor.
2. Tải dataset từ Kaggle → `data/raw/`.
3. Khởi tạo cấu trúc project + `requirements.txt`.
4. Bắt đầu **Giai đoạn 1 — EDA**.
