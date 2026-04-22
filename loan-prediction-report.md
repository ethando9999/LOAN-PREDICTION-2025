# TỐI ƯU HÓA DỰ ĐOÁN VỠ NỢ TÍN DỤNG BẰNG MÔ HÌNH HYBRID SHAP-FOCAL-LIGHTGBM TRÊN TẬP DỮ LIỆU LOAN PREDICTION 2025

**Tác giả:** Thanh-Chien Do
**Email:** ethando369@gmail.com
**Ngày báo cáo:** 2026-04-22
**Mã nguồn:** `src/`, `configs/config.yaml`
**Artifacts:** `reports/`

---

## TÓM TẮT (ABSTRACT)

**Bối cảnh.** Trong kỷ nguyên Fintech và biến động kinh tế 2025, hệ thống chấm điểm tín dụng đòi hỏi đồng thời độ chính xác cao và tính minh bạch để đáp ứng yêu cầu pháp lý.

**Vấn đề.** Các mô hình thống kê truyền thống (Logistic Regression) dễ giải thích nhưng hiệu năng thấp; các mô hình SOTA (XGBoost, CatBoost, Deep Learning) có AUC cao nhưng là "hộp đen" và xử lý kém trên tình trạng mất cân bằng dữ liệu.

**Giải pháp.** Nghiên cứu đề xuất mô hình **Hybrid SHAP-Focal-LightGBM**, kết hợp (i) LightGBM với **Focal Loss** tùy biến để tập trung vào mẫu khó (thiểu số vỡ nợ), (ii) **Optuna** để tinh chỉnh siêu tham số, và (iii) **SHAP** để minh bạch hóa quyết định ở cả cấp độ toàn cục và cục bộ.

**Kết quả.** Thử nghiệm trên *Loan Prediction Dataset 2025* (Kaggle, 20.000 bản ghi, default rate 20%), mô hình **Focal-LightGBM tuned** đạt **PR-AUC = 0.8021**, **ROC-AUC = 0.9017**, **Gini = 0.8035**, **KS = 0.6046** trên tập test — gần như ngang CatBoost (chênh 0.001 PR-AUC) và vượt rõ rệt Logistic Regression (+0.086 PR-AUC) cùng các baseline tree khác. Kết hợp SHAP cho phép giải trình từng quyết định cấp tín dụng — minh chứng rằng khoảng cách Accuracy-vs-Explainability có thể được thu hẹp.

**Từ khóa.** Credit scoring, Focal Loss, LightGBM, SHAP, Imbalanced Learning, Explainable AI.

---

## 1. GIỚI THIỆU

### 1.1. Phát biểu bài toán

Bài toán dự đoán xác suất vỡ nợ (Probability of Default — PD) là phân loại nhị phân:

$$
y \in \{0, 1\}, \quad y = 1 \text{ nếu khách hàng vỡ nợ}
$$

Mục tiêu: xây dựng hàm $f: \mathcal{X} \to [0,1]$ ước lượng $P(y=1 \mid \mathbf{x})$, với $\mathbf{x}$ là vector đặc trưng demographic/behavior/history.

### 1.2. Tầm quan trọng

- Giảm tỷ lệ nợ xấu (NPL) và bảo vệ thanh khoản.
- Tự động hóa quy trình duyệt tín dụng với tính giải trình bắt buộc theo Basel III/IV, IFRS 9.
- Hỗ trợ khiếu nại — khách hàng có quyền biết *vì sao* bị từ chối.

### 1.3. Mục tiêu nghiên cứu

1. Mô hình hóa quan hệ **phi tuyến** giữa biến tín dụng và PD.
2. Xử lý tốt **mất cân bằng** (default rate ≈ 20%).
3. Cung cấp **giải thích** cả ở cấp mô hình (global) và cấp từng đơn (local).

---

## 2. NGHIÊN CỨU LIÊN QUAN

| Hướng tiếp cận | Đại diện | Ưu | Nhược |
|----------------|----------|----|-------|
| Thống kê truyền thống | Logistic Regression, Altman Z-score (1968), Credit Scorecard | Giải thích trực tiếp qua hệ số | Giả định tuyến tính, yếu với tương tác |
| Tree-based Ensemble | XGBoost [Chen & Guestrin 2016], LightGBM [Ke et al. 2017], CatBoost | SOTA trên dữ liệu bảng | Hộp đen |
| Deep Learning cho tabular | TabNet [Arik & Pfister 2021], TabTransformer | Học đặc trưng mạnh | Dữ liệu + tài nguyên lớn, khó triển khai |
| Rule-based | `IF overdue > X THEN reject` | Minh bạch tuyệt đối | Cứng nhắc, khó mở rộng |
| XAI | SHAP [Lundberg & Lee 2017], LIME, PDP | Giải thích mô hình đen | Chi phí tính toán |

**Khoảng trống.** Thiếu khung tích hợp cả (a) xử lý imbalance bằng loss function, (b) tuning hệ thống, (c) giải trình cấp đơn — đồng thời trên dữ liệu tín dụng thực.

---

## 3. ĐÓNG GÓP

1. **Custom Focal Loss cho LightGBM:** triển khai gradient/hessian tường minh để tương thích LightGBM 4.x (qua `params["objective"]`).
2. **Feature selection định lượng:** pipeline Chi-square + ANOVA F + Weight-of-Evidence + Information Value, giữ lại 7/21 biến đạt `IV ≥ 0.02`.
3. **Hybrid SHAP-Focal-LightGBM:** mô hình cuối kết hợp Optuna tuning và SHAP explainer, đóng gói reproducible trong `src/pipeline.py`.

---

## 4. PHƯƠNG PHÁP

### 4.1. Tiền xử lý

- **Target flip:** đổi nhãn sang quy ước `1 = default, 0 = paid_back` để phù hợp với các metric imbalanced (PR-AUC, F2 ưu tiên nhóm default).
- **Encoding:** `OrdinalEncoder` cho 6 biến categorical (gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade).
- **Missing values:** dataset không có NA, nên `IterativeImputer` (Bayesian Ridge) khai báo trong pipeline nhưng không kích hoạt.
- **Split:** stratified 70/15/15 (train/val/test) với `random_state=42`.

### 4.2. Kiểm định thống kê

Tính thống kê trên tập train:

- **Chi-square** cho biến categorical vs target.
- **ANOVA F-test** cho biến numeric vs target.

Kết quả tiêu biểu (xem `reports/anova_test.csv`, `reports/chi2_test.csv`):

| Feature | Test | p-value |
|---------|------|---------|
| `debt_to_income_ratio` | ANOVA | 5.1e-162 |
| `credit_score` | ANOVA | 2.7e-125 |
| `interest_rate` | ANOVA | 4.8e-42 |
| `gender` | Chi² | 0.64 (không có ý nghĩa) |

### 4.3. Weight of Evidence & Information Value

$$
WoE_i = \ln\!\left(\frac{\text{Good}_i / \text{Good}_{\text{total}}}{\text{Bad}_i / \text{Bad}_{\text{total}}}\right), \quad
IV = \sum_i (\text{Good}\%_i - \text{Bad}\%_i)\, WoE_i
$$

Giữ biến nếu $IV \geq 0.02$. Kết quả top 10 (xem `reports/iv_table.csv`):

| Feature | IV | Diễn giải |
|---------|----|-----------|
| `employment_status` | **1.903** | Rất mạnh (thường > 0.5 là bất thường — kiểm tra leakage) |
| `debt_to_income_ratio` | 0.337 | Mạnh |
| `credit_score` | 0.294 | Mạnh |
| `grade_subgrade` | 0.271 | Mạnh |
| `interest_rate` | 0.088 | Trung bình |
| `delinquency_history` | 0.043 | Yếu–trung bình |
| `num_of_delinquencies` | 0.033 | Yếu–trung bình |
| `loan_purpose` | 0.007 | Loại |
| `total_credit_limit` | 0.006 | Loại |
| `age`, `gender`, `marital_status` | < 0.01 | Loại |

> **Lưu ý về leakage:** `employment_status` có IV = 1.90 — bất thường cao. Trong thực tế phải kiểm tra xem biến có được *cập nhật sau* khi đã vỡ nợ không. Ở đây ta giữ lại vì dataset Kaggle là snapshot cố định.

**Feature set cuối:** 7 biến được chọn.

### 4.4. Đa cộng tuyến (VIF)

Tính VIF trên các biến numeric đã chọn:

| Feature | VIF |
|---------|-----|
| `num_of_delinquencies` | 17.9 |
| `credit_score` | 15.9 |
| `delinquency_history` | 15.4 |
| `interest_rate` | 15.3 |
| `debt_to_income_ratio` | 4.0 |

VIF cao chủ yếu vì scale không chuẩn hóa; **không drop** vì mô hình tree-based (LightGBM/XGBoost/RF/CatBoost) ít bị ảnh hưởng bởi multicollinearity. Chỉ logistic regression sẽ chịu chi phí nhỏ về diễn giải hệ số — chấp nhận đánh đổi.

### 4.5. Mô hình đề xuất — Focal-LightGBM

**Focal Loss** [Lin et al. 2017], áp dụng cho phân loại nhị phân:

$$
\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t \,(1 - p_t)^{\gamma}\, \log p_t
$$

với $p_t = p$ nếu $y=1$, $1-p$ nếu $y=0$, và $\alpha_t$ là trọng số lớp. Hệ số $(1-p_t)^\gamma$ **giảm đóng góp của mẫu dễ** (đã được phân loại tốt) để mô hình tập trung vào các mẫu khó — đặc biệt là mẫu vỡ nợ thiểu số.

**Gradient & Hessian** (triển khai tại `src/models/focal_lgbm.py:focal_loss_obj`):

$$
\frac{\partial \mathcal{L}}{\partial z} = \alpha_t\, (1-p_t)^{\gamma}\, \big[\gamma\, p_t \log p_t + p_t - 1\big]\,(2y - 1)
$$

với $z$ là raw-score (logit). Hessian tính tương tự rồi lấy trị tuyệt đối + epsilon để đảm bảo positive-definite khi LightGBM cập nhật lá.

**Tích hợp với LightGBM 4.x:** truyền callable qua `params["objective"]` (API cũ `fobj` đã bị loại bỏ).

### 4.6. Tinh chỉnh siêu tham số — Optuna

Không gian tìm kiếm (30 trial, TPE sampler, `seed=42`):

| Tham số | Dải |
|---------|-----|
| `num_leaves` | [15, 255] |
| `learning_rate` | [0.01, 0.2] log-uniform |
| `min_child_samples` | [5, 100] |
| `reg_lambda` | [1e-3, 10] log-uniform |
| `feature_fraction` | [0.6, 1.0] |
| `bagging_fraction` | [0.6, 1.0] |
| `bagging_freq` | [0, 10] |
| `alpha` (focal) | [0.25, 0.9] |
| `gamma` (focal) | [0.5, 4.0] |

Mục tiêu: **PR-AUC** trên tập val.

**Cấu hình tốt nhất** (xem `reports/best_params.json`):

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

Điểm đáng chú ý: **γ ≈ 3.5** (cao hơn mặc định 2.0) — hệ thống "ép" xuống các mẫu khó mạnh hơn bình thường; **α ≈ 0.33** — hơi ưu ái lớp âm (paid_back), cân bằng lại với hiệu ứng focusing.

### 4.7. SHAP Explainability

$$
g(\mathbf{z}') = \phi_0 + \sum_{j=1}^{M} \phi_j z'_j, \quad \phi_j = \text{đóng góp của feature } j
$$

Sử dụng `shap.TreeExplainer` trên mô hình Focal-LightGBM đã tune, với 3.000 mẫu random từ tập test (từ 3.000 bản ghi) để tiết kiệm chi phí tính toán.

---

## 5. THỰC NGHIỆM

### 5.1. Dataset

- **Nguồn:** [`nabihazahid/loan-prediction-dataset-2025`](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025)
- **Kích thước thực tế:** 20.000 bản ghi × 22 cột (docs ban đầu ước 100K+ — khác biệt với thực tế).
- **Target:** `loan_paid_back` (1 = trả xong, 0 = default). **Đã flip** thành `1 = default` trong pipeline.
- **Tỷ lệ class:** 80% paid_back vs **20% default** — imbalanced trung bình (docs kỳ vọng < 5%, thực tế nhẹ hơn nhưng vẫn cần xử lý).
- **Missing values:** 0 (dataset đã được làm sạch trước).

### 5.2. Môi trường

```
Python 3.12.3
pandas 2.2.x, scikit-learn 1.5.x
lightgbm 4.6.x, xgboost 3.2.0, catboost 1.2.x
shap 0.44.x, optuna 3.5+
```

### 5.3. Metric

Theo thông lệ credit scoring, **không** dùng Accuracy làm metric chính. Sử dụng:

| Metric | Ý nghĩa |
|--------|---------|
| **PR-AUC** | *Metric chính* — ưu tiên khi imbalance |
| ROC-AUC | Khả năng xếp hạng rủi ro |
| Gini = 2·AUC − 1 | Tiêu chuẩn ngành tín dụng |
| F2-score | Nhấn trọng số Recall (không bỏ sót default) |
| KS | Khoảng cách phân phối Good/Bad |
| Brier | Độ hiệu chỉnh xác suất (thấp là tốt) |

### 5.4. Baselines

Bốn mô hình so sánh với cấu hình cân bằng imbalance:

| Model | Xử lý imbalance |
|-------|----------------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest (n=500) | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| CatBoost | `auto_class_weights='Balanced'` |

### 5.5. Kết quả chính

**Leaderboard trên tập test** (sắp theo PR-AUC):

| # | Model | PR-AUC | ROC-AUC | Gini | F2 | KS | Brier |
|---|-------|--------|---------|------|-----|-----|-------|
| 1 | CatBoost | **0.8031** | **0.9023** | **0.8046** | 0.6988 | **0.6162** | 0.1156 |
| 2 | **Focal-LightGBM (tuned)** | **0.8021** | 0.9017 | 0.8035 | 0.6006 | 0.6046 | 0.1197 |
| 3 | Focal-LightGBM (base) | 0.7900 | 0.8931 | 0.7862 | 0.6684 | 0.5905 | 0.1345 |
| 4 | Random Forest | 0.7827 | 0.8808 | 0.7617 | 0.6100 | 0.5916 | **0.0796** |
| 5 | XGBoost | 0.7768 | 0.8832 | 0.7664 | 0.6488 | 0.5705 | 0.1002 |
| 6 | Logistic Regression | 0.7158 | 0.8658 | 0.7316 | 0.6736 | 0.5731 | 0.1472 |

**Quan sát chính:**

1. **Focal-LightGBM tuned (#2) tiệm cận CatBoost (#1)** — chênh 0.001 PR-AUC (trong phạm vi sai số seed).
2. **Optuna cải thiện base Focal-LightGBM +0.012 PR-AUC** (0.790 → 0.802) — khẳng định giá trị của search γ/α cùng tree hyperparams.
3. **Focal-LightGBM tuned vượt Logistic Regression +0.086 PR-AUC** (+12% tương đối) — giá trị của mô hình phi tuyến rõ ràng.
4. **Random Forest có Brier thấp nhất (0.0796)** — xác suất dự đoán hiệu chỉnh tốt, dù PR-AUC không đứng đầu.
5. **Logistic Regression F2 = 0.674**, cao hơn Focal-LightGBM tuned (0.601) — hệ quả của `class_weight='balanced'` đẩy Recall lên, nhưng Precision giảm mạnh làm PR-AUC tụt.

### 5.6. Phân tích SHAP

**Global importance** (xem `reports/figures/shap_summary_bar.png`, `shap_summary_dot.png`) cho thấy thứ tự đóng góp trung bình |SHAP|:

1. `employment_status`
2. `debt_to_income_ratio`
3. `credit_score`
4. `grade_subgrade`
5. `interest_rate`

Thứ tự này **nhất quán với bảng IV** — tín hiệu mô hình hoạt động hợp lý, không bị nhiễu bởi tương tác giả tạo.

**Local explanations** (xem `reports/figures/shap_waterfall_sample_{0,1,2}.png`): mỗi biểu đồ waterfall tách `log-odds(default)` cho một khách hàng cụ thể thành `base_value + Σ φ_j`, giúp trả lời câu hỏi "*Vì sao khách hàng X bị xếp hạng rủi ro cao?*" — phục vụ trực tiếp cho compliance và explanability request.

---

## 6. THẢO LUẬN

### 6.1. Interpretability vs Accuracy — khoảng cách đã thu hẹp

| So sánh | PR-AUC | Giải thích |
|---------|--------|-----------|
| LR (dễ giải thích nhất) | 0.716 | Hệ số tuyến tính trực tiếp |
| Focal-LightGBM tuned + SHAP | **0.802** | SHAP cung cấp giải thích cấp đơn + toàn cục |

Focal-LightGBM + SHAP đạt **+12% PR-AUC** so với LR mà vẫn cung cấp giải thích cấp đơn. Đây chính là *khoảng trống Accuracy-vs-Explainability* được bít lại.

### 6.2. Production Deployment (dự kiến — Giai đoạn 8)

- **REST API scoring** qua FastAPI: `/score` nhận JSON đặc trưng, trả `{pd, risk_grade, top_shap_features}`.
- **Batch scoring**: pipeline đọc CSV → scored CSV (phục vụ re-evaluation định kỳ portfolio).
- **Monitoring**: log distribution của input features và output PD để phát hiện **data drift** / **concept drift**.

### 6.3. Hạn chế

1. **Kích thước dataset.** 20K bản ghi nhỏ hơn kỳ vọng 100K+ trong docs; các tune hyperparam có thể khác nếu chạy trên dữ liệu thực lớn.
2. **Dataset Kaggle có thể không phản ánh portfolio thực.** `employment_status` với IV = 1.90 đáng ngờ; cần kiểm tra xem có bị update *sau thời điểm ra quyết định* không (label leakage).
3. **Single train/test split.** Không có K-Fold cross-validation cho leaderboard cuối — con số có variance ±0.005–0.01 PR-AUC.
4. **Fraud adversarial attack.** Mô hình hiện tại chưa có cơ chế phát hiện đặc trưng giả mạo cố ý (e.g., khai `annual_income` bị thổi phồng).
5. **Concept drift.** Hành vi khách hàng thay đổi theo chu kỳ kinh tế — mô hình 2025 có thể mất hiệu lực 2026+ nếu không re-train.
6. **Data leakage risk.** Biến `grade_subgrade` có thể là hệ quả của scoring nội bộ ngân hàng (circular signal).
7. **Calibration.** Chưa áp dụng Platt/Isotonic calibration — xác suất hiện tại có thể không phản ánh đúng tần suất default thực.
8. **VIF cao (15-17)** cho các biến liên quan đến delinquency — không ảnh hưởng tree models nhưng sẽ gây bất ổn cho LR nếu mở rộng feature set.

---

## 7. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 7.1. Kết luận

Nghiên cứu trình bày một pipeline credit scoring hoàn chỉnh **SHAP-Focal-LightGBM** đạt **PR-AUC = 0.8021** (Gini 0.80, KS 0.60) trên Loan Prediction 2025, tiệm cận SOTA (CatBoost) trong khi cung cấp khả năng giải trình cấp đơn qua SHAP. Các đóng góp cụ thể:

- Triển khai Focal Loss tương thích LightGBM 4.x (gradient/hessian tường minh).
- Pipeline feature selection định lượng (Chi² + ANOVA + WoE/IV) chọn 7 biến có IV ≥ 0.02.
- Tuning Optuna cải thiện +0.012 PR-AUC vs baseline Focal (+1.5% tương đối).
- Bộ biểu đồ SHAP global + local đi kèm model.

### 7.2. Hướng phát triển

1. **Graph Neural Networks** cho quan hệ giữa khách hàng (co-borrower, shared address).
2. **Temporal modeling** (LSTM/Transformer) trên chuỗi hành vi thanh toán.
3. **Federated learning** cho phép nhiều ngân hàng học chung mà không chia sẻ raw data — tuân thủ GDPR/PDPL.
4. **AutoML credit scoring** với search space rộng hơn (bao gồm neural tabular models: TabNet, FT-Transformer, Tabular Foundation Models).
5. **Calibration** (Platt / Isotonic / Beta) để xác suất dùng được trực tiếp cho pricing và capital allocation.
6. **Adversarial robustness** — huấn luyện với perturbation trên input để chống fraud.
7. **Drift monitoring** (PSI, KS trên production traffic) + trigger re-train tự động.

---

## TÀI LIỆU THAM KHẢO

1. Altman, E. I. (1968). *Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.* Journal of Finance.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV.
5. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
6. Arik, S. O., & Pfister, T. (2021). *TabNet: Attentive interpretable tabular learning.* AAAI.
7. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased boosting with categorical features.* NeurIPS.
8. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A next-generation hyperparameter optimization framework.* KDD.

---

## PHỤ LỤC — Hướng dẫn tái lập

```bash
# 1. Setup
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Dữ liệu
.venv/bin/python -m src.data.download

# 3. EDA
.venv/bin/python -m src.data.eda

# 4. Pipeline end-to-end (preprocessing → WoE/IV → baselines → Focal-LightGBM tune → SHAP → leaderboard)
.venv/bin/python -m src.pipeline
```

**Artifacts sinh ra tại `reports/`:**

```
reports/
├── leaderboard.csv        # Bảng so sánh 6 models
├── iv_table.csv           # IV toàn bộ biến
├── chi2_test.csv          # Kiểm định Chi-square
├── anova_test.csv         # Kiểm định ANOVA
├── selected_features.json # 7 biến được chọn
├── best_params.json       # Optuna best config
├── focal_lgbm_tuned.txt   # Model đã train
├── eda_report.json
├── numeric_describe.csv
├── categorical_counts.json
└── figures/
    ├── shap_summary_dot.png
    ├── shap_summary_bar.png
    └── shap_waterfall_sample_{0,1,2}.png
```
