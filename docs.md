````md
# TIÊU ĐỀ: TỐI ƯU HÓA DỰ ĐOÁN VỠ NỢ TÍN DỤNG BẰNG MÔ HÌNH HYBRID [TÊN MÔ HÌNH] TRÊN TẬP DỮ LIỆU LOAN PREDICTION 2025

## TÓM TẮT (ABSTRACT)

**Bối cảnh:** Trong kỷ nguyên số tài chính (Fintech) và sự biến động kinh tế năm 2025, hệ thống chấm điểm tín dụng (Credit Scoring) đòi hỏi sự kết hợp giữa độ chính xác cao và tính minh bạch tuyệt đối để tuân thủ các quy định pháp lý.

**Vấn đề:** Các phương pháp hiện tại đang gặp phải sự đánh đổi lớn: mô hình thống kê dễ giải thích nhưng độ chính xác thấp, trong khi các mô hình SOTA (Deep Learning, Ensemble) lại là những "hộp đen" và xử lý kém trước tình trạng mất cân bằng dữ liệu cực hạn. Việc áp dụng các hệ thống luật cứng nhắc kiểu cũ cũng là một rào cản lớn khi hành vi khách hàng thay đổi nhanh chóng.

**Giải pháp:** Nghiên cứu này đề xuất một hướng tiếp cận Hybrid (Lai), kết hợp sức mạnh phân loại của *LightGBM/XGBoost* với công cụ giải thích mô hình XAI (SHAP) để bóc tách và minh bạch hóa quá trình ra quyết định.

**Kết quả:** Đánh giá trên tập dữ liệu *Loan Prediction Dataset 2025* từ Kaggle, mô hình đề xuất dự kiến không chỉ cải thiện các chỉ số hiệu năng (AUC, F1-Score, KS-test) đối với nhóm nợ xấu (thiểu số) mà còn cung cấp khả năng giải trình chi tiết cho từng quyết định cấp tín dụng.

---

# 1. GIỚI THIỆU (INTRODUCTION)

## 1.1. Phát biểu bài toán

Định nghĩa việc dự đoán xác suất vỡ nợ (Probability of Default - PD) là một bài toán phân loại nhị phân:

\[
y \in \{0,1\}
\]

Trong đó mục tiêu là xác định chính xác khách hàng có khả năng trễ hạn hoặc mất khả năng thanh toán.

## 1.2. Tầm quan trọng

- Giảm thiểu rủi ro nợ xấu (NPL - Non-Performing Loans).
- Bảo vệ thanh khoản ngân hàng.
- Tối ưu quy trình phê duyệt tín dụng tự động.

## 1.3. Mục tiêu nghiên cứu

Xây dựng kiến trúc mô hình cân bằng:

1. Khả năng học dữ liệu phi tuyến phức tạp.  
2. Khả năng giải trình rõ ràng để triển khai thực tế.

---

# 2. NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

## 2.1. Phương pháp thống kê truyền thống

- Logistic Regression (LR)
- Credit Scorecard
- Altman Z-score (1968)

Ưu điểm: dễ giải thích.  
Nhược điểm: giả định tuyến tính mạnh.

## 2.2. Tree-based Ensemble

- XGBoost
- LightGBM
- CatBoost

Ưu điểm: hiệu năng rất cao trên dữ liệu bảng.  
Nhược điểm: khó giải thích.

## 2.3. Deep Learning cho dữ liệu bảng

- TabNet
- TabTransformer
- Tabular Foundation Models

Ưu điểm: học đặc trưng mạnh.  
Nhược điểm: tốn tài nguyên.

## 2.4. Rule-based Systems

Các luật IF-THEN truyền thống:

```text
IF overdue_group3 > 0 THEN high_risk
````

Ưu điểm: minh bạch tuyệt đối.
Nhược điểm: cứng nhắc, khó mở rộng.

## 2.5. Explainable AI

* SHAP
* LIME
* Partial Dependence Plot

---

# 3. KHOẢNG TRỐNG NGHIÊN CỨU & ĐÓNG GÓP

## 3.1. Research Gaps

### Gap 1: Accuracy vs Explainability

Mô hình mạnh thì khó giải thích, mô hình dễ giải thích thì yếu.

### Gap 2: Extreme Imbalance

Tỷ lệ nợ xấu thường dưới 5%.

### Gap 3: Alternative Data

Các biến hành vi số mới khó xử lý bằng luật cứng.

---

## 3.2. Contributions

### Đề xuất mô hình Hybrid:

**SHAP-Focal-LightGBM**

### Xử lý imbalance:

* Cost-sensitive learning
* Focal Loss

### Giải thích:

* Global explanation
* Local explanation

---

# 4. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

## 4.1. Tiền xử lý dữ liệu

### Missing Values

* Iterative Imputer
* Bayesian Ridge

### Kiểm định thống kê

* Chi-square
* ANOVA F-test

### Đa cộng tuyến

Loại biến có:

$$
VIF > 5
$$

---

## 4.2. Feature Engineering

### Weight of Evidence (WoE)

$$
WoE_i = \ln \left(\frac{Good_i}{Bad_i}\right)
$$

### Information Value (IV)

$$
IV = \sum (Good_i - Bad_i)\times WoE_i
$$

Giữ biến nếu:

$$
IV > 0.02
$$

---

## 4.3. Mô hình đề xuất: Focal LightGBM

LightGBM dùng:

* GOSS
* EFB

### Focal Loss

FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)

Trong đó:

* (p_t): xác suất đúng
* (\alpha_t): trọng số lớp
* (\gamma): focusing parameter

Ví dụ:

$$
\gamma = 2
$$

---

## 4.4. SHAP Explainability

g(z')=\phi_0+\sum_{j=1}^{M}\phi_j z'_j

Trong đó:

* (\phi_0): base value
* (\phi_j): đóng góp của feature j

---

# 5. THỰC NGHIỆM & ĐÁNH GIÁ

## 5.1. Dataset

Loan Prediction Dataset 2025:

* link: https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025
* 100,000+ records
* demographic features
* behavior features
* history features

## 5.2. Môi trường

* Python
* Scikit-learn
* LightGBM
* SHAP
* Optuna

## 5.3. Metrics

Không dùng Accuracy làm metric chính.

### Metrics quan trọng:

* PR-AUC
* ROC-AUC
* Gini
* F2-score
* KS-test

## 5.4. Baselines

So sánh với:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

---

# 6. THẢO LUẬN

## 6.1. Interpretability

* SHAP Summary Plot
* SHAP Force Plot
* Waterfall Plot

## 6.2. Production Deployment

* REST API scoring
* Batch scoring
* Real-time inference

## 6.3. Hạn chế

* Fraud adversarial attack
* Concept drift
* Data leakage risk

---

# 7. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## Kết luận

Mô hình Hybrid giúp thu hẹp khoảng cách giữa:

* Accuracy
* Explainability
* Imbalance handling

## Future Work

* Graph Neural Networks
* Temporal modeling
* Federated learning
* AutoML credit scoring

---

# TÀI LIỆU THAM KHẢO (REFERENCES)

1. Altman, E. I. (1968). Financial ratios, discriminant analysis and prediction of corporate bankruptcy.

2. Chen, T., & Guestrin, C. (2016). XGBoost.

3. Ke, G. et al. (2017). LightGBM.

4. Arik, S. O., & Pfister, T. (2021). TabNet.

5. Lundberg, S. M., & Lee, S. I. (2017). SHAP.

```
```
