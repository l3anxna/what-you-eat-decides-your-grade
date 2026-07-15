## 1. Data Context

The objective of this study is to predict **GPA** from a food and lifestyle survey containing **125 observations** and **61 features**. This is a challenging prediction task due to several limitations in the dataset:

- The dataset is relatively small (125 samples), making it difficult for machine learning models to generalize.
- Both **`GPA`** and **`weight`** required string-to-numeric conversion before modeling.
- Several features contain substantial missing values, with **13–26 missing entries** out of 125 observations.
- The combination of limited sample size, missing data, and many predictor variables increases the risk of overfitting and unstable feature importance estimates.

---

## 2. SHAP-Based Feature Selection

Random Forest (RF) and Gradient Boosting (GB) models were initially trained using **all 60 numeric features** after median imputation with an **80/20 train-test split**.

The baseline predictive performance was already weak:

| Model | MAE | R² |
|-------|----:|---:|
| Random Forest | 0.324 | 0.114 |
| Gradient Boosting | 0.362 | -0.118 |

The SHAP feature rankings produced by the two models showed substantial disagreement. For example:

- **`father_education`** and **`Thai_food`** ranked highly in the Random Forest model but appeared near the bottom of the Gradient Boosting ranking.
- **`weight`**, **`exercise`**, and **`Greek_food`** exhibited relatively high SHAP importance in Gradient Boosting but almost no importance in Random Forest.

This disagreement is expected because both models exhibit little predictive power. When models are trained on approximately 100 training samples with weak signal, SHAP explains the patterns learned by each individual model—including noise—rather than identifying stable underlying relationships.

To reduce model-specific bias, an **agreement filter** was applied by selecting features with a **mean absolute SHAP value greater than 0.01 in both models**, resulting in **12 retained features**.

However, no stability analysis (such as repeated cross-validation or bootstrap resampling) was performed to verify whether these 12 features remain important across different train-test splits.

---

## 3. Model Comparison Using the Reduced Feature Set

### Single 80/20 Train-Test Split

Four regression models were trained using the reduced feature set.

**Note:** All tree-based models were configured with **`n_estimators = 10`**, which is considerably smaller than typical values (100–500 trees). Such a small ensemble is likely undertrained and may not capture complex relationships in the data.

| Model | MAE | R² |
|-------|----:|---:|
| Linear Regression | 0.326 | 0.183 |
| Gradient Boosting | 0.342 | 0.003 |
| Random Forest | 0.364 | -0.160 |
| XGBoost | 0.411 | -0.319 |

Linear Regression achieved the highest R² on this particular split, although the value (0.183) remains relatively low.

---

### Five-Fold Cross-Validation

To obtain a more reliable estimate of model performance for this small dataset, five-fold cross-validation was performed.

| Model | MAE (mean ± std) | R² (mean ± std) |
|-------|-----------------:|----------------:|
| Linear Regression | 0.302 ± 0.014 | -0.051 ± 0.230 |
| Gradient Boosting | 0.305 ± 0.024 | -0.022 ± 0.141 |
| Random Forest | 0.331 ± 0.019 | -0.267 ± 0.519 |
| XGBoost | 0.337 ± 0.041 | -0.320 ± 0.424 |

The cross-validation results show that:

- MAE is relatively stable across folds for all models.
- Mean R² values are approximately zero or negative.
- R² exhibits large standard deviations, indicating that performance changes substantially depending on the train-test split.

These findings suggest that the dataset contains relatively weak predictive signal compared with the level of noise. The small sample size also contributes to unstable performance estimates.

---

## 4. Anomaly Detection

Three anomaly detection approaches were evaluated.

### Residual-Based Detection

Residual analysis detected **no anomalous observations**. This outcome is not surprising because the regression models themselves have limited predictive ability, making unusually large residuals difficult to distinguish from ordinary prediction error.

### Isolation Forest and Local Outlier Factor (LOF)

Isolation Forest and Local Outlier Factor (LOF) were both configured with a contamination rate of **0.10**, causing each algorithm to classify approximately **12 of the 120 observations** as anomalies.

However, only **two observations** were identified as anomalies by both methods.

The limited overlap indicates that the detected anomalies depend strongly on the algorithm rather than representing a clearly separable anomalous subgroup. This suggests that the dataset does not contain well-defined outliers, and the anomaly structure is weak.