# Heart Attack Prediction with SHAP

This project leverages **Machine Learning Models** (Logistic Regression, Random Forest, XGBoost) to predict heart attack risk with up to **89% accuracy**. Additionally, **SHAP (SHapley Additive exPlanations)** is used to explain feature importance both **globally** and at the **instance level**, enhancing the interpretability of the models through visualizations like **bar plots**, **force plots**, and **summary plots**.

---

## Goals of SHAP Visualizations
SHAP visualizations provide insights into the model's predictions and interpretability, addressing the following:

### 1. Contribution of Features to Predictions
- Each feature in the visualization represents a variable from the dataset.
- **Positive SHAP values**:
  - Represented in red or blue (pushing to the right).
  - Indicate the feature **increases** the prediction (e.g., predicting "1" in classification or higher values in regression).
- **Negative SHAP values**:
  - Pushing to the left.
  - Indicate the feature **decreases** the prediction (e.g., predicting "0" or lower values).

### 2. Relative Importance of Features
- **Force Plot**:
  - Features with greater influence push the prediction farther to the left or right.
- **Bar Plot**:
  - Longer bars indicate features with greater influence on the model's prediction.

### 3. Baseline Prediction vs. Final Prediction
- The **base value** represents the model's average prediction across all samples, serving as a starting point.
- Features "push" the prediction away from this baseline, either **increasing** or **decreasing** the final predicted value.
- **Final Prediction Calculation**:
  - The sum of SHAP values, plus the base value, equals the model's final prediction for the instance.

#### Example:
- **Original Instance**:
  - Features show how they combine to push the prediction toward the actual output.
- **Counterfactual Instance**:
  - Features show how the prediction changes when a feature is modified (e.g., changing "cp" from 3 to 2).

---

## Counterfactual Analysis
**Counterfactual Analysis** answers the question:

> "How would the prediction change if a specific feature were modified?"

### Example Scenario:
- **Original Prediction**:
  - Patient is at high risk (predicted class = 1).
- **Counterfactual Change**:
  - Modify "cp" (chest pain type) from 3 (asymptomatic) to 2 (non-anginal pain).
- **New Prediction**:
  - Lower risk of heart disease (predicted class = 0).

---

## Decision Transparency
SHAP visualizations enhance the transparency of **black-box models** (e.g., logistic regression, random forests, or neural networks). These benefits include:

### 1. Validate the Model
- Confirm whether features contributing to predictions align with **domain knowledge**.

### 2. Debug the Model
- Identify if **irrelevant features** are influencing predictions.

### 3. Build Trust
- Enable **non-technical stakeholders** to understand how and why the model makes its decisions.

---

## Conclusion
By combining **machine learning** with **SHAP-based explainability**, this project not only predicts heart attack risk effectively but also provides actionable insights for decision-makers. Visualizations such as **bar plots**, **force plots**, and **summary plots** bridge the gap between model accuracy and interpretability.
