# Mathematical Mapping in SHAP for Feature Contribution

SHAP (SHapley Additive exPlanations) values are based on Shapley values from game theory, which distribute the credit of a model's prediction among its input features based on their contributions. This document explains how SHAP values mathematically map feature contributions, particularly in models like logistic regression.

---

## 1. SHAP Value Calculation

### (a) Model Prediction Breakdown

For a given instance $x$:

$$
\hat{y}(x) = f(x) = \phi_0 + \sum_{i=1}^M \phi_i
$$

Where:
- $\hat{y}(x)$: The model's predicted value for the instance $x$.
- $f(x)$: The prediction model (e.g., logistic regression, random forest, etc.).
- $\phi_0$: The base value or expected prediction of the model over the dataset.
- $\phi_i$: SHAP value for feature $i$, indicating its contribution to the deviation from $\phi_0$.
- $M$: Total number of features.

SHAP computes marginal contributions for each feature by observing how the prediction changes when the feature is included versus excluded.

---

### (b) Sign of SHAP Values and Colors

The sign of $\phi_i$ determines its visual representation:
- **Red (positive $\phi_i$):** The feature increases the predicted output.
- **Blue (negative $\phi_i$):** The feature decreases the predicted output.

For logistic regression:
- The model outputs probabilities as $\hat{y}(x) = \sigma(z)$, where $\sigma(z)$ is the sigmoid function, and:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_Mx_M + b
$$

- SHAP colors are derived based on the weighted input $w_ix_i$:
  - $w_ix_i > 0$: Positive contribution (red).
  - $w_ix_i < 0$: Negative contribution (blue).

---

### (c) Logistic Regression Specifics

Logistic regression predicts probabilities using the sigmoid function:

$$
\hat{y}(x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \sum_{i=1}^M w_ix_i + b
$$

#### Mapping Feature Importance
- **Feature Weight $w_i$:** Reflects the importance of feature $i$ for the prediction.
  - Larger $|w_i|$ indicates a more significant contribution.
- **Feature Contribution to $z$:**
  - $w_ix_i$ determines the specific contribution of feature $i$ to the logit $z$.

#### SHAP Values from Contributions
SHAP calculates marginal contributions using the Shapley value formula:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[f(S \cup \{i\}) - f(S)\right]
$$

Where:
- $S$: Subset of all features excluding $i$.
- $f(S)$: Model output when only features in subset $S$ are used.

#### Colors and Relevance
- The magnitude of $\phi_i$ reflects how $w_ix_i$ influences $\hat{y}(x)$:
  - **Red (positive):** $w_ix_i$ increases $\hat{y}(x)$.
  - **Blue (negative):** $w_ix_i$ decreases $\hat{y}(x)$.

---

## 2. Activation Values and Neural Networks

For complex models like neural networks, SHAP approximates feature contributions using:
- A reference dataset.
- Gradients of the output with respect to the input features.

### (a) Colors and Activation Functions

SHAP captures how feature contributions interact with activation functions like ReLU, sigmoid, or tanh. For example, in logistic regression:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_Mx_M + b
$$

The sigmoid function is sensitive to changes in $z$:

$$
\frac{\partial \sigma}{\partial z} = \sigma(z)(1 - \sigma(z))
$$

#### Key Insights:
- Features $w_ix_i$ near the steep region of the sigmoid curve will have larger SHAP values.

---

## 3. Interpreting Relevance and Colors

### (a) Connecting SHAP to Models
- SHAP maps feature contributions to the output via the underlying mathematical structure.
- Colors signify:
  - **Direction:** Positive or negative contribution of the feature.
  - **Strength:** Magnitude of the contribution.

### (b) Logistic Regression Insights
- In logistic regression:
  - Colors directly relate to the signs and magnitudes of $w_ix_i$.
  - These collectively impact $z$, the logit value.

---

## Summary: How Colors Are Derived
- **Red:** Features that increase the prediction ($+\phi_i$).
- **Blue:** Features that decrease the prediction ($-\phi_i$).
- **Magnitude:** Reflects the importance or contribution of the feature.

SHAP provides an interpretable mapping of feature relevance in predictive models, offering insights into how each feature influences the output and how these contributions are represented visually.
