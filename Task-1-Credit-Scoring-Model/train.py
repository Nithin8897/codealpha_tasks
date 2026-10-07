import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import joblib

# 1. Settings
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 2. Create synthetic credit dataset
n_samples = 2000

data = pd.DataFrame({
    "age": np.random.randint(21, 65, n_samples),

    "annual_income": np.random.randint(
        18000, 120000, n_samples
    ),

    "loan_amount": np.random.randint(
        2000, 50000, n_samples
    ),

    "credit_history_years": np.random.randint(
        1, 25, n_samples
    ),

    "number_of_loans": np.random.randint(
        0, 8, n_samples
    ),

    "late_payments": np.random.randint(
        0, 10, n_samples
    ),

    "debt_to_income": np.round(
        np.random.uniform(0.05, 0.80, n_samples), 2
    ),

    "savings": np.random.randint(
        500, 50000, n_samples
    )
})


# 3. Create credit risk score

risk_score = (
    0.00001 * data["loan_amount"]
    - 0.000008 * data["annual_income"]
    + 0.45 * data["late_payments"]
    + 2.0 * data["debt_to_income"]
    - 0.04 * data["credit_history_years"]
    - 0.00001 * data["savings"]
    + 0.15 * data["number_of_loans"]
    + np.random.normal(0, 0.8, n_samples)
)


# 4. Create target

# 1 = Good credit
# 0 = Bad credit

data["credit_score"] = risk_score

data["credit_risk"] = np.where(
    data["credit_score"] < 2.5,
    1,
    0
)

data.drop("credit_score", axis=1, inplace=True)


# 5. Display dataset information

print("Credit Scoring Dataset")
print("----------------------")

print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nCredit risk distribution:")
print(data["credit_risk"].value_counts())


# 6. Separate features and target


X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. Create Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# 9. Train model

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Training completed.")


# 10. Make predictions

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# 11. Evaluate model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


print("\nModel Performance")
print("-----------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Bad Credit", "Good Credit"]
))


# 12. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bad Credit", "Good Credit"]
)

disp.plot()

plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()


# 13. ROC Curve

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Credit Scoring - ROC Curve")
plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.close()


# 14. Feature importance

classifier = model.named_steps["classifier"]

coefficients = classifier.coef_[0]

feature_names = X.columns

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Importance": np.abs(coefficients)
})

importance = importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nFeature Importance:")
print(
    importance[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)


# 15. Feature importance plot

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Coefficient"]
)

plt.xlabel("Model Coefficient")
plt.ylabel("Feature")
plt.title("Credit Scoring Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)
plt.close()

# 16. Save predictions
prediction_results = X_test.copy()
prediction_results["Actual"] = y_test.values
prediction_results["Predicted"] = y_pred
prediction_results["Probability_Good_Credit"] = y_probability
prediction_results.to_csv(
    "results/predictions.csv",
    index=False
)
# 17. Save model
joblib.dump(
    model,
    "models/credit_scoring_model.joblib"
)
# 18. Save metrics
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})
metrics.to_csv(
    "results/metrics.csv",
    index=False
)
print("\nProject completed successfully!")
print("\nFiles created:")
print("results/confusion_matrix.png")
print("results/roc_curve.png")
print("results/feature_importance.png")
print("results/predictions.csv")
print("results/metrics.csv")
print("models/credit_scoring_model.joblib")
