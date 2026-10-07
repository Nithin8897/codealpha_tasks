## Tasks-1-Credit Scoring Model

## 1.Project Overview

This project builds a simple Credit Scoring Model using Machine
Learning. A Logistic Regression algorithm is used to predict whether a
customer has Good Credit or Bad Credit.

The project uses a synthetically generated dataset, so no real customer
information is used.

## 2.Technologies Used

• Python
• NumPy
• Pandas
• Matplotlib
• Scikit-learn
• Joblib

## 3.Features

The model uses:

• Age
• Annual Income
• Loan Amount
• Credit History Years
• Number of Loans
• Late Payments
• Debt-to-Income Ratio
• Savings

## 4.Machine Learning Method

1. Generate synthetic credit data.
2. Create Good Credit / Bad Credit labels.
3. Split the data into training and testing sets.
4. Scale the features using StandardScaler.
5. Train a Logistic Regression model.
6. Make predictions on test data.
7. Evaluate the model using Accuracy, Precision, Recall, F1 Score, and
ROC-AUC.
8. Create a confusion matrix, ROC curve, and feature-importance plot.
9. Save the trained model.

## 5.Project Structure

```test
CodeAlpha_CreditScoringModel/
|
|-- train.py
|-- requirements.txt
|-- README.md
|
|-- results/
|   |-- confusion_matrix.png
|   |-- roc_curve.png
|   |-- feature_importance.png
|   |-- predictions.csv
|   `-- metrics.csv
|
`-- models/
    `-- credit_scoring_model.joblib
```


## 6.How to Run

Install the required libraries:
```bash
pip install -r requirements.txt
```

Run the project:
```bash
python train.py
```

## 7.Output

The program creates:

• confusion_matrix.png - model prediction matrix.
• roc_curve.png - ROC curve.
• feature_importance.png - feature coefficients.
• predictions.csv - test predictions.
• metrics.csv - evaluation scores.
• credit_scoring_model.joblib - saved trained model.

## 8.Dataset

This project uses synthetic credit data generated with NumPy and
Pandas. It is intended for educational and machine-learning
demonstration purposes.



## 9.Internship Task

CodeAlpha Machine Learning Internship

Task 1 - Credit Scoring Model

