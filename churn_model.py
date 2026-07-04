"""
churn_model.py

Trains a churn-risk classifier (Random Forest, with Logistic Regression
as a comparison baseline), evaluates with precision/recall/ROC-AUC
(accuracy alone is misleading on imbalanced churn data), and extracts
feature importance so you can explain *why* someone is flagged as risky.

Output: data/customers_with_risk_scores.csv (segmented data + ChurnRiskScore)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score, recall_score, f1_score
)

DATA_PATH = "data/customers_segmented.csv"
OUTPUT_PATH = "data/customers_with_risk_scores.csv"

FEATURE_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
CATEGORICAL_COLS = ["Geography", "Gender"]
TARGET = "Exited"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_features(df):
    X = df[FEATURE_COLS].copy()
    # One-hot encode categoricals
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    y = df[TARGET]
    return X, y


def evaluate(name, y_test, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Precision (Churn): {precision_score(y_test, y_pred):.3f}")
    print(f"Recall (Churn):    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 (Churn):        {f1_score(y_test, y_pred):.3f}")


def main():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    lr_pred = log_reg.predict(X_test_scaled)
    lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, lr_pred, lr_proba)

    # Main model: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    evaluate("Random Forest", y_test, rf_pred, rf_proba)

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Feature Importance (Random Forest) ---")
    print(importance.round(3))

    # Score the full dataset (not just test set) so the dashboard has
    # a risk score for every customer
    full_proba = rf.predict_proba(X)[:, 1]
    df["ChurnRiskScore"] = full_proba
    df["ChurnRiskBand"] = pd.cut(
        df["ChurnRiskScore"], bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"], include_lowest=True
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved risk-scored data -> {OUTPUT_PATH}")
    print("\nRisk band distribution:")
    print(df["ChurnRiskBand"].value_counts())


if __name__ == "__main__":
    main()
