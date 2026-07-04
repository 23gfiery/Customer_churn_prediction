"""
generate_synthetic_data.py

Creates a synthetic bank-customer dataset that mirrors the schema of the
popular Kaggle "Bank Customer Churn Prediction" dataset. This lets you run
the whole pipeline end-to-end right away.

Once you download the real Kaggle CSV, just drop it in data/Churn_Modelling.csv
with the same column names and everything downstream keeps working unchanged.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

def generate():
    customer_id = np.arange(1, N + 1)

    geography = np.random.choice(["France", "Germany", "Spain"], size=N, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(["Male", "Female"], size=N)
    age = np.clip(np.random.normal(38, 10, N), 18, 85).astype(int)
    tenure = np.random.randint(0, 11, N)  # years with the bank
    credit_score = np.clip(np.random.normal(650, 90, N), 350, 850).astype(int)
    num_products = np.random.choice([1, 2, 3, 4], size=N, p=[0.5, 0.35, 0.1, 0.05])
    has_cr_card = np.random.choice([0, 1], size=N, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
    estimated_salary = np.round(np.random.uniform(1000, 200000, N), 2)

    # Balance: many customers with 0 balance, others with a real balance
    has_balance = np.random.rand(N) > 0.35
    balance = np.where(
        has_balance,
        np.clip(np.random.normal(90000, 45000, N), 0, None),
        0.0
    )
    balance = np.round(balance, 2)

    # Build churn probability from realistic-ish rules so the patterns
    # are actually discoverable during EDA/modeling (this mirrors the
    # well-known pattern in the real dataset: 3-4 products + inactive +
    # older age + Germany geography => much higher churn risk)
    churn_prob = (
        0.05
        + 0.25 * (num_products >= 3)
        + 0.15 * (is_active_member == 0)
        + 0.10 * (geography == "Germany")
        + 0.15 * (age > 50)
        + 0.10 * (balance > 150000)
        - 0.10 * (tenure > 6)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.9)
    exited = (np.random.rand(N) < churn_prob).astype(int)

    df = pd.DataFrame({
        "CustomerId": customer_id,
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Exited": exited,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/Churn_Modelling.csv", index=False)
    print(f"Generated {len(df)} rows -> data/Churn_Modelling.csv")
    print(df["Exited"].value_counts(normalize=True))
