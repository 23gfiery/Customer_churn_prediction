"""
eda.py

Exploratory Data Analysis for the bank customer churn dataset.
Run this after generate_synthetic_data.py (or after dropping in the real
Kaggle CSV at data/Churn_Modelling.csv).

Produces:
- Printed summary stats and churn-rate breakdowns
- PNG charts saved to outputs/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — just saves files, no GUI/Tk needed
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/Churn_Modelling.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def basic_overview(df):
    print("=" * 60)
    print("BASIC OVERVIEW")
    print("=" * 60)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nOverall churn rate: {:.2%}".format(df["Exited"].mean()))


def churn_by_segment(df, column):
    rate = df.groupby(column)["Exited"].mean().sort_values(ascending=False)
    print(f"\nChurn rate by {column}:")
    print(rate.apply(lambda x: f"{x:.2%}"))
    return rate


def plot_churn_by_categorical(df, column, filename):
    plt.figure(figsize=(6, 4))
    rate = df.groupby(column)["Exited"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="viridis", legend=False)
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {column}")
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}")
    plt.close()
    print(f"Saved outputs/{filename}")


def plot_churn_by_numeric_bucket(df, column, bins, labels, filename):
    plt.figure(figsize=(6, 4))
    df["_bucket"] = pd.cut(df[column], bins=bins, labels=labels)
    rate = df.groupby("_bucket", observed=True)["Exited"].mean()
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="magma", legend=False)
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {column} Bucket")
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}")
    plt.close()
    df.drop(columns="_bucket", inplace=True)
    print(f"Saved outputs/{filename}")


def main():
    df = load_data()
    basic_overview(df)

    churn_by_segment(df, "Geography")
    churn_by_segment(df, "NumOfProducts")
    churn_by_segment(df, "IsActiveMember")
    churn_by_segment(df, "Gender")

    plot_churn_by_categorical(df, "Geography", "churn_by_geography.png")
    plot_churn_by_categorical(df, "NumOfProducts", "churn_by_num_products.png")
    plot_churn_by_categorical(df, "IsActiveMember", "churn_by_active_member.png")

    plot_churn_by_numeric_bucket(
        df, "Age",
        bins=[18, 30, 40, 50, 60, 85],
        labels=["18-30", "31-40", "41-50", "51-60", "61+"],
        filename="churn_by_age_bucket.png",
    )
    plot_churn_by_numeric_bucket(
        df, "Tenure",
        bins=[-1, 2, 5, 8, 11],
        labels=["0-2 yrs", "3-5 yrs", "6-8 yrs", "9-10 yrs"],
        filename="churn_by_tenure_bucket.png",
    )
    plot_churn_by_numeric_bucket(
        df, "Balance",
        bins=[-1, 1, 50000, 100000, 150000, 300000],
        labels=["Zero", "Low", "Mid", "High", "Very High"],
        filename="churn_by_balance_bucket.png",
    )

    print("\n" + "=" * 60)
    print("EDA complete. Charts saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
