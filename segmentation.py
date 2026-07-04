"""
segmentation.py

Segments customers into behavioral clusters using k-means, then labels
each cluster in plain business terms based on its characteristics.

Output: data/customers_segmented.csv (original data + Cluster + ClusterLabel)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "data/Churn_Modelling.csv"
OUTPUT_PATH = "data/customers_segmented.csv"

FEATURES = ["Balance", "Tenure", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]
N_CLUSTERS = 4


def load_data():
    return pd.read_csv(DATA_PATH)


def run_clustering(df, n_clusters=N_CLUSTERS):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans


def label_clusters(df):
    """
    Look at each cluster's average characteristics and churn rate,
    then assign a human-readable label. Thresholds are relative to the
    overall dataset averages so this adapts if you swap in real data.
    """
    summary = df.groupby("Cluster")[FEATURES + ["Exited"]].mean()
    overall_balance = df["Balance"].mean()
    overall_tenure = df["Tenure"].mean()
    overall_churn = df["Exited"].mean()

    labels = {}
    for cluster_id, row in summary.iterrows():
        high_balance = row["Balance"] > overall_balance
        high_tenure = row["Tenure"] > overall_tenure
        low_activity = row["IsActiveMember"] < 0.5
        high_churn = row["Exited"] > overall_churn

        if high_balance and high_tenure and not high_churn:
            label = "High-Value Stable"
        elif high_balance and (low_activity or high_churn):
            label = "At-Risk High-Balance"
        elif not high_tenure and low_activity:
            label = "Low-Engagement New"
        elif low_activity or high_churn:
            label = "Dormant / Flight Risk"
        else:
            label = "Steady Mid-Value"
        labels[cluster_id] = label

    # Ensure uniqueness — if two clusters collide, disambiguate
    seen = {}
    for cid, lbl in labels.items():
        if lbl in seen.values():
            labels[cid] = f"{lbl} (B)"
        seen[cid] = labels[cid]

    df["ClusterLabel"] = df["Cluster"].map(labels)
    return df, summary, labels


def main():
    df = load_data()
    df, kmeans = run_clustering(df)
    df, summary, labels = label_clusters(df)

    print("=" * 60)
    print("CLUSTER SUMMARY (average feature values + churn rate)")
    print("=" * 60)
    summary["ClusterLabel"] = summary.index.map(labels)
    print(summary.round(2))

    print("\nCluster sizes:")
    print(df["ClusterLabel"].value_counts())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved segmented data -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
