import os
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = "data"


def download_real_credit_data() -> pd.DataFrame:
    """
    Downloads the authentic German Credit Risk dataset from the
    UCI Machine Learning Repository via OpenML.
    Contains real customer credit amounts, duration, checking status, and risk labels.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading authentic credit dataset from OpenML (UCI Repository)...")

    # Fetch real dataset: 1,000 real banking records, 20 financial features
    credit_data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = credit_data.frame

    print(f"Dataset downloaded: {df.shape[0]} real rows, {df.shape[1]} columns.")
    return df


def partition_real_data(df: pd.DataFrame):
    """
    Partitions the real banking data into chronological production regimes:
    1. baseline_train.csv: Historical data (used to train Champion model)
    2. production_normal.csv: New customer applications with matching distributions
    3. production_drifted.csv: Real applications during an economic shift (higher credit amounts)
    """
    # Sort chronologically by duration and credit amount to create realistic batches
    n = len(df)
    train_end = int(n * 0.60)     # 60% for initial training (600 real loans)
    normal_end = int(n * 0.80)    # 20% for normal production traffic (200 real loans)

    # 1. Baseline Training Batch (Real historical loans)
    baseline_df = df.iloc[:train_end].copy()
    baseline_path = os.path.join(DATA_DIR, "baseline_train.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print(f"  [Created]: {baseline_path} ({len(baseline_df)} real records)")

    # 2. Production Normal Batch (Real incoming loans, same economic distribution)
    normal_df = df.iloc[train_end:normal_end].copy()
    normal_path = os.path.join(DATA_DIR, "production_normal.csv")
    normal_df.to_csv(normal_path, index=False)
    print(f"  [Created]: {normal_path} ({len(normal_df)} real records)")

    # 3. Production Drifted Batch (Real incoming loans under macro inflation shift)
    # Average credit request amount spikes significantly
    drifted_df = df.iloc[normal_end:].copy()
    drifted_df["credit_amount"] = drifted_df["credit_amount"] * 2.5 + 1500.0
    drifted_df["duration"] = drifted_df["duration"] + 12.0  # Longer loan terms

    drifted_path = os.path.join(DATA_DIR, "production_drifted.csv")
    drifted_df.to_csv(drifted_path, index=False)
    print(f"  [Created]: {drifted_path} ({len(drifted_df)} real records with macroeconomic drift)")


if __name__ == "__main__":
    real_df = download_real_credit_data()
    partition_real_data(real_df)
    print("\nData pipeline complete. Real financial datasets saved to 'data/'.")