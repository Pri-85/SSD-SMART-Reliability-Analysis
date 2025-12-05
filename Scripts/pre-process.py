"""
preprocess.py
-------------
Preprocessing pipeline for MIS581 SSD dataset:
- Load raw SMART dataset (CSV)
- Clean missing values and duplicates
- Normalize units (hours → days, sectors → GB)
- Scale selected features
- Save processed dataset
"""

import pandas as pd
import numpy as np
import os

# ---------- Load ----------
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw SSD SMART dataset from CSV."""
    return pd.read_csv(filepath)

# ---------- Cleaning ----------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")  # forward fill missing values
    return df

# ---------- Unit Conversion ----------
def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SMART attributes to consistent units."""
    if "Power_On_Hours" in df.columns:
        df["Power_On_Days"] = df["Power_On_Hours"] / 24
    if "Data_Units_Written" in df.columns:
        # Convert 512-byte sectors to GB
        df["Data_Units_GB"] = df["Data_Units_Written"] * 512 / (1024**3)
    if "Data_Units_Read" in df.columns:
        df["Data_Read_GB"] = df["Data_Units_Read"] * 512 / (1024**3)
    return df

# ---------- Feature Scaling ----------
def scale_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Normalize numeric features to 0–1 range."""
    for col in cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

# ---------- Save ----------
def save_processed(df: pd.DataFrame, filename: str):
    """Save processed dataset to /data/processed folder."""
    out_path = os.path.join("data", "processed", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed file saved to {out_path}")

# ---------- Main ----------
if __name__ == "__main__":
    raw_file = os.path.join("data", "raw", "MIS581_SSD_dataset.csv")
    df = load_data(raw_file)
    df = clean_data(df)
    df = normalize_units(df)
    df = scale_features(df, ["Power_On_Days", "Data_Units_GB", "Data_Read_GB"])
    save_processed(df, "MIS581_SSD_processed.csv")