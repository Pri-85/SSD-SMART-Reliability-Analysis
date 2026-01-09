"""
Hypothesis2: model_sensitivity_experiment.py

Author: Priya Pooja Hariharan
Script Version: 1.1
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 


1. Loads and cleans the SMART dataset
Reads the CSV file.
Ensures required columns exist.
Converts all metric columns to numeric.
Automatically creates a synthetic failed label if none exists.

2. Computes group‑level summaries
Aggregates IOPS, throttling events, error counts, and other metrics.
Groups results by drive_manufacturer and NAND_type.
Saves the summary as a CSV file.

3. Computes correlation matrices
For each manufacturer and each NAND type, it calculates correlations among all degradation metrics.
Saves each correlation matrix as a CSV.
Generates heatmap PNGs for visual interpretation.

4. Trains a predictive model
Uses a Random Forest classifier.
Predicts the failed label using all SMART metrics.
Performs 5‑fold stratified cross‑validation.
Computes ROC AUC and PR AUC.
Saves model performance results as CSV.
Saves performance distribution plots as PNG.

5. Generates visualizations
Bar charts of IOPS by manufacturer × NAND type.
Boxplots of IOPS and throttling events.
Correlation heatmaps.
Model performance histograms.
All plots saved as PNG files in a plots/ folder.

6. Saves all outputs
Summary tables → results/summary_by_vendor_nand.csv
Correlation matrices → results/corr_<vendor>.csv
Model performance → results/model_performance.csv
All plots → plots/*.png

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass, field
from typing import List, Dict, Callable

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

# ============================================================
# 0. OUTPUT FOLDERS
# ============================================================

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# 1. METRIC DEFINITIONS
# ============================================================

METRICS = [
    "iops",
    "throttling_events",
    "unsafe_shutdowns",
    "background_scrub_time_pct",
    "gc_active_time_pct",
    "media_errors",
    "error_information_log_entries",
    "bad_block_count_grown",
    "pcie_correctable_errors",
    "pcie_uncorrectable_errors",
]

GROUP_COLS = ["drive_manufacturer", "NAND_type"]
TARGET_COL = "failed"   # will be created if missing


# ============================================================
# 2. SAFE LOADING + AUTO FIXES
# ============================================================

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    print("\n=== Loaded Columns ===")
    print(df.columns.tolist())

    # Ensure grouping columns exist
    for col in GROUP_COLS:
        if col not in df.columns:
            raise KeyError(f"Required grouping column '{col}' not found in dataset.")

    # Ensure all METRICS exist (skip missing ones)
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        print(f"\n[WARNING] Missing metric columns: {missing}")
        for m in missing:
            METRICS.remove(m)

    # Convert metrics to numeric safely
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Create failure label if missing
    if TARGET_COL not in df.columns:
        print("\n[INFO] No 'failed' column found. Creating synthetic failure label...")
        df[TARGET_COL] = (
            (df["media_errors"].fillna(0) > df["media_errors"].quantile(0.95)) |
            (df["pcie_uncorrectable_errors"].fillna(0) > 0) |
            (df["throttling_events"].fillna(0) > df["throttling_events"].quantile(0.90))
        ).astype(int)

    return df


# ============================================================
# 3. GROUPWISE SUMMARY
# ============================================================

def summarize_by_vendor_nand(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {m: ["mean", "std"] for m in METRICS}
    agg_dict[TARGET_COL] = ["mean"]  # failure rate

    grouped = df.groupby(GROUP_COLS).agg(agg_dict)

    # Flatten MultiIndex columns
    grouped.columns = [
        f"{metric}_{stat}" for metric, stat in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    return grouped


# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================

def compute_correlations(df: pd.DataFrame, group_col: str) -> Dict[str, pd.DataFrame]:
    corrs = {}
    for key, sub in df.groupby(group_col):
        corrs[key] = sub[METRICS].corr()
    return corrs


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

@dataclass
class ModelConfig:
    random_state: int = 42
    n_splits: int = 5
    model: object = field(default_factory=lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))


def evaluate_model(df: pd.DataFrame, metrics: list, target_col: str, cfg: ModelConfig):
    X = df[metrics].copy()
    y = df[target_col].astype(int)

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state
    )

    roc_scores, pr_scores = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        numeric_cols = X.columns.tolist()
        preprocessor = ColumnTransformer(
            [("num", StandardScaler(), numeric_cols)],
            remainder="drop"
        )

        model = clone(cfg.model)
        clf = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]

        roc_scores.append(roc_auc_score(y_te, proba))
        pr_scores.append(average_precision_score(y_te, proba))

    return {
        "roc_auc": np.array(roc_scores),
        "pr_auc": np.array(pr_scores),
    }


# ============================================================
# 6. PLOTTING UTILITIES (SAVE PNGs)
# ============================================================

def plot_group_summary(summary_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary_df,
        x="drive_manufacturer",
        y="iops_mean",
        hue="NAND_type"
    )
    plt.title("Average IOPS by Manufacturer × NAND Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/group_summary_iops.png")
    plt.close()


def plot_correlation_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def plot_metric_distributions(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="drive_manufacturer", y="iops")
    plt.title("IOPS Distribution by Manufacturer")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/iops_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="drive_manufacturer", y="throttling_events")
    plt.title("Thermal Throttling Events by Manufacturer")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/throttling_distribution.png")
    plt.close()


def plot_model_performance(results):
    plt.figure(figsize=(8, 5))
    sns.histplot(results["roc_auc"], kde=True)
    plt.title("ROC AUC Distribution Across CV Folds")
    plt.tight_layout()
    plt.savefig("plots/roc_auc_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(results["pr_auc"], kde=True)
    plt.title("PR AUC Distribution Across CV Folds")
    plt.tight_layout()
    plt.savefig("plots/pr_auc_distribution.png")
    plt.close()


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    df = load_and_clean(
        r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"
    )

    print("\n=== SUMMARY BY MANUFACTURER × NAND TYPE ===")
    summary_df = summarize_by_vendor_nand(df)
    summary_df.to_csv("results/summary_by_vendor_nand.csv", index=False)
    print(summary_df.head())

    plot_group_summary(summary_df)

    print("\n=== CORRELATIONS BY MANUFACTURER ===")
    corr_by_vendor = compute_correlations(df, "drive_manufacturer")

    for vendor, corr in corr_by_vendor.items():
        print(f"\nVendor: {vendor}")
        corr.to_csv(f"results/corr_{vendor}.csv")
        plot_correlation_heatmap(corr, f"Correlation Heatmap — {vendor}", f"corr_heatmap_{vendor}")

    print("\n=== MODEL PERFORMANCE (REAL DATA) ===")
    cfg = ModelConfig()
    results = evaluate_model(df, METRICS, TARGET_COL, cfg)

    pd.DataFrame({
        "roc_auc": results["roc_auc"],
        "pr_auc": results["pr_auc"]
    }).to_csv("results/model_performance.csv", index=False)

    print("ROC AUC mean:", results["roc_auc"].mean())
    print("PR AUC mean:", results["pr_auc"].mean())

    plot_model_performance(results)
    plot_metric_distributions(df)
