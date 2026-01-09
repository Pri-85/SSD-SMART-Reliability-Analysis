"""
Hypothesis2: model_sensitivity_experiment.py

Author: Priya Pooja Hariharan
Script Version: 1.2
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 


1. Loads and cleans the SMART dataset
Reads the CSV file.
Ensures required columns exist.
Converts all metric columns to numeric.
Automatically creates a synthetic failed label if none exists.

2. Computes group‑level summaries
Aggregates IOPS, throttling events, error counts, and other metrics.
Groups results by nvme_capacity_tb and nand_type.
Saves the summary as a CSV file.

3. Computes correlation matrices
For each Drive Capacity and each nand type, it calculates correlations among all degradation metrics.
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
Bar charts of IOPS by Drive Capacity × nand type.
Boxplots of IOPS and throttling events.
Correlation heatmaps.
Model performance histograms.
All plots saved as PNG files in a plots/ folder.

6. Saves all outputs
Summary tables → results/summary_by_drivecap_nand.csv
Correlation matrices → results/corr_<drivecap>.csv
Model performance → results/model_performance.csv
All plots → plots/*.png

"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass, field
from typing import List, Dict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


# ============================================================
# 0. CONFIGURATION — INPUT + OUTPUT PATHS
# ============================================================

INPUT_DATA_PATH = (
    r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"
)

OUTPUT_BASE = (
    r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\hypothesis2_results"
)

OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_BASE, "Hyp2_plots")
OUTPUT_RESULTS_DIR = os.path.join(OUTPUT_BASE, "Hyp2_results")

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


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
    "bandwidth_write_gbps",
]

GROUP_COLS = ["nvme_capacity_tb", "nand_type"]
TARGET_COL = "failed"


# ============================================================
# 2. SAFE LOADING + AUTO FIXES
# ============================================================

def load_and_clean(path: str):
    df = pd.read_csv(path, low_memory=False)

    print("\n=== Loaded Columns ===")
    print(df.columns.tolist())

    # Ensure grouping columns exist
    for col in GROUP_COLS:
        if col not in df.columns:
            raise KeyError(f"Required grouping column '{col}' not found in dataset.")

    # ---------------------------------------------------------
    # Synthesize throttling_events if missing
    # ---------------------------------------------------------
    if "throttling_events" not in df.columns:
        if "composite_temperature_c" not in df.columns:
            raise KeyError(
                "throttling_events is missing and composite_temperature_c "
                "is not available to derive it."
            )
        print("[INFO] 'throttling_events' missing — deriving from workload + thermal stress logic.")

        df["throttling_events"] = (
        (df["queue_depth"] > 16) &
        (df["power_on_hours"] > 4000) &
        (df["workload_type"].str.lower().str.contains("random")) &
        (df["data_units_written"] > 100000000)
        ).astype(int)

#        print("[INFO] 'throttling_events' missing — deriving from composite_temperature_c > 40°C.")
#        df["throttling_events"] = (df["composite_temperature_c"] > 40).astype(int)

    # Identify available metrics
    available_metrics = [m for m in METRICS if m in df.columns]
    missing = [m for m in METRICS if m not in df.columns]

    if missing:
        print(f"[WARNING] Missing metric columns: {missing}")

    # Convert metrics to numeric
    for m in available_metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # ---------------------------------------------------------
    # Synthetic failure label (SAFE VERSION)
    # ---------------------------------------------------------
    if TARGET_COL not in df.columns:
        print("\n[INFO] No 'failed' column found. Creating synthetic failure label...")

        def safe_col(col):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                print(f"[WARNING] Missing '{col}' — using zeros.")
                return pd.Series([0] * len(df))

        media_err = safe_col("media_errors")
        pcie_unc = safe_col("pcie_uncorrectable_errors")
        pcie_c = safe_col("pcie_correctable_errors")
        throttling = safe_col("throttling_events")
        bw_w = safe_col("bandwidth_write_gbps")
        scrub = safe_col("background_scrub_time_pct")
        gc = safe_col("gc_active_time_pct")

        df[TARGET_COL] = (
            (media_err > media_err.quantile(0.50)) |
            (pcie_unc > 1) |
            (pcie_c > 40) |
            (bw_w > 5) |
            (throttling > throttling.quantile(0.50)) |
            (scrub > scrub.quantile(0.50)) |
            (gc > gc.quantile(0.50))
        ).astype(int)

    return df, available_metrics

# ============================================================
# 3. GROUPWISE SUMMARY
# ============================================================

def summarize_by_drivecap_nand(df, metrics):
    agg_dict = {m: ["mean", "std"] for m in metrics}
    agg_dict[TARGET_COL] = ["mean"]

    grouped = df.groupby(GROUP_COLS).agg(agg_dict)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns.to_flat_index()]
    return grouped.reset_index()


# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================

def compute_correlations(df, metrics, group_col):
    return {key: sub[metrics].corr() for key, sub in df.groupby(group_col)}


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

@dataclass
class ModelConfig:
    random_state: int = 200
    n_splits: int = 10
    model: object = field(default_factory=lambda: RandomForestClassifier(
        n_estimators=2000,
        max_depth=None,
        random_state=200,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))


def evaluate_model(df, metrics, target_col, cfg: ModelConfig):
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

        preprocessor = ColumnTransformer(
            [("num", StandardScaler(), metrics)],
            remainder="drop"
        )

        model = clone(cfg.model)
        clf = Pipeline([("pre", preprocessor), ("model", model)])

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]

        roc_scores.append(roc_auc_score(y_te, proba))
        pr_scores.append(average_precision_score(y_te, proba))

    return {"roc_auc": np.array(roc_scores), "pr_auc": np.array(pr_scores)}


# ============================================================
# 6. PLOTTING UTILITIES
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_group_summary(summary_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary_df,
        x="nvme_capacity_tb",
        y="iops_mean",
        hue="nand_type"
    )
    plt.title("Average IOPS by nvme_capacity_tb × NAND Type")
    plt.xticks(rotation=45)
    save_plot("group_summary_iops.png")


def plot_correlation_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title(title)
    save_plot(filename + ".png")


def plot_metric_distributions(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="nvme_capacity_tb", y="iops")
    plt.title("IOPS Distribution by Drive Capacity")
    plt.xticks(rotation=45)
    save_plot("iops_distribution.png")

    if "throttling_events" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="nvme_capacity_tb", y="throttling_events")
        plt.title("Thermal Throttling Events by Drive Capacity")
        plt.xticks(rotation=45)
        save_plot("throttling_distribution.png")


def plot_model_performance(results):
    plt.figure(figsize=(8, 5))
    sns.histplot(results["roc_auc"], kde=True)
    plt.title("ROC AUC Distribution Across CV Folds")
    save_plot("roc_auc_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(results["pr_auc"], kde=True)
    plt.title("PR AUC Distribution Across CV Folds")
    save_plot("pr_auc_distribution.png")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    df, available_metrics = load_and_clean(INPUT_DATA_PATH)

    print("\n=== SUMMARY BY Drive Capacity × NAND TYPE ===")
    summary_df = summarize_by_drivecap_nand(df, available_metrics)
    summary_df.to_csv(os.path.join(OUTPUT_RESULTS_DIR, "summary_by_drivecap_nand.csv"), index=False)
    print(summary_df.head())

    plot_group_summary(summary_df)

    print("\n=== CORRELATIONS BY Drive Capacity ===")
    corr_by_drivecap = compute_correlations(df, available_metrics, "nvme_capacity_tb")

    for drivecap, corr in corr_by_drivecap.items():
        corr.to_csv(os.path.join(OUTPUT_RESULTS_DIR, f"corr_{drivecap}.csv"))
        plot_correlation_heatmap(corr, f"Correlation Heatmap — {drivecap}", f"corr_heatmap_{drivecap}")

    print("\n=== MODEL PERFORMANCE ===")
    cfg = ModelConfig()
    results = evaluate_model(df, available_metrics, TARGET_COL, cfg)

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_RESULTS_DIR, "model_performance.csv"), index=False)

    print("ROC AUC mean:", results["roc_auc"].mean())
    print("PR AUC mean:", results["pr_auc"].mean())

    plot_model_performance(results)
    plot_metric_distributions(df)
