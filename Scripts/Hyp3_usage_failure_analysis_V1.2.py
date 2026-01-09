"""
Hypothesis 3: Usage Intensity and SSD Failure Likelihood

Author: Priya Pooja Hariharan
Script Version: 1.2
Project: MIS581 SSD-SMART-Reliability-Analysis

Hypothesis 3:
H0 (Null Hypothesis): Cumulative power-on hours and repeated power cycles are not
    significantly correlated with increased failure likelihood in SSDs.
H1 (Alternative Hypothesis): Cumulative power-on hours and repeated power cycles are
    positively correlated with increased failure likelihood in SSDs.

To evaluate Hypothesis 3, the analysis focused on whether long-term usage intensity—
operationalized as cumulative power-on hours and repeated power cycles—was associated
with increased SSD failure likelihood. First, descriptive statistics and group
comparisons were computed to summarize differences in power-on hours and power cycles
between failing and non-failing drives. Independent-samples ANOVA was applied to test
whether the mean values of these usage metrics differed significantly across failure
groups. Pearson correlation coefficients were then calculated to quantify the strength
and direction of linear relationships between usage metrics and the binary failure
label. To assess predictive value, logistic regression and Random Forest models were
trained using power-on hours, power cycles, and selected control features as predictors
of failure. Model performance was evaluated using ROC curves and area under the curve
(AUC) metrics, providing a measure of discriminative ability. Comparisons between
logistic regression and Random Forest models helped determine whether simple linear
effects were sufficient to capture failure risk or whether nonlinear relationships
offered additional predictive power. Together, these statistical tests and predictive
models provide a rigorous framework for testing the null hypothesis that usage
intensity is unrelated to failure likelihood against the alternative that higher
cumulative usage is associated with elevated risk.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, f_oneway

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


# ------------------------------------------------------------
# 0. OUTPUT DIRECTORIES
# ------------------------------------------------------------

RESULTS_DIR = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\hypothesis3_results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

@dataclass
class Hyp3Config:
    target_col: str = "failure_label"

    # NEW USAGE INTENSITY FEATURES
    usage_features: List[str] = field(default_factory=lambda: [
        "read_activity_rate",
        "busy_time_intensity",
        "write_intensity",
        "cpu_iops_intensity",
    ])

    extra_features: List[str] = field(default_factory=lambda: [
        "percentage_used",
        "wear_level_avg",
        "wear_level_max",
        "endurance_estimate_remaining",
    ])

    test_size: float = 0.3
    random_state: int = 42


# ------------------------------------------------------------
# 2. DATA LOADING + FEATURE ENGINEERING
# ------------------------------------------------------------

def load_dataset(path: str, cfg: Hyp3Config) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # If failure_label missing, derive automatically
    if cfg.target_col not in df.columns:
        print("[INFO] failure_label not found — deriving automatically...")
        df["failure_label"] = (
            (df["percentage_used"] >= 10) |
            (df["wear_level_max"] >= 10) |
            (df["endurance_estimate_remaining"] <= 50) |
            (df["media_errors"] > 20) |
            (df["bad_block_count_grown"] > 20) |
            (df["pcie_uncorrectable_errors"] > 0) |
            (df["unsafe_shutdowns"] > 1) |
            (df["composite_temperature_c"] > 40)
        ).astype(int)

    # ------------------------------------------------------------
    # NEW USAGE INTENSITY FEATURES
    # ------------------------------------------------------------

    # 1. Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 2. Read activity rate = Δ data_units_read / Δ time (seconds)
    if "timestamp" in df.columns and "data_units_read" in df.columns:
        time_diff = df["timestamp"].diff().dt.total_seconds().fillna(1)
        read_diff = df["data_units_read"].diff().fillna(0)
        df["read_activity_rate"] = read_diff / time_diff

    # 3. Busy-time intensity = controller_busy_time / power_on_hours
    if "controller_busy_time" in df.columns and "power_on_hours" in df.columns:
        df["busy_time_intensity"] = df["controller_busy_time"] / (df["power_on_hours"] + 1)

    # 4. Workload-weighted write intensity
    if "workload_type" in df.columns and "data_units_written" in df.columns:
        df["workload_type_num"] = df["workload_type"].astype("category").cat.codes
        df["write_intensity"] = df["data_units_written"] * (df["workload_type_num"] + 1)

    # 5. CPU–IOPS coupling
    if "cpu_name" in df.columns and "iops" in df.columns:
        df["cpu_name_num"] = df["cpu_name"].astype("category").cat.codes
        df["cpu_iops_intensity"] = df["cpu_name_num"] * df["iops"]

    # Convert to numeric
    for col in cfg.usage_features + cfg.extra_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.target_col] = df[cfg.target_col].astype(int)

    # Drop rows missing usage-intensity metrics
    df = df.dropna(subset=cfg.usage_features)

    return df


# ------------------------------------------------------------
# 3. DESCRIPTIVE STATS + ANOVA
# ------------------------------------------------------------

def descriptive_and_anova(df: pd.DataFrame, cfg: Hyp3Config) -> Dict[str, Dict[str, float]]:
    results = {}

    group_means = df.groupby(cfg.target_col)[cfg.usage_features].mean()
    group_means.to_csv(os.path.join(RESULTS_DIR, "usage_group_means_by_failure.csv"))

    for feat in cfg.usage_features:
        fail_group = df[df[cfg.target_col] == 1][feat].dropna()
        ok_group = df[df[cfg.target_col] == 0][feat].dropna()

        if len(fail_group) > 1 and len(ok_group) > 1:
            f_stat, p_val = f_oneway(fail_group, ok_group)
        else:
            f_stat, p_val = np.nan, np.nan

        results[feat] = {
            "mean_non_failure": float(ok_group.mean()) if len(ok_group) else np.nan,
            "mean_failure": float(fail_group.mean()) if len(fail_group) else np.nan,
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        }

    pd.DataFrame(results).T.to_csv(
        os.path.join(RESULTS_DIR, "anova_usage_vs_failure.csv")
    )

    return results


# ------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# ------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame, cfg: Hyp3Config) -> pd.DataFrame:
    rows = []
    for feat in cfg.usage_features:
        x = df[feat].dropna()
        y = df.loc[x.index, cfg.target_col]

        if y.nunique() > 1:
            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan

        rows.append({"feature": feat, "pearson_r": r, "p_value": p})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(os.path.join(RESULTS_DIR, "pearson_usage_vs_failure.csv"), index=False)
    return corr_df


# ------------------------------------------------------------
# 5. MODEL BUILDING
# ------------------------------------------------------------

def build_logistic_model(cfg: Hyp3Config):
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")


def build_rf_model(cfg: Hyp3Config):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1
    )


def prepare_features(df: pd.DataFrame, cfg: Hyp3Config):
    features = cfg.usage_features + [f for f in cfg.extra_features if f in df.columns]
    X = df[features].copy()
    y = df[cfg.target_col].astype(int)
    return X, y, features


# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

def evaluate_models(df: pd.DataFrame, cfg: Hyp3Config) -> pd.DataFrame:
    X, y, feature_list = prepare_features(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    numeric_cols = feature_list
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols)],
        remainder="drop"
    )

    models = {
        "logistic_regression": build_logistic_model(cfg),
        "random_forest": build_rf_model(cfg),
    }

    rows = []

    for name, base_model in models.items():
        clf = Pipeline([("pre", preprocessor), ("model", clone(base_model))])
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        rows.append({
            "model": name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_failure": report["1"]["precision"],
            "recall_failure": report["1"]["recall"],
            "f1_failure": report["1"]["f1-score"]
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_performance_summary_hyp3.csv"), index=False)

    return metrics_df


# ------------------------------------------------------------
# 7. VISUALIZATION
# ------------------------------------------------------------

def plot_usage_vs_failure(df: pd.DataFrame, cfg: Hyp3Config) -> None:
    for feat in cfg.usage_features:
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df, x=cfg.target_col, y=feat)
        plt.xticks([0, 1], ["Non-failure", "Failure"])
        plt.title(f"{feat} by Failure Status")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{feat}_by_failure.png"), dpi=300)
        plt.close()


# ------------------------------------------------------------
# 8. MAIN RUNNER
# ------------------------------------------------------------

def run_hypothesis3_analysis(data_path: str):
    cfg = Hyp3Config()

    print("\n=== LOADING DATASET FOR HYPOTHESIS 3 ===")
    df = load_dataset(data_path, cfg)
    print(f"Dataset loaded with {len(df)} rows.")

    print("\n=== DESCRIPTIVE STATISTICS AND ANOVA ===")
    anova_results = descriptive_and_anova(df, cfg)
    for feat, vals in anova_results.items():
        print(f"{feat}: F={vals['f_stat']:.3f}, p={vals['p_value']:.3g}")

    print("\n=== CORRELATION ANALYSIS ===")
    corr_df = correlation_analysis(df, cfg)
    print(corr_df.to_string(index=False))

    print("\n=== VISUALIZING USAGE VS FAILURE ===")
    plot_usage_vs_failure(df, cfg)
    print("Plots saved.")

    print("\n=== MODEL EVALUATION ===")
    metrics_df = evaluate_models(df, cfg)
    print(metrics_df.to_string(index=False))

    print("\n=== Hypothesis 3 analysis complete ===")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to:   {PLOTS_DIR}")


# ------------------------------------------------------------
# 9. MAIN EXECUTION ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    data_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"
    run_hypothesis3_analysis(data_path)
