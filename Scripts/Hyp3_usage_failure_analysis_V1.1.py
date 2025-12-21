"""
Hypothesis 3: Usage Intensity and SSD Failure Likelihood

Author: Priya Pooja Hariharan
Script Version: 1.0
Project: MIS581 SSD-SMART-Reliability-Analysis

Hypothesis 3:
H₀ (Null Hypothesis): Cumulative power-on hours and repeated power cycles are not
    significantly correlated with increased failure likelihood in SSDs.
H₁ (Alternative Hypothesis): Cumulative power-on hours and repeated power cycles are
    positively correlated with increased failure likelihood in SSDs.

APA-style methodology paragraph (for your paper):

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_hyp3")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_hyp3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

@dataclass
class Hyp3Config:
    target_col: str = "failure_label"
    usage_features: List[str] = field(default_factory=lambda: [
        "power_on_hours",
        "power_cycles",
    ])
    # Optional additional covariates if present in your dataset
    extra_features: List[str] = field(default_factory=lambda: [
        "percentage_used",
        "wear_level_avg",
        "wear_level_max",
        "endurance_estimate_remaining",
    ])
    test_size: float = 0.3
    random_state: int = 42


# ------------------------------------------------------------
# 2. DATA LOADING AND BASIC VALIDATION
# ------------------------------------------------------------

def load_dataset(path: str, cfg: Hyp3Config) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # If failure_label is missing, derive it automatically
    if cfg.target_col not in df.columns:
        print("[INFO] failure_label not found — deriving automatically...")
        df["failure_label"] = (
            (df["percentage_used"] >= 90) |
            (df["wear_level_max"] >= 95) |
            (df["endurance_estimate_remaining"] <= 10) |
            (df["media_errors"] > 0) |
            (df["bad_block_count_grown"] > 0) |
            (df["pcie_uncorrectable_errors"] > 0) |
            (df["unsafe_shutdowns"] > 0) |
            (df["throttling_events"] > 0)
        ).astype(int)

    # Convert usage metrics to numeric
    for col in cfg.usage_features + cfg.extra_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.target_col] = df[cfg.target_col].astype(int)

    df = df.dropna(subset=cfg.usage_features)

    return df

# ------------------------------------------------------------
# 3. DESCRIPTIVE STATS AND ANOVA
# ------------------------------------------------------------

def descriptive_and_anova(df: pd.DataFrame, cfg: Hyp3Config) -> Dict[str, Dict[str, float]]:
    """
    Compute descriptive stats and simple one-way ANOVA comparing usage metrics
    between failure and non-failure groups.
    """
    results = {}

    # Group-wise means
    group_means = df.groupby(cfg.target_col)[cfg.usage_features].mean()
    group_means.to_csv(os.path.join(RESULTS_DIR, "usage_group_means_by_failure.csv"))

    # ANOVA: for each usage metric, compare distributions between groups
    for feat in cfg.usage_features:
        fail_group = df[df[cfg.target_col] == 1][feat].dropna()
        ok_group = df[df[cfg.target_col] == 0][feat].dropna()

        if len(fail_group) > 1 and len(ok_group) > 1:
            f_stat, p_val = f_oneway(fail_group, ok_group)
        else:
            f_stat, p_val = np.nan, np.nan

        results[feat] = {
            "mean_non_failure": float(ok_group.mean()) if len(ok_group) > 0 else np.nan,
            "mean_failure": float(fail_group.mean()) if len(fail_group) > 0 else np.nan,
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        }

    anova_df = pd.DataFrame(results).T
    anova_df.to_csv(os.path.join(RESULTS_DIR, "anova_usage_vs_failure.csv"))

    return results


# ------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# ------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame, cfg: Hyp3Config) -> pd.DataFrame:
    """
    Compute Pearson correlations between usage metrics and failure label.
    """
    rows = []
    for feat in cfg.usage_features:
        x = df[feat].dropna()
        y = df.loc[x.index, cfg.target_col]
        if y.nunique() > 1:
            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan
        rows.append({
            "feature": feat,
            "pearson_r": r,
            "p_value": p,
        })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(os.path.join(RESULTS_DIR, "pearson_usage_vs_failure.csv"), index=False)
    return corr_df


# ------------------------------------------------------------
# 5. MODEL BUILDING UTILITIES
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
    """
    Build feature matrix X and target y from usage + optional extra features.
    Only include extra features that actually exist.
    """
    features = cfg.usage_features + [f for f in cfg.extra_features if f in df.columns]
    X = df[features].copy()
    y = df[cfg.target_col].astype(int)
    return X, y, features


# ------------------------------------------------------------
# 6. TRAIN/TEST SPLIT AND MODEL EVALUATION
# ------------------------------------------------------------

def evaluate_models(df: pd.DataFrame, cfg: Hyp3Config) -> pd.DataFrame:
    """
    Train and evaluate Logistic Regression and Random Forest models.
    Save metrics and ROC curve data.
    """
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
        clf = Pipeline([
            ("pre", preprocessor),
            ("model", clone(base_model))
        ])

        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save ROC curve points
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        })
        roc_df.to_csv(
            os.path.join(RESULTS_DIR, f"roc_curve_{name}.csv"),
            index=False
        )

        # For logistic regression: save coefficients
        if name == "logistic_regression":
            log_reg = clf.named_steps["model"]
            coef = log_reg.coef_[0]
            coef_df = pd.DataFrame({
                "feature": numeric_cols,
                "coefficient": coef
            })
            coef_df.to_csv(
                os.path.join(RESULTS_DIR, "logistic_regression_coefficients.csv"),
                index=False
            )

        # For random forest: save feature importances
        if name == "random_forest":
            rf = clf.named_steps["model"]
            imp = rf.feature_importances_
            imp_df = pd.DataFrame({
                "feature": numeric_cols,
                "importance": imp
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(
                os.path.join(RESULTS_DIR, "random_forest_feature_importance.csv"),
                index=False
            )

        # Save classification report per model
        report_df = pd.DataFrame(report).T
        report_df.to_csv(
            os.path.join(RESULTS_DIR, f"classification_report_{name}.csv")
        )

        rows.append({
            "model": name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_failure": report["1"]["precision"],
            "recall_failure": report["1"]["recall"],
            "f1_failure": report["1"]["f1-score"]
        })

        # Plot ROC curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"roc_curve_{name}.png"), dpi=300)
        plt.close()

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "model_performance_summary_hyp3.csv"),
        index=False
    )

    return metrics_df


# ------------------------------------------------------------
# 7. VISUALIZATION OF USAGE VS FAILURE
# ------------------------------------------------------------

def plot_usage_vs_failure(df: pd.DataFrame, cfg: Hyp3Config) -> None:
    """
    Generate visualizations for usage metrics by failure group.
    """
    # Boxplots for power_on_hours and power_cycles by failure_label
    for feat in cfg.usage_features:
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df, x=cfg.target_col, y=feat)
        plt.xticks([0, 1], ["Non-failure", "Failure"])
        plt.title(f"{feat} by Failure Status")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{feat}_by_failure.png"), dpi=300)
        plt.close()

    # Scatter: power_on_hours vs power_cycles colored by failure
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="power_on_hours",
        y="power_cycles",
        hue=cfg.target_col,
        alpha=0.5,
        palette="Set1"
    )
    plt.title("Power-on Hours vs Power Cycles by Failure Status")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scatter_power_on_vs_cycles_by_failure.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# 8. MAIN RUNNER FOR HYPOTHESIS 3
# ------------------------------------------------------------

def run_hypothesis3_analysis(data_path: str):
    cfg = Hyp3Config()

    print("\n=== LOADING DATASET FOR HYPOTHESIS 3 ===")
    df = load_dataset(data_path, cfg)
    print(f"Dataset loaded with {len(df)} rows.")

    print("\n=== DESCRIPTIVE STATISTICS AND ANOVA ===")
    anova_results = descriptive_and_anova(df, cfg)
    print("ANOVA results (saved to CSV):")
    for feat, vals in anova_results.items():
        print(f"{feat}: F={vals['f_stat']:.3f}, p={vals['p_value']:.3g}")

    print("\n=== CORRELATION ANALYSIS (PEARSON) ===")
    corr_df = correlation_analysis(df, cfg)
    print(corr_df.to_string(index=False))

    print("\n=== VISUALIZING USAGE VS FAILURE ===")
    plot_usage_vs_failure(df, cfg)
    print("Usage vs failure plots saved.")

    print("\n=== MODEL EVALUATION (LOGISTIC REGRESSION & RANDOM FOREST) ===")
    metrics_df = evaluate_models(df, cfg)
    print("Model performance summary:")
    print(metrics_df.to_string(index=False))

    print("\n=== Hypothesis 3 analysis complete. All outputs saved to:")
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}")


# ------------------------------------------------------------
# 9. SCRIPT ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    # Update this path to your actual processed dataset
    DATA_PATH = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"
    run_hypothesis3_analysis(DATA_PATH)
