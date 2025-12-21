"""
Hypothesis1: cross_vendor_normalization_experiment.py

Author: Priya Pooja Hariharan
Script Version: 1.1
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 

The paired comparison between raw and normalized feature sets demonstrated that cross vendor normalization to check whether significantly improve predictive performance. 
 - with ROC AUC and PR-AUC values 
 and the differences between conditions were not statistically significant (p > .50). 

"""

"""
Hypothesis1: cross_vendor_normalization_experiment.py

Author: Priya Pooja Hariharan
Script Version: 2.0
Project: MIS581 SSD-SMART-Reliability-Analysis 

Hypothesis 1:
H₀: Cross-vendor normalization does not significantly improve predictive performance
    compared to raw features alone.
H₁: Cross-vendor normalization improves predictive performance by aligning vendor-
    specific attributes into semantically comparable latent dimensions.

This script:
- Derives a binary failure label.
- Prepares raw and cross-vendor–normalized feature sets.
- Evaluates predictive performance (ROC AUC, PR AUC, Brier score) for both.
- Compares conditions with paired t-tests.
- Evaluates cross-vendor generalization.
- Saves all metrics as CSV and all plots as PNG.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone

from scipy.stats import ttest_rel


# ------------------------------------------------------------
# 0. OUTPUT DIRECTORIES
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_hyp1")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_hyp1")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. Outcome and feature configuration
# ------------------------------------------------------------

@dataclass
class OutcomeConfig:
    binary_failure_col: str = "failure_label"


@dataclass
class FeatureConfig:
    comparable_features: List[str] = field(default_factory=lambda: [
        "power_on_hours", "power_cycles", "data_units_read", "data_units_written",
        "host_read_commands", "host_write_commands", "iops",
        "bandwidth_read_gbps", "bandwidth_write_gbps", "io_completion_time_ms",
        "controller_busy_time", "percentage_used", "wear_level_avg",
        "wear_level_max", "endurance_estimate_remaining",
        "background_scrub_time_pct", "gc_active_time_pct",
        "power_draw_w", "composite_temperature_c"
    ])

    drive_manufacturer_specific_features: List[str] = field(default_factory=lambda: [
        "media_errors", "error_information_log_entries",
        "bad_block_count_grown", "pcie_correctable_errors",
        "pcie_uncorrectable_errors", "unsafe_shutdowns",
        "throttling_events"
    ])

    drive_manufacturer_col: str = "drive_manufacturer"
    drive_model_number_col: str = "drive_model_number"


# ------------------------------------------------------------
# 2. Normalization schema
# ------------------------------------------------------------

@dataclass
class NormalizationSchema:
    semantic_map: Dict[str, str] = field(default_factory=lambda: {
        "media_errors": "media_health",
        "bad_block_count_grown": "media_health",
        "error_information_log_entries": "correctable_errors",
        "pcie_correctable_errors": "correctable_errors",
        "pcie_uncorrectable_errors": "uncorrectable_errors",
        "unsafe_shutdowns": "stress_events",
        "throttling_events": "stress_events",
    })

    scaling_strategy: Dict[str, str] = field(default_factory=lambda: {
        "media_health": "zscore",
        "correctable_errors": "zscore",
        "uncorrectable_errors": "zscore",
        "stress_events": "zscore",
    })

    lifetime_columns: Dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------
# 3. Cross-vendor normalizer
# ------------------------------------------------------------

class CrossVendorNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, schema: NormalizationSchema, feature_config: FeatureConfig):
        self.schema = schema
        self.feature_config = feature_config
        self._scalers: Dict[str, StandardScaler] = {}

    def fit(self, X, y=None):
        latent_df = self._build_latent_df(X)
        for latent in latent_df.columns:
            scaler = StandardScaler()
            scaler.fit(latent_df[[latent]])
            self._scalers[latent] = scaler
        return self

    def transform(self, X):
        latent_df = self._build_latent_df(X)
        norm_cols = {
            latent: self._scalers[latent].transform(latent_df[[latent]])[:, 0]
            for latent in latent_df.columns
        }
        norm_df = pd.DataFrame(norm_cols, index=X.index)

        comparable = X[self.feature_config.comparable_features].copy()
        comparable = comparable.reset_index(drop=True)
        norm_df = norm_df.reset_index(drop=True)

        return pd.concat([comparable, norm_df], axis=1)

    def _build_latent_df(self, X):
        latent_values = {}
        latent_to_cols: Dict[str, List[str]] = {}

        for raw_col, latent in self.schema.semantic_map.items():
            if raw_col in X.columns:
                latent_to_cols.setdefault(latent, []).append(raw_col)

        for latent, cols in latent_to_cols.items():
            latent_values[latent] = X[cols].astype(float).mean(axis=1).values

        if latent_values:
            return pd.DataFrame(latent_values, index=X.index)
        else:
            return pd.DataFrame(index=X.index)


# ------------------------------------------------------------
# 4. Failure label derivation
# ------------------------------------------------------------

def derive_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df


# ------------------------------------------------------------
# 5. Raw vs normalized dataset preparation
# ------------------------------------------------------------

def prepare_datasets(
    df: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    feature_cfg: FeatureConfig,
    normalizer: CrossVendorNormalizer
):
    y = df[outcome_cfg.binary_failure_col].astype(int)

    raw_cols = feature_cfg.comparable_features + feature_cfg.drive_manufacturer_specific_features
    raw_cols = [c for c in raw_cols if c in df.columns]
    X_raw = df[raw_cols].copy()

    normalizer.fit(df)
    X_norm = normalizer.transform(df)

    return X_raw, y, X_norm, y


# ------------------------------------------------------------
# 6. Model evaluation
# ------------------------------------------------------------

@dataclass
class ModelConfig:
    model_family: str = "rf"
    random_state: int = 42
    n_splits: int = 5


def build_model(cfg: ModelConfig):
    if cfg.model_family == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    return LogisticRegression(max_iter=500, class_weight="balanced")


def evaluate_condition(
    X: pd.DataFrame,
    y: pd.Series,
    model_cfg: ModelConfig,
    label: str,
    save_predictions: bool = True
) -> Dict[str, np.ndarray]:
    model = build_model(model_cfg)
    skf = StratifiedKFold(
        n_splits=model_cfg.n_splits,
        shuffle=True,
        random_state=model_cfg.random_state
    )

    roc_scores, pr_scores, brier_scores = [], [], []
    all_pred_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pre = ColumnTransformer(
            [("num", StandardScaler(), X.columns)],
            remainder="drop"
        )
        clf = Pipeline([("pre", pre), ("model", clone(model))])

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]

        roc_scores.append(roc_auc_score(y_te, proba))
        pr_scores.append(average_precision_score(y_te, proba))
        brier_scores.append(brier_score_loss(y_te, proba))

        if save_predictions:
            fold_df = pd.DataFrame({
                "fold": fold,
                "true_label": y_te.values,
                "predicted_prob": proba
            })
            all_pred_rows.append(fold_df)

    if save_predictions and all_pred_rows:
        preds_df = pd.concat(all_pred_rows, ignore_index=True)
        preds_path = os.path.join(RESULTS_DIR, f"predictions_{label}.csv")
        preds_df.to_csv(preds_path, index=False)

    return {
        "roc_auc": np.array(roc_scores),
        "pr_auc": np.array(pr_scores),
        "brier": np.array(brier_scores)
    }


# ------------------------------------------------------------
# 7. Cross-vendor generalization
# ------------------------------------------------------------

def train_test_vendor_split(
    df: pd.DataFrame,
    vendors_train: List[str],
    outcome_cfg: OutcomeConfig,
    feature_cfg: FeatureConfig,
    normalizer: CrossVendorNormalizer,
    model_cfg: ModelConfig
) -> Dict[str, float]:

    train_mask = df[feature_cfg.drive_manufacturer_col].isin(vendors_train)
    df_train = df[train_mask]
    df_test = df[~train_mask]

    # If test set is empty or one class only, skip
    if df_test.empty or df_test[outcome_cfg.binary_failure_col].nunique() < 2:
        print("Skipping cross-vendor eval: insufficient test data or only one class.")
        return {}

    y_train = df_train[outcome_cfg.binary_failure_col].astype(int)
    y_test = df_test[outcome_cfg.binary_failure_col].astype(int)

    raw_cols = feature_cfg.comparable_features + feature_cfg.drive_manufacturer_specific_features
    raw_cols = [c for c in raw_cols if c in df.columns]

    X_raw_train = df_train[raw_cols].copy()
    X_raw_test = df_test[raw_cols].copy()

    normalizer.fit(df_train)
    X_norm_train = normalizer.transform(df_train)
    X_norm_test = normalizer.transform(df_test)

    base_model = build_model(model_cfg)

    # Raw condition
    pre_raw = ColumnTransformer(
        [("num", StandardScaler(), X_raw_train.columns)],
        remainder="drop"
    )
    clf_raw = Pipeline([
        ("pre", pre_raw),
        ("model", clone(base_model))
    ])
    clf_raw.fit(X_raw_train, y_train)
    proba_raw = clf_raw.predict_proba(X_raw_test)[:, 1]

    roc_raw = roc_auc_score(y_test, proba_raw)
    pr_raw = average_precision_score(y_test, proba_raw)
    brier_raw = brier_score_loss(y_test, proba_raw)

    # Normalized condition
    pre_norm = ColumnTransformer(
        [("num", StandardScaler(), X_norm_train.columns)],
        remainder="drop"
    )
    clf_norm = Pipeline([
        ("pre", pre_norm),
        ("model", clone(base_model))
    ])
    clf_norm.fit(X_norm_train, y_train)
    proba_norm = clf_norm.predict_proba(X_norm_test)[:, 1]

    roc_norm = roc_auc_score(y_test, proba_norm)
    pr_norm = average_precision_score(y_test, proba_norm)
    brier_norm = brier_score_loss(y_test, proba_norm)

    results = {
        "roc_auc_raw": roc_raw,
        "roc_auc_norm": roc_norm,
        "pr_auc_raw": pr_raw,
        "pr_auc_norm": pr_norm,
        "brier_raw": brier_raw,
        "brier_norm": brier_norm,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "vendors_train": ",".join(vendors_train),
        "vendors_test": ",".join(df_test[feature_cfg.drive_manufacturer_col].unique().tolist())
    }

    # Save cross-vendor predictions (optional)
    cv_pred_df = pd.DataFrame({
        "true_label": y_test.values,
        "predicted_prob_raw": proba_raw,
        "predicted_prob_norm": proba_norm
    })
    cv_pred_df.to_csv(
        os.path.join(RESULTS_DIR, "cross_vendor_predictions.csv"),
        index=False
    )

    return results


# ------------------------------------------------------------
# 8. Statistical comparison
# ------------------------------------------------------------

def compare_conditions(res_raw: Dict[str, np.ndarray],
                       res_norm: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for metric in ["roc_auc", "pr_auc", "brier"]:
        stat, p = ttest_rel(res_raw[metric], res_norm[metric])
        out[metric] = {
            "raw_mean": float(np.mean(res_raw[metric])),
            "norm_mean": float(np.mean(res_norm[metric])),
            "diff": float(np.mean(res_norm[metric]) - np.mean(res_raw[metric])),
            "t_stat": float(stat),
            "p_value": float(p)
        }
    return out


# ------------------------------------------------------------
# 8B. Visualization utilities
# ------------------------------------------------------------

def plot_boxplots(res_raw: Dict[str, np.ndarray],
                  res_norm: Dict[str, np.ndarray]) -> None:
    metrics = ["roc_auc", "pr_auc", "brier"]
    frames = []

    for m in metrics:
        frames.append(pd.DataFrame({
            "metric": m,
            "score": res_raw[m],
            "condition": "raw"
        }))
        frames.append(pd.DataFrame({
            "metric": m,
            "score": res_norm[m],
            "condition": "normalized"
        }))

    df_plot = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x="metric", y="score", hue="condition")
    sns.stripplot(
        data=df_plot,
        x="metric",
        y="score",
        hue="condition",
        dodge=True,
        alpha=0.4,
        palette="dark:black"
    )
    # Avoid duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2])

    plt.title("Raw vs Normalized Performance Comparison")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "raw_vs_norm_boxplots.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_distributions(res_raw: Dict[str, np.ndarray],
                       res_norm: Dict[str, np.ndarray]) -> None:
    metrics = ["roc_auc", "pr_auc", "brier"]

    for m in metrics:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(res_raw[m], fill=True, label="Raw", alpha=0.5, color="blue")
        sns.kdeplot(res_norm[m], fill=True, label="Normalized", alpha=0.5, color="orange")

        sns.histplot(res_raw[m], kde=False, color="blue", alpha=0.3)
        sns.histplot(res_norm[m], kde=False, color="orange", alpha=0.3)

        plt.title(f"Distribution of {m.upper()} Scores")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, f"raw_vs_norm_distribution_{m}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


# ------------------------------------------------------------  
# 9. Main experiment runner
# ------------------------------------------------------------  

def run_cross_vendor_experiment(df_raw: pd.DataFrame, df_syn: pd.DataFrame) -> None:
    print("\n=== VALIDATING SCHEMA (raw vs synthetic columns match) ===")
    print(set(df_raw.columns) == set(df_syn.columns))

    print("\n=== DERIVING FAILURE LABEL ON SYNTHETIC DATASET ===")
    df_syn = derive_failure_label(df_syn)

    outcome_cfg = OutcomeConfig()
    feature_cfg = FeatureConfig()
    schema = NormalizationSchema()
    normalizer = CrossVendorNormalizer(schema, feature_cfg)

    print("\n=== PREPARING DATASETS (RAW VS NORMALIZED) ===")
    X_raw, y, X_norm, y_norm = prepare_datasets(df_syn, outcome_cfg, feature_cfg, normalizer)

    print("\n=== RAW VS NORMALIZED PERFORMANCE (CROSS-VALIDATION) ===")
    model_cfg = ModelConfig()
    res_raw = evaluate_condition(X_raw, y, model_cfg, label="raw")
    res_norm = evaluate_condition(X_norm, y_norm, model_cfg, label="normalized")

    # Save per-fold metrics
    pd.DataFrame({
        "roc_auc": res_raw["roc_auc"],
        "pr_auc": res_raw["pr_auc"],
        "brier": res_raw["brier"]
    }).to_csv(os.path.join(RESULTS_DIR, "raw_model_metrics.csv"), index=False)

    pd.DataFrame({
        "roc_auc": res_norm["roc_auc"],
        "pr_auc": res_norm["pr_auc"],
        "brier": res_norm["brier"]
    }).to_csv(os.path.join(RESULTS_DIR, "normalized_model_metrics.csv"), index=False)

    print("\n=== GENERATING PLOTS ===")
    plot_boxplots(res_raw, res_norm)
    plot_distributions(res_raw, res_norm)

    print("\n=== STATISTICAL COMPARISON (PAIRED T-TESTS) ===")
    comp = compare_conditions(res_raw, res_norm)
    comp_df = pd.DataFrame(comp).T
    comp_df.to_csv(os.path.join(RESULTS_DIR, "statistical_comparison_raw_vs_normalized.csv"))

    for metric, vals in comp.items():
        print(f"\nMetric: {metric}")
        for k, v in vals.items():
            print(f"  {k}: {v}")

    print("\n=== CROSS-VENDOR GENERALIZATION (TRAIN ON N-1 VENDORS, TEST ON HELD-OUT) ===")
    vendors = df_syn["drive_manufacturer"].unique().tolist()
    if len(vendors) > 1:
        vendors_train = vendors[:-1]
        cvg_results = train_test_vendor_split(
            df_syn,
            vendors_train,
            outcome_cfg,
            feature_cfg,
            normalizer,
            model_cfg
        )
        if cvg_results:
            cvg_df = pd.DataFrame([cvg_results])
            cvg_df.to_csv(
                os.path.join(RESULTS_DIR, "cross_vendor_generalization_results.csv"),
                index=False
            )
            print("\nCross-vendor generalization results:")
            print(cvg_df.to_string(index=False))
        else:
            print("Cross-vendor generalization could not be evaluated (insufficient test data).")
    else:
        print("Not enough vendors for cross-vendor generalization.")


# ------------------------------------------------------------
# 10. Script entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    df_raw = pd.read_csv(
        r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step1-processed_smart_dataset_V1.1.csv"
    )
    df_syn = pd.read_csv(
        r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"
    )

    run_cross_vendor_experiment(df_raw, df_syn)
