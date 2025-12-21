import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone

from scipy.stats import ttest_rel


# ------------------------------------------------------------
# 1. Outcome and feature configuration
# ------------------------------------------------------------

@dataclass
class OutcomeConfig:
    """Config for defining degradation / failure outcomes from SMART logs."""
    binary_failure_col: str = "failure_label"      # derived 0/1 label
    time_to_failure_col: str = "time_to_failure"   # placeholder if needed later


@dataclass
class FeatureConfig:
    """Config for selecting SMART predictor sets."""
    # Cross-vendor comparable SMART / telemetry attributes
    comparable_features: List[str] = field(default_factory=lambda: [
        "power_on_hours",
        "power_cycles",
        "data_units_read",
        "data_units_written",
        "host_read_commands",
        "host_write_commands",
        "iops",
        "bandwidth_read_gbps",
        "bandwidth_write_gbps",
        "io_completion_time_ms",
        "controller_busy_time",
        "percentage_used",
        "wear_level_avg",
        "wear_level_max",
        "endurance_estimate_remaining",
        "background_scrub_time_pct",
        "gc_active_time_pct",
        "power_draw_w",
        "composite_temperature_c",
    ])

    # Attributes that are treated as more vendor/implementation specific
    # and will be mapped into latent constructs.
    drive_manufacturer_specific_features: List[str] = field(default_factory=lambda: [
        "media_errors",
        "error_information_log_entries",
        "bad_block_count_grown",
        "pcie_correctable_errors",
        "pcie_uncorrectable_errors",
        "unsafe_shutdowns",
        "throttling_events",
    ])

    drive_manufacturer_col: str = "drive_manufacturer"
    drive_model_number_col: str = "drive_model_number"


# ------------------------------------------------------------
# 2. Normalization schema: semantic mapping & scaling
# ------------------------------------------------------------

@dataclass
class NormalizationSchema:
    """
    Defines the cross-vendor normalization logic:
    - semantic mapping of vendor-specific cols to latent constructs
    - numerical scaling strategies per construct
    - optional lifetime/denominator columns for ratio scaling
    """
    # Map vendor-specific columns to shared latent constructs
    semantic_map: Dict[str, str] = field(default_factory=lambda: {
        # "raw_col": "latent_category"
        "media_errors": "media_health",
        "bad_block_count_grown": "media_health",
        "error_information_log_entries": "correctable_errors",
        "pcie_correctable_errors": "correctable_errors",
        "pcie_uncorrectable_errors": "uncorrectable_errors",
        "unsafe_shutdowns": "stress_events",
        "throttling_events": "stress_events",
    })

    # Scaling strategy per latent construct
    scaling_strategy: Dict[str, str] = field(default_factory=lambda: {
        # "latent_category": "zscore" | "minmax" | "ratio_lifetime"
        "media_health": "zscore",
        "correctable_errors": "zscore",
        "uncorrectable_errors": "zscore",
        "stress_events": "zscore",
    })

    # Optional: lifetime / denominator columns for ratio scaling
    lifetime_columns: Dict[str, str] = field(default_factory=lambda: {
        # e.g. "wear": "nand_write_cycles_max"
    })


# ------------------------------------------------------------
# 3. Transformer: raw vendor-specific → normalized cross-vendor features
# ------------------------------------------------------------

class CrossVendorNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformer implementing cross-vendor normalization:
    - Builds latent constructs from vendor-specific SMART attributes
    - Applies construct-level scaling (z-score, min-max, ratio-to-lifetime)
    - Returns a feature set combining comparable + normalized latent constructs
    """
    def __init__(self, schema: NormalizationSchema, feature_config: FeatureConfig):
        self.schema = schema
        self.feature_config = feature_config
        self._scalers: Dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Build latent construct values before fitting scalers
        latent_df = self._build_latent_df(X)

        # Fit scalers per latent construct
        for latent_name in latent_df.columns:
            strategy = self.schema.scaling_strategy.get(latent_name, "zscore")

            if strategy == "zscore":
                scaler = StandardScaler()
            elif strategy == "minmax":
                scaler = MinMaxScaler()
            elif strategy == "ratio_lifetime":
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling strategy: {strategy}")

            scaler.fit(latent_df[[latent_name]])
            self._scalers[latent_name] = scaler

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        latent_df = self._build_latent_df(X)

        # Apply learned scalers
        norm_cols = {}
        for latent_name in latent_df.columns:
            scaler = self._scalers.get(latent_name, None)
            if scaler is None:
                norm_cols[latent_name] = latent_df[latent_name].values
            else:
                norm_cols[latent_name] = scaler.transform(latent_df[[latent_name]])[:, 0]

        norm_df = pd.DataFrame(norm_cols, index=X.index)

        # Keep comparable features alongside normalized latent ones
        comparable_cols = [c for c in self.feature_config.comparable_features if c in X.columns]
        comparable = X[comparable_cols].copy()
        comparable = comparable.reset_index(drop=True)
        norm_df = norm_df.reset_index(drop=True)

        out = pd.concat([comparable, norm_df], axis=1)
        return out

    def _build_latent_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs per-latent-category numeric values by aggregating mapped
        vendor-specific features. Current implementation: mean across all mapped
        columns per latent construct.
        """
        latent_values: Dict[str, np.ndarray] = {}

        # Group raw columns by latent construct
        latent_to_cols: Dict[str, List[str]] = {}
        for raw_col, latent_name in self.schema.semantic_map.items():
            if raw_col in X.columns:
                latent_to_cols.setdefault(latent_name, []).append(raw_col)

        for latent_name, cols in latent_to_cols.items():
            subset = X[cols].astype(float)
            latent_val = subset.mean(axis=1).values

            # If ratio_lifetime: divide by a specified lifetime column
            strategy = self.schema.scaling_strategy.get(latent_name, None)
            if strategy == "ratio_lifetime":
                denom_col = self.schema.lifetime_columns.get(latent_name, None)
                if denom_col is None or denom_col not in X.columns:
                    raise ValueError(f"No lifetime column for latent '{latent_name}'")
                denom = X[denom_col].replace(0, np.nan)
                latent_val = latent_val / denom.values

            latent_values[latent_name] = latent_val

        latent_df = pd.DataFrame(latent_values, index=X.index)

        # Example: composite degradation score if desired
        # if {"media_health", "correctable_errors"}.issubset(latent_df.columns):
        #     latent_df["degradation_score"] = (
        #         0.6 * latent_df["media_health"] +
        #         0.4 * latent_df["correctable_errors"]
        #     )

        return latent_df


# ------------------------------------------------------------
# 4. Outcome derivation: build failure_label from real SMART fields
# ------------------------------------------------------------

def derive_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives a binary failure/degradation label from available SMART-like fields.
    This creates 'failure_label' = 1 when any failure precursor condition is met.
    """

    df = df.copy()

    # Fill missing with zeros for error-like fields to avoid NaNs in logic
    for col in [
        "percentage_used",
        "wear_level_max",
        "endurance_estimate_remaining",
        "media_errors",
        "bad_block_count_grown",
        "pcie_uncorrectable_errors",
        "unsafe_shutdowns",
        "throttling_events",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    conditions = []

    # High wear / near end-of-life
    if "percentage_used" in df.columns:
        conditions.append(df["percentage_used"] >= 90)

    if "wear_level_max" in df.columns:
        conditions.append(df["wear_level_max"] >= 95)

    if "endurance_estimate_remaining" in df.columns:
        conditions.append(df["endurance_estimate_remaining"] <= 10)

    # Error indicators
    if "media_errors" in df.columns:
        conditions.append(df["media_errors"] > 0)

    if "bad_block_count_grown" in df.columns:
        conditions.append(df["bad_block_count_grown"] > 0)

    if "pcie_uncorrectable_errors" in df.columns:
        conditions.append(df["pcie_uncorrectable_errors"] > 0)

    # Operational stress
    if "unsafe_shutdowns" in df.columns:
        conditions.append(df["unsafe_shutdowns"] > 0)

    if "throttling_events" in df.columns:
        conditions.append(df["throttling_events"] > 0)

    if len(conditions) == 0:
        raise ValueError("No usable columns found to derive failure_label.")

    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined | cond

    df["failure_label"] = combined.astype(int)
    return df


# ------------------------------------------------------------
# 5. Dataset preparation: raw vs normalized conditions
# ------------------------------------------------------------

def prepare_datasets(
    df: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    feature_cfg: FeatureConfig,
    normalizer: CrossVendorNormalizer
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns:
        X_raw, y_raw, X_norm, y_norm
    """
    y = df[outcome_cfg.binary_failure_col].astype(int)

    raw_feature_cols = (
        feature_cfg.comparable_features +
        feature_cfg.drive_manufacturer_specific_features
    )
    raw_feature_cols = [c for c in raw_feature_cols if c in df.columns]
    X_raw = df[raw_feature_cols].copy()

    normalizer.fit(df)
    X_norm = normalizer.transform(df)

    return X_raw, y, X_norm, y


# ------------------------------------------------------------
# 6. Model building and evaluation for each condition
# ------------------------------------------------------------

@dataclass
class ModelConfig:
    model_family: str = "logreg"   # "logreg" | "rf"
    random_state: int = 42
    n_splits: int = 5


def build_model(cfg: ModelConfig):
    if cfg.model_family == "logreg":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced"
        )
    elif cfg.model_family == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
    else:
        raise ValueError(f"Unknown model family: {cfg.model_family}")
    return model


def evaluate_condition(
    X: pd.DataFrame,
    y: pd.Series,
    model_cfg: ModelConfig,
    description: str = "raw"
) -> Dict[str, np.ndarray]:
    """
    Cross-validated evaluation of a single condition (raw or normalized).
    Returns dict of metric arrays across folds.
    """
    model = build_model(model_cfg)
    skf = StratifiedKFold(
        n_splits=model_cfg.n_splits,
        shuffle=True,
        random_state=model_cfg.random_state
    )

    roc_scores, pr_scores, brier_scores = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        numeric_cols = X.columns.tolist()
        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
            ],
            remainder="drop"
        )

        clf = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", clone(model))
        ])

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]

        roc = roc_auc_score(y_te, proba)
        pr = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)

        roc_scores.append(roc)
        pr_scores.append(pr)
        brier_scores.append(brier)

    results = {
        "description": description,
        "roc_auc": np.array(roc_scores),
        "pr_auc": np.array(pr_scores),
        "brier": np.array(brier_scores)
    }
    return results


# ------------------------------------------------------------
# 7. Cross-vendor generalization experiment
# ------------------------------------------------------------

def train_test_vendor_split(
    df: pd.DataFrame,
    vendors_train: List[str],
    outcome_cfg: OutcomeConfig,
    feature_cfg: FeatureConfig,
    normalizer: CrossVendorNormalizer
) -> Dict[str, float]:
    """
    Cross-vendor generalization: train on vendors_train, test on others.
    Returns metrics for raw and normalized models.
    """
    train_mask = df[feature_cfg.drive_manufacturer_col].isin(vendors_train)
    test_mask = ~train_mask

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    y_train = df_train[outcome_cfg.binary_failure_col].astype(int)
    y_test = df_test[outcome_cfg.binary_failure_col].astype(int)

    raw_cols = (
        feature_cfg.comparable_features +
        feature_cfg.drive_manufacturer_specific_features
    )
    raw_cols = [c for c in raw_cols if c in df_train.columns]

    X_raw_train = df_train[raw_cols].copy()
    X_raw_test = df_test[raw_cols].copy()

    normalizer.fit(df_train)
    X_norm_train = normalizer.transform(df_train)
    X_norm_test = normalizer.transform(df_test)

    model_cfg = ModelConfig(model_family="rf")
    model = build_model(model_cfg)

    def train_eval(X_tr, X_te, label: str) -> Dict[str, float]:
        numeric_cols = X_tr.columns.tolist()
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        preprocessor = ColumnTransformer(
            [("num", numeric_transformer, numeric_cols)],
            remainder="drop"
        )
        clf = Pipeline([
            ("pre", preprocessor),
            ("model", clone(model))
        ])
        clf.fit(X_tr, y_train)
        proba = clf.predict_proba(X_te)[:, 1]

        return {
            f"{label}_roc_auc": roc_auc_score(y_test, proba),
            f"{label}_pr_auc": average_precision_score(y_test, proba),
        }

    raw_metrics = train_eval(X_raw_train, X_raw_test, "raw")
    norm_metrics = train_eval(X_norm_train, X_norm_test, "norm")

    out = {}
    out.update(raw_metrics)
    out.update(norm_metrics)
    return out


# ------------------------------------------------------------
# 8. Synthetic error injection
# ------------------------------------------------------------

def inject_synthetic_errors(
    df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    pattern_fn: Callable[[pd.DataFrame], pd.DataFrame],
    affected_fraction: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Applies pattern_fn to a random subset of drives to create synthetic degradation signatures.
    pattern_fn: function that takes a subset and returns modified subset.
    """
    rng = np.random.RandomState(random_state)
    df = df.copy()

    unique_drives = df[feature_cfg.drive_model_number_col].unique()
    n_affected = max(1, int(len(unique_drives) * affected_fraction))
    affected_drives = rng.choice(unique_drives, size=n_affected, replace=False)

    mask = df[feature_cfg.drive_model_number_col].isin(affected_drives)
    df_affected = df[mask].copy()
    df_unaffected = df[~mask].copy()

    df_affected_mod = pattern_fn(df_affected)
    out = pd.concat([df_unaffected, df_affected_mod], axis=0).sort_index()
    return out


def accelerated_wear_pattern(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Example synthetic degradation pattern:
    Accelerated wear and error growth for selected drives.
    """
    df_subset = df_subset.copy()

    if "percentage_used" in df_subset.columns:
        df_subset["percentage_used"] = df_subset["percentage_used"] + 10

    if "wear_level_max" in df_subset.columns:
        df_subset["wear_level_max"] = df_subset["wear_level_max"] + 10

    if "media_errors" in df_subset.columns:
        df_subset["media_errors"] = df_subset["media_errors"] + 5

    if "bad_block_count_grown" in df_subset.columns:
        df_subset["bad_block_count_grown"] = df_subset["bad_block_count_grown"] + 10

    return df_subset


# ------------------------------------------------------------
# 9. Statistical comparison: raw vs normalized performance
# ------------------------------------------------------------

def compare_conditions(
    results_raw: Dict[str, np.ndarray],
    results_norm: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Paired t-tests on CV metrics across folds for raw vs normalized models.
    """
    comparisons = {}

    for metric in ["roc_auc", "pr_auc", "brier"]:
        x = results_raw[metric]
        y = results_norm[metric]
        stat, p = ttest_rel(x, y)
        comparisons[metric] = {
            "raw_mean": float(np.mean(x)),
            "norm_mean": float(np.mean(y)),
            "diff": float(np.mean(y) - np.mean(x)),
            "t_stat": float(stat),
            "p_value": float(p),
        }

    return comparisons


# ------------------------------------------------------------
# 10. Driver function to run the full experiment
# ------------------------------------------------------------

def run_experiment(df: pd.DataFrame):
    """
    Top-level orchestration:
    - Derive failure_label
    - Prepare raw and normalized feature sets
    - Evaluate predictive models in both conditions
    - Run statistical comparisons
    - Run cross-vendor generalization
    """
    df = derive_failure_label(df)

    outcome_cfg = OutcomeConfig(binary_failure_col="failure_label")
    feature_cfg = FeatureConfig(
        comparable_features=[
            "power_on_hours",
            "power_cycles",
            "data_units_read",
            "data_units_written",
            "host_read_commands",
            "host_write_commands",
            "iops",
            "bandwidth_read_gbps",
            "bandwidth_write_gbps",
            "io_completion_time_ms",
            "controller_busy_time",
            "percentage_used",
            "wear_level_avg",
            "wear_level_max",
            "endurance_estimate_remaining",
            "background_scrub_time_pct",
            "gc_active_time_pct",
            "power_draw_w",
            "composite_temperature_c",
        ],
        drive_manufacturer_specific_features=[
            "media_errors",
            "error_information_log_entries",
            "bad_block_count_grown",
            "pcie_correctable_errors",
            "pcie_uncorrectable_errors",
            "unsafe_shutdowns",
            "throttling_events",
        ],
        drive_manufacturer_col="drive_manufacturer",
        drive_model_number_col="drive_model_number"
    )

    schema = NormalizationSchema()
    normalizer = CrossVendorNormalizer(schema, feature_cfg)

    # Prepare datasets
    X_raw, y, X_norm, y_norm = prepare_datasets(df, outcome_cfg, feature_cfg, normalizer)

    # Evaluate both conditions
    model_cfg = ModelConfig(model_family="rf", n_splits=5)
    res_raw = evaluate_condition(X_raw, y, model_cfg, description="raw")
    res_norm = evaluate_condition(X_norm, y_norm, model_cfg, description="normalized")

    # Statistical comparison
    comp = compare_conditions(res_raw, res_norm)
    print("Paired comparison raw vs normalized:")
    for metric, vals in comp.items():
        print(f"Metric: {metric}")
        print(f"  raw_mean   = {vals['raw_mean']:.4f}")
        print(f"  norm_mean  = {vals['norm_mean']:.4f}")
        print(f"  diff       = {vals['diff']:.4f} (norm - raw)")
        print(f"  t_stat     = {vals['t_stat']:.4f}")
        print(f"  p_value    = {vals['p_value']:.4e}")
        print("")

    # Cross-vendor generalization
    unique_vendors = df[feature_cfg.drive_manufacturer_col].dropna().unique().tolist()
    if len(unique_vendors) >= 2:
        vendors_train = unique_vendors[:-1]
        cvg_metrics = train_test_vendor_split(
            df, vendors_train, outcome_cfg, feature_cfg, normalizer
        )
        print("Cross-vendor generalization metrics:")
        for k, v in cvg_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Not enough distinct vendors in data for cross-vendor generalization.")


# ------------------------------------------------------------
# 11. Example usage placeholder
# ------------------------------------------------------------

if __name__ == "__main__":
    # Replace this with actual data loading.
    # Example:
    df = pd.read_csv(r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv")
    header=None
    #parse_dates=["timestamp"], # optional 
    #low_memory=False
    #)
    n = 1000
    rng = np.random.RandomState(42)

    df_dummy = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
        "form_factor": rng.choice(["U.2", "E1.s", "E3.s", "U.3", "M.2"], size=n),
        "NAND_type": rng.choice(["TLC", "QLC"], size=n),
        "drive_model_number": rng.randint(0, 50, size=n),
        "drive_firmware_revision": rng.choice(["FW1", "FW2", "FW3"], size=n),
        "drive_manufacturer": rng.choice(
            ["Samsung", "Sandisk", "Micron", "Intel", "Kioxia", "Solidigm", "SKHynix", "Seagate"],
            size=n
        ),
        "Server_configuration": rng.choice(["SYS-2029U", "ThinkSystem SR650"], size=n),
        "Server_CPU_Name": rng.choice(["Intel", "AMD"], size=n),
        "nvme_capacity_tb": rng.choice([1, 2, 4], size=n),
        "overprovisioning_ratio": rng.uniform(0.05, 0.3, size=n),
        "composite_temperature_c": rng.normal(40, 5, size=n),
        "data_units_read": rng.uniform(0, 1e7, size=n),
        "data_units_written": rng.uniform(0, 1e7, size=n),
        "host_read_commands": rng.uniform(0, 1e6, size=n),
        "host_write_commands": rng.uniform(0, 1e6, size=n),
        "avg_queue_depth": rng.uniform(0, 16, size=n),
        "iops": rng.uniform(1e3, 1e5, size=n),
        "bandwidth_read_gbps": rng.uniform(0, 7, size=n),
        "bandwidth_write_gbps": rng.uniform(0, 7, size=n),
        "io_completion_time_ms": rng.exponential(scale=1.0, size=n),
        "power_cycles": rng.poisson(300, size=n),
        "power_on_hours": rng.uniform(0, 50000, size=n),
        "controller_busy_time": rng.uniform(0, 1e6, size=n),
        "percentage_used": rng.uniform(0, 100, size=n),
        "wear_level_avg": rng.uniform(0, 100, size=n),
        "wear_level_max": rng.uniform(0, 100, size=n),
        "endurance_estimate_remaining": rng.uniform(0, 100, size=n),
        "unsafe_shutdowns": rng.poisson(0.1, size=n),
        "background_scrub_time_pct": rng.uniform(0, 20, size=n),
        "gc_active_time_pct": rng.uniform(0, 20, size=n),
        "media_errors": rng.poisson(0.05, size=n),
        "error_information_log_entries": rng.poisson(1.0, size=n),
        "bad_block_count_grown": rng.poisson(0.1, size=n),
        "pcie_correctable_errors": rng.poisson(0.5, size=n),
        "pcie_uncorrectable_errors": rng.poisson(0.02, size=n),
        "workload_type": rng.choice(["read_heavy", "write_heavy", "mixed"], size=n),
        "power_draw_w": rng.uniform(5, 20, size=n),
        "throttling_events": rng.poisson(0.05, size=n),
    })

    # Optional: inject synthetic degradation to test sensitivity
    # df_dummy = inject_synthetic_errors(
    #     df_dummy,
    #     FeatureConfig(),
    #     pattern_fn=accelerated_wear_pattern,
    #     affected_fraction=0.2
    # )

    run_experiment(df_dummy)
