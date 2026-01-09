"""
Hypothesis2: Hyp4_Bayesian_Model_experiment_V1.2.py

Author: Priya Pooja Hariharan
Script Version: 1.2
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 




"""




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import pymc as pm
import arviz as az


# ============================================================
# 0. OUTPUT FOLDERS
# ============================================================
RESULTS_DIR = r"C:\Users\venki\Desktop\Bayesian_Hyp4_Results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ============================================================
# 1. MAIN EXECUTION BLOCK
# ============================================================
def main():

    # ============================================================
    # 2. LOAD DATA
    # ============================================================
    #data_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"
    data_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"
    df = pd.read_csv(data_path, low_memory=False)
    df["throttling_events"] = (
        (df["queue_depth"].isin([128, 256])) &
        (df["overprovisioning_ratio"] > 10) &
        (df["composite_temperature_c"] > 50) &
        (df["data_units_read"] > 200_000_000) &
        (df["data_units_written"] > 100_000_000)
    )
    # ============================================================
    # 3. FAILURE LABEL DEFINITION
    # ============================================================
    df["failure_label"] = (

    # ============================================================
    # A. THERMAL FAILURE INDICATORS
    # ============================================================
    (df["composite_temperature_c"] >= 60) |      # High temperature
    (df["composite_temperature_c"] >= 70) |      # Critical temperature

    # ============================================================
    # B. PCIE / INTERFACE RELIABILITY
    # ============================================================
    (df["pcie_uncorrectable_errors"] > 0) |      # Catastrophic
    (df["pcie_correctable_errors"] >= 50) |      # Interface instability
    (df["controller_busy_time"] > 1000) |        # Controller saturation

    # ============================================================
    # C. PERFORMANCE DEGRADATION
    # ============================================================
    (df["bandwidth_read_gbps"] < 1.0) |          # Severe read degradation
    (df["iops"] < 50000) |                       # Low IOPS
    (df["io_completion_time_ms"] > 2.0) |        # High latency

    # ============================================================
    # D. MEDIA / NAND HEALTH
    # ============================================================
    (df["media_errors"] > 0) |                   # Media error
    (df["bad_block_count_grown"] > 0) |          # Grown bad blocks
    (df["error_information_log_entries"] > 0) |  # Firmware‑reported issues
    (df["wear_level_max"] >= 90) |               # Near end of life
    (df["endurance_estimate_remaining"] <= 5) |  # Endurance nearly exhausted

    # ============================================================
    # E. POWER / SHUTDOWN INTEGRITY
    # ============================================================
    (df["unsafe_shutdowns"] > 2) |               # Unsafe shutdowns
    (df["power_cycles"] > 200) |                # Excessive power cycling

    # ============================================================
    # F. BACKGROUND MAINTENANCE ANOMALIES
    # ============================================================
    (df["gc_active_time_pct"] > 10) |            # Excessive GC
    (df["background_scrub_time_pct"] > 5) |      # Excessive scrubbing

    # ============================================================
    # G. USAGE / WEAR / ENDURANCE
    # ============================================================
    (df["percentage_used"] >= 95) |              # Near end of life
    (df["wear_level_avg"] >= 80) |               # High wear

    # ============================================================
    # H. ORIGINAL RULES YOU PROVIDED (kept for compatibility)
    # ============================================================
        

    (df["bandwidth_read_gbps"] <= 5) |       # Original threshold
    (df["pcie_correctable_errors"] >= 50) |       # Original threshold
    (df["throttling_events"])               # Throttling events

	).astype(int)


    # ============================================================
    # 4. FEATURE SET
    # ============================================================
    feature_cols = [
        "overprovisioning_ratio",
        "composite_temperature_c",
        "iops",
        "bandwidth_read_gbps",
        "bandwidth_write_gbps",
        "io_completion_time_ms",
        "power_cycles",
        "power_on_hours",
        "controller_busy_time",
        "percentage_used",
        "wear_level_avg",
        "wear_level_max",
        "endurance_estimate_remaining",
        "unsafe_shutdowns",
        "background_scrub_time_pct",
        "gc_active_time_pct",
        "media_errors",
        "error_information_log_entries",
        "bad_block_count_grown",
        "pcie_correctable_errors",
        "pcie_uncorrectable_errors",
        "throttling_events"
    ]

    df_model = df.dropna(subset=feature_cols + ["failure_label"]).copy()
    # ============================================================
    # 4.a ENSURE FAILURES ARE INCLUDED IN SAMPLE
    # ============================================================
    df_fail = df_model[df_model["failure_label"] == 1]
    df_ok = df_model[df_model["failure_label"] == 0]

    df_sample = pd.concat([
    df_fail.sample(n=min(200, len(df_fail)), random_state=42),
    #df_ok.sample(n=2000, random_state=42)
    df_ok.sample(n=min(2000, len(df_ok)), random_state=42)
    ]).sample(frac=1, random_state=42)

    X = df_sample[feature_cols].values
    y = df_sample["failure_label"].values.astype(int)
    # ============================================================
    # 5. SAMPLE DATA FOR SPEED
    # ============================================================
    df_sample = df_model.sample(n=3000, random_state=42)

    X = df_sample[feature_cols].values
    y = df_sample["failure_label"].values.astype(int)

    # ============================================================
    # 6. TRAIN-TEST SPLIT + SCALING
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_features = X_train_scaled.shape[1]

    # ============================================================
    # 7. BAYESIAN MODEL (FAST ADVI VERSION)
    # ============================================================
    with pm.Model() as bayes_logit_model:

        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        betas = pm.Normal("betas", mu=0.0, sigma=2.5, shape=n_features)

        mu = alpha + pm.math.dot(X_train_scaled, betas)
        p = pm.Deterministic("p", pm.math.sigmoid(mu))

        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

        approx = pm.fit(
            method="advi",
            n=15000,
            random_seed=42
        )

        trace = approx.sample(1000)

    # ============================================================
    # 8. SAVE POSTERIOR SUMMARY
    # ============================================================
    summary = az.summary(trace, var_names=["alpha", "betas"])
    summary.to_csv(os.path.join(CSV_DIR, "posterior_summary.csv"))

    coef_summary = az.summary(trace, var_names=["betas"])
    coef_summary["feature"] = feature_cols
    coef_summary.to_csv(os.path.join(CSV_DIR, "coefficients_with_features.csv"))

    # ============================================================
    # 9. TRACE PLOTS
    # ============================================================
    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trace_plots.png"))
    plt.close()

    # ============================================================
    # 10. POSTERIOR PREDICTIVE
    # ============================================================
    posterior = trace.posterior
    alpha_samples = posterior["alpha"].values.reshape(-1, 1)
    beta_samples = posterior["betas"].values.reshape(-1, n_features)

    logits = alpha_samples + np.dot(beta_samples, X_test_scaled.T)
    p_test = 1 / (1 + np.exp(-logits))
    p_test_mean = p_test.mean(axis=0)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_prob": p_test_mean
    })
    pred_df.to_csv(os.path.join(CSV_DIR, "posterior_predictions.csv"), index=False)

    # ============================================================
    # 11. ROC CURVE
    # ============================================================
    fpr, tpr, _ = roc_curve(y_test, p_test_mean)
    roc_auc = roc_auc_score(y_test, p_test_mean)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Bayesian Logistic Regression — ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()

    # ============================================================
    # 12. PRECISION-RECALL CURVE
    # ============================================================
    precision, recall, _ = precision_recall_curve(y_test, p_test_mean)
    pr_auc = average_precision_score(y_test, p_test_mean)

    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Bayesian Logistic Regression — Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"))
    plt.close()

    # ============================================================
    # 13. COEFFICIENT INTERVAL PLOT
    # ============================================================
    coef_means = coef_summary["mean"].values
    coef_low = coef_summary["hdi_3%"].values
    coef_high = coef_summary["hdi_97%"].values

    plt.figure(figsize=(10, 12))
    plt.errorbar(coef_means, range(len(feature_cols)),
                 xerr=[coef_means - coef_low, coef_high - coef_means],
                 fmt="o")
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Posterior Coefficient Intervals")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "coefficient_intervals.png"))
    plt.close()

    # ============================================================
    # 14. PRINT METRICS
    # ============================================================
    print("\n=== BAYESIAN MODEL PERFORMANCE ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")


# ============================================================
# RUN MAIN
# ============================================================
if __name__ == "__main__":
    main()
