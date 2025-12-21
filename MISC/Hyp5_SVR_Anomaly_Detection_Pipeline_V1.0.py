import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA
# ============================================================

data_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"   # <-- update this
df = pd.read_csv(data_path)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Convert timestamp to numeric time (hours since start)
df["time_hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0


# ============================================================
# 2. SELECT FEATURES FOR SVR
# ============================================================

# Target variable to model "healthy" behavior
target_col = "composite_temperature_c"

# Feature set (you can expand or refine)
feature_cols = [
    "time_hours",
    "overprovisioning_ratio",
    "data_units_read",
    "data_units_written",
    "host_read_commands",
    "host_write_commands",
    "avg_queue_depth",
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
    "background_scrub_time_pct",
    "gc_active_time_pct",
    "power_draw_w",
]

# Drop missing rows
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols].values
y = df_model[target_col].values


# ============================================================
# 3. FILTER HEALTHY DATA FOR TRAINING
# ============================================================

healthy_mask = (
    (df_model["media_errors"] == 0) &
    (df_model["error_information_log_entries"] == 0) &
    (df_model["bad_block_count_grown"] == 0) &
    (df_model["pcie_uncorrectable_errors"] == 0) &
    (df_model["throttling_events"] == 0)
)

X_healthy = X[healthy_mask]
y_healthy = y[healthy_mask]


# ============================================================
# 4. TRAIN/TEST SPLIT + SCALING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_healthy, y_healthy, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. TRAIN SVR MODEL
# ============================================================

svr = SVR(
    kernel="rbf",
    C=1.0,
    epsilon=0.1,
    gamma="scale"
)

svr.fit(X_train_scaled, y_train)


# ============================================================
# 6. COMPUTE ANOMALY SCORES ON FULL DATASET
# ============================================================

X_all_scaled = scaler.transform(X)
y_pred_all = svr.predict(X_all_scaled)

# Absolute error as anomaly score
anomaly_score = np.abs(y_pred_all - y)

df_model["svr_pred_temp"] = y_pred_all
df_model["svr_anomaly_score"] = anomaly_score


# ============================================================
# 7. VISUALIZE ANOMALY SCORE OVER TIME
# ============================================================

plt.figure(figsize=(14, 5))
plt.plot(df_model["timestamp"], df_model["svr_anomaly_score"], label="SVR Anomaly Score")
plt.xlabel("Time")
plt.ylabel("Anomaly Score (|pred - actual|)")
plt.title("SVR-Based Anomaly Score Over Time")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. SAVE OUTPUT FOR RL PIPELINE
# ============================================================

output_path = "svr_anomaly_scores_output.csv"
df_model.to_csv(output_path, index=False)

print(f"SVR anomaly detection completed. Output saved to: {output_path}")
