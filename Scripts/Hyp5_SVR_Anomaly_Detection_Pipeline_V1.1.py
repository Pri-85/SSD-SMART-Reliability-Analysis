"""
Hypothesis 5: SVR-Based Anomaly Detection + Reinforcement Learning for Adaptive Telemetry Scheduling
Script Name: Hyp5_SVR_RL_Telemetry_Experiment_V1.1.py
Author: Priya Pooja Hariharan
Script Version: 1.0
Date: December 2025

Project: MIS581 – SSD SMART Reliability Analysis
Institution: Colorado State University – Global Campus
Course: MIS581 – Capstone Project

Description:
    This script implements the experimental pipeline for Hypothesis 5, which evaluates whether
    Support Vector Regression (SVR) combined with a Reinforcement Learning (RL) telemetry scheduler
    can outperform static threshold-based monitoring strategies in detecting SSD degradation.

    The experiment consists of:
        1. Loading and preprocessing SMART telemetry data.
        2. Constructing a composite throttling-events indicator based on thermal, workload, and
           endurance-related stress conditions.
        3. Building a domain-informed failure label consistent with prior hypotheses.
        4. Training an SVR model on healthy SSD intervals to learn expected behavior and compute
           anomaly scores.
        5. Constructing an RL environment that uses anomaly scores, workload, and power draw to
           determine optimal telemetry sampling frequency.
        6. Training a Q-learning agent to balance monitoring cost against early detection value.
        7. Saving all plots, anomaly score outputs, and RL training summaries to the designated
           results directory.

    Hypothesis 5:
        H₀: Adaptive telemetry scheduling using SVR + RL provides no improvement over fixed-interval
            monitoring strategies.
        H₁: SVR-driven anomaly detection combined with RL-based adaptive telemetry scheduling
            improves early detection capability while reducing unnecessary monitoring overhead.

    All outputs—including plots, CSV summaries, and Q-table snapshots—are saved to:
        C:/Users/venki/SSD-SMART-Reliability-Analysis/TestResults/hypothesis5_results

Notes:
    - This script assumes the cleaned SMART dataset from Step 2 is available at:
        C:/Users/venki/SSD-SMART-Reliability-Analysis/Data/Processed_V1.2/Step2-cleaned_SSD_dataset.csv
    - The SVR model is trained only on healthy intervals to capture baseline device behavior.
    - The RL agent uses a simplified discrete state space and Q-learning for interpretability.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 0. PATHS AND OUTPUT SETUP
# ============================================================

INPUT_PATH = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"
OUTPUT_DIR = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\hypothesis5_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA AND BUILD FAILURE LABEL
# ============================================================

df = pd.read_csv(INPUT_PATH)

# Parse timestamp and sort
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Time in hours since start
df["time_hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0

# ------------------------------------------------------------
# 1.1 DEFINE COMPOSITE THROTTLING EVENTS CONDITION
# ------------------------------------------------------------
# host_read_cmds_per_power_cycle = host_read_commands / power_cycles (safe for zero)
df["host_read_cmds_per_power_cycle"] = (
    df["host_read_commands"] / df["power_cycles"].replace(0, np.nan)
)

df["throttling_events"] = (
    (df["composite_temperature_c"] > 40) &
    (df["power_cycles"] > 200) &
    (df["controller_busy_time"] > 500) &
    (df["endurance_estimate_remaining"] > 100) &
    (df["host_read_cmds_per_power_cycle"] > 7_000_000)
).astype(int)

# ------------------------------------------------------------
# 1.2 UPDATED FAILURE LABEL (VERSION B, WITH CORRECT PARENTHESES)
# ------------------------------------------------------------
df["failure_label"] = (
    (df["composite_temperature_c"] >= 60) |
    ((df["iops"] <= 15000) & (df["pcie_correctable_errors"] >= 90)) |
    (df["pcie_uncorrectable_errors"] > 2) |
    (df["throttling_events"] == 1) |
    (df["media_errors"] > 20) |
    (df["error_information_log_entries"] > 15) |
    (df["bad_block_count_grown"] > 20) |
    (df["unsafe_shutdowns"] > 2)
).astype(int)

# Save a snapshot of failure-label summary
failure_summary_path = os.path.join(OUTPUT_DIR, "failure_label_summary.csv")
df["failure_label"].value_counts().rename_axis("failure_label").reset_index(name="count") \
  .to_csv(failure_summary_path, index=False)


# ============================================================
# 2. SVR ANOMALY DETECTION (ON HEALTHY INTERVALS)
# ============================================================

target_col = "composite_temperature_c"

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
    "host_read_cmds_per_power_cycle",
]

df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols].values
y = df_model[target_col].values

# Healthy subset: no failure_label and low-risk error indicators
healthy_mask = (
    (df_model["failure_label"] == 0) &
    (df_model["pcie_uncorrectable_errors"] == 0) &
    (df_model["media_errors"] <= 1) &
    (df_model["bad_block_count_grown"] <= 1)
)

print("Total samples for SVR model:", len(df_model))
print("Healthy samples for SVR training:", healthy_mask.sum())

if healthy_mask.sum() < 50:
    print("Warning: too few healthy samples – training SVR on full dataset.")
    X_healthy = X
    y_healthy = y
else:
    X_healthy = X[healthy_mask]
    y_healthy = y[healthy_mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_healthy, y_healthy, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
svr.fit(X_train_scaled, y_train)

# Predict on full dataset (df_model)
X_all_scaled = scaler.transform(X)
y_pred_all = svr.predict(X_all_scaled)
anomaly_score = np.abs(y_pred_all - y)

df_model["svr_pred_temp"] = y_pred_all
df_model["svr_anomaly_score"] = anomaly_score

# Save SVR outputs (full model frame)
svr_output_path = os.path.join(OUTPUT_DIR, "svr_anomaly_scores_output.csv")
df_model.to_csv(svr_output_path, index=False)
print(f"SVR anomaly scores saved to: {svr_output_path}")

# Plot anomaly score over time
plt.figure(figsize=(12, 4))
plt.plot(df_model["timestamp"], df_model["svr_anomaly_score"], label="SVR anomaly score")
plt.xlabel("Time")
plt.ylabel("Anomaly score")
plt.title("SVR-based anomaly score over time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svr_anomaly_score_over_time.png"), dpi=300)
plt.close()


# ============================================================
# 3. BUILD RL DATAFRAME (STATE + ACTION CONTEXT)
# ============================================================

rl_cols = [
    "timestamp",
    "time_hours",
    "svr_anomaly_score",
    "workload_type",
    "host_read_cmds_per_power_cycle",
    "throttling_events",
    "iops",
    "bandwidth_read_gbps",
    "bandwidth_write_gbps",
    "percentage_used",
    "failure_label",
]

df_rl = df_model[rl_cols].dropna().copy()
df_rl = df_rl.sort_values("timestamp").reset_index(drop=True)

# Save RL input snapshot
rl_input_path = os.path.join(OUTPUT_DIR, "rl_environment_input_snapshot.csv")
df_rl.head(500).to_csv(rl_input_path, index=False)


# ============================================================
# 4. RL ENVIRONMENT (SIMPLE GYM-LIKE CLASS)
# ============================================================

class SSDTelemetryEnv:
    """
    State: (anomaly_bucket, workload_index, power_bucket)
    Actions: 0=low telemetry, 1=medium, 2=high
    Reward:
      - cost penalty proportional to telemetry level
      - bonus for high telemetry when anomaly is high
      - penalty for low telemetry before failures
    """

    def __init__(self, df, horizon=50):
        self.df = df
        self.horizon = horizon   # number of timesteps per episode
        self.n_actions = 3       # 0=low, 1=med, 2=high
        self.current_idx = 0

        # Encode workload as categorical index
        self.workload_categories = {
            w: i for i, w in enumerate(df["workload_type"].astype(str).unique())
        }

    def reset(self):
        # Start at random index with enough room for horizon
        max_start = len(self.df) - self.horizon - 1
        if max_start <= 0:
            self.current_idx = 0
        else:
            self.current_idx = random.randint(0, max_start)
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_idx]
        anomaly = row["svr_anomaly_score"]
        power = row["host_read_cmds_per_power_cycle"]
        workload = str(row["workload_type"])

        # Bucket anomaly: 0=low, 1=medium, 2=high
        if anomaly < 1.0:
            anomaly_bucket = 0
        elif anomaly < 3.0:
            anomaly_bucket = 1
        else:
            anomaly_bucket = 2

        # Bucket power draw: 0=low, 1=medium, 2=high
        if power < 5:
            power_bucket = 0
        elif power < 10:
            power_bucket = 1
        else:
            power_bucket = 2

        workload_idx = self.workload_categories.get(workload, 0)

        return (anomaly_bucket, workload_idx, power_bucket)

    def step(self, action):
        """
        action: 0=low freq, 1=med, 2=high
        reward: combines cost and failure/coverage.
        """
        row = self.df.iloc[self.current_idx]
        anomaly = row["svr_anomaly_score"]
        failure = row["failure_label"]

        # Telemetry cost: higher for more aggressive sampling
        telemetry_cost = [0.0, -0.5, -1.0][action]
        reward = telemetry_cost

        # Risk-based reward:
        # - If anomaly high and action high -> reward
        # - If anomaly high and action low -> big penalty
        # - If anomaly low and action high -> small penalty (waste)
        if anomaly > 3.0:
            if action == 2:
                reward += 2.0   # good: high sampling under high risk
            elif action == 0:
                reward -= 2.0   # bad: low sampling under high risk
        elif anomaly < 1.0:
            if action == 2:
                reward -= 0.5   # unnecessary high sampling

        # Additional penalty if a failure occurs while sampling low
        if failure == 1 and action == 0:
            reward -= 3.0

        # Advance time
        self.current_idx += 1
        done = (self.current_idx >= len(self.df) - 1)

        next_state = self._get_state() if not done else None
        return next_state, reward, done, {}

    def n_states(self):
        # rough upper bound for discrete table:
        return (3, len(self.workload_categories), 3)


# ============================================================
# 5. SIMPLE Q-LEARNING AGENT
# ============================================================

env = SSDTelemetryEnv(df_rl, horizon=50)
n_anom, n_work, n_power = env.n_states()
n_actions = env.n_actions

# Initialize Q-table: [anomaly_bucket, workload_idx, power_bucket, action]
Q = np.zeros((n_anom, n_work, n_power, n_actions))

n_episodes = 200
alpha = 0.1     # learning rate
gamma = 0.95    # discount factor
epsilon = 0.2   # exploration rate

episode_rewards = []

for ep in range(n_episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        a_idx, w_idx, p_idx = state

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = int(np.argmax(Q[a_idx, w_idx, p_idx, :]))

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if not done:
            na_idx, nw_idx, np_idx = next_state
            best_next = np.max(Q[na_idx, nw_idx, np_idx, :])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[a_idx, w_idx, p_idx, action]
            Q[a_idx, w_idx, p_idx, action] += alpha * td_error

            state = next_state

    episode_rewards.append(total_reward)

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Q-learning training on SVR-based anomaly telemetry environment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rl_training_rewards.png"), dpi=300)
plt.close()

# Save Q-table summary as flattened CSV (for inspection & documentation)
Q_flat = Q.reshape(-1, n_actions)
q_table_path = os.path.join(OUTPUT_DIR, "q_table_summary.csv")
np.savetxt(q_table_path, Q_flat, delimiter=",")
print("Q-table summary saved to:", q_table_path)

# Save episode reward trace
episode_rewards_path = os.path.join(OUTPUT_DIR, "rl_episode_rewards.csv")
pd.DataFrame({"episode": np.arange(len(episode_rewards)),
              "total_reward": episode_rewards}).to_csv(episode_rewards_path, index=False)

print("RL training completed. Average reward over last 20 episodes:",
      np.mean(episode_rewards[-20:]))
print("All Hypothesis 5 results saved to:", OUTPUT_DIR)
