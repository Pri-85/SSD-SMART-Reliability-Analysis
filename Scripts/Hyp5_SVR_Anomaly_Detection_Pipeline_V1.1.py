import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA AND BUILD FAILURE LABEL
# ============================================================

data_path = "Step3-synthetic_smart_data_V1.1.csv"
df = pd.read_csv(data_path)

# Parse timestamp and sort
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Time in hours since start
df["time_hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0

# --- Failure label (same spirit as Hypothesis 4) ---
df["failure_label"] = (
    (df["composite_temperature_c"] >= 60) |
    (df["iops"] <= 10000) |
    (df["pcie_correctable_errors"] >= 5) |
    (df["pcie_uncorrectable_errors"] > 0) |
    (df["throttling_events"] >= 3) |
    (df["media_errors"] > 0) |
    (df["error_information_log_entries"] > 0) |
    (df["bad_block_count_grown"] > 0) |
    (df["unsafe_shutdowns"] > 0)
).astype(int)


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
    "power_draw_w",
]

# Keep only rows with all needed fields
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols].values
y = df_model[target_col].values

# "Healthy" training subset: no failure at this time and no catastrophic errors
healthy_mask = (
    (df_model["failure_label"] == 0) &
    (df_model["pcie_uncorrectable_errors"] == 0) &
    (df_model["media_errors"] <= 1) &
    (df_model["bad_block_count_grown"] <= 1)
)

print("Total samples:", len(df_model))
print("Healthy samples:", healthy_mask.sum())

if healthy_mask.sum() < 50:
    print("Warning: too few healthy samples – training SVR on full dataset.")
    X_healthy = X
    y_healthy = y
else:
    X_healthy = X[healthy_mask]
    y_healthy = y[healthy_mask]

# Train/test split for SVR
X_train, X_test, y_train, y_test = train_test_split(
    X_healthy, y_healthy, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
svr.fit(X_train_scaled, y_train)

# Predict on full dataset and compute anomaly score
X_all_scaled = scaler.transform(X)
y_pred_all = svr.predict(X_all_scaled)
anomaly_score = np.abs(y_pred_all - y)

df_model["svr_pred_temp"] = y_pred_all
df_model["svr_anomaly_score"] = anomaly_score

# Quick visualization (optional)
plt.figure(figsize=(12, 4))
plt.plot(df_model["timestamp"], df_model["svr_anomaly_score"], label="SVR anomaly score")
plt.xlabel("Time")
plt.ylabel("Anomaly score")
plt.title("SVR-based anomaly score over time")
plt.legend()
plt.tight_layout()
plt.savefig("svr_anomaly_score_over_time.png", dpi=300)
plt.close()

# Save SVR output
svr_output_path = "svr_anomaly_scores_output.csv"
df_model.to_csv(svr_output_path, index=False)
print(f"SVR anomaly scores saved to: {svr_output_path}")


# ============================================================
# 3. BUILD RL DATAFRAME (STATE + ACTION CONTEXT)
# ============================================================

# For RL we need: time-ordered series with:
# - anomaly score
# - workload
# - power draw
# - throttling events
# - failure_label (for reward)

rl_cols = [
    "timestamp",
    "time_hours",
    "svr_anomaly_score",
    "workload_type",
    "power_draw_w",
    "throttling_events",
    "iops",
    "bandwidth_read_gbps",
    "bandwidth_write_gbps",
    "percentage_used",
    "failure_label",
]

df_rl = df_model[rl_cols].dropna().copy()
df_rl = df_rl.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 4. RL ENVIRONMENT (SIMPLE GYM-LIKE CLASS)
# ============================================================

class SSDTelemetryEnv:
    """
    State: [anomaly_bucket, workload_index, power_bucket]
    Actions: 0=low telemetry, 1=medium, 2=high
    Reward:
      - cost penalty proportional to telemetry level
      - bonus for high telemetry when anomaly is high and failure is near
      - penalty for low telemetry before failures
    """

    def __init__(self, df, horizon=50):
        self.df = df
        self.horizon = horizon  # number of timesteps per episode
        self.n_actions = 3      # 0=low, 1=med, 2=high
        self.current_idx = 0

        # Encode workload as category
        self.workload_categories = {w: i for i, w in enumerate(df["workload_type"].astype(str).unique())}

    def reset(self):
        # Start episode at a random index that leaves enough horizon
        max_start = len(self.df) - self.horizon - 1
        if max_start <= 0:
            self.current_idx = 0
        else:
            self.current_idx = random.randint(0, max_start)
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_idx]
        anomaly = row["svr_anomaly_score"]
        power = row["power_draw_w"]
        workload = str(row["workload_type"])

        # Bucket anomaly (0=low, 1=medium, 2=high)
        if anomaly < 1.0:
            anomaly_bucket = 0
        elif anomaly < 3.0:
            anomaly_bucket = 1
        else:
            anomaly_bucket = 2

        # Bucket power
        if power < 5:
            power_bucket = 0
        elif power < 10:
            power_bucket = 1
        else:
            power_bucket = 2

        workload_idx = self.workload_categories.get(workload, 0)

        # Represent state as tuple of discrete values
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

        # Risk-based reward:
        # - If anomaly high and action high -> reward
        # - If anomaly high and action low -> big penalty
        # - If anomaly low and action high -> extra penalty (waste)
        reward = telemetry_cost

        if anomaly > 3.0:
            if action == 2:
                reward += 2.0   # good: high sampling under high risk
            elif action == 0:
                reward -= 2.0   # bad: low sampling under high risk
        elif anomaly < 1.0:
            if action == 2:
                reward -= 0.5   # unnecessary high sampling

        # Add a penalty if a failure happens and we were sampling low
        # (this is very simplistic, but illustrates the idea)
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

# Initialize Q-table
Q = np.zeros((n_anom, n_work, n_power, n_actions))

n_episodes = 200
alpha = 0.1     # learning rate
gamma = 0.95    # discount
epsilon = 0.2   # exploration

episode_rewards = []

for ep in range(n_episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        a_idx, w_idx, p_idx = state

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[a_idx, w_idx, p_idx, :])

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
plt.savefig("rl_training_rewards.png", dpi=300)
plt.close()

print("RL training completed. Average reward over last 20 episodes:",
      np.mean(episode_rewards[-20:]))
