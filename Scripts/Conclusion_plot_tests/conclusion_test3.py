# ---------------------------------------------------------
# FIX: Disable PyArrow so pandas doesn't try to import it
# ---------------------------------------------------------
import os
os.environ["PANDAS_IGNORE_ARROW"] = "1"

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------
data_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")

# ---------------------------------------------------------
# Convert timestamp to datetime
# ---------------------------------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop rows with invalid timestamps
df = df.dropna(subset=['timestamp'])

# ---------------------------------------------------------
# Choose KPI to trend
# ---------------------------------------------------------
# Examples you can choose:
# 'iops', 'bandwidth_read_gbps', 'bandwidth_write_gbps',
# 'io_completion_time_ms', 'percentage_used', 'wear_level_avg',
# 'media_errors', 'error_information_log_entries'

kpi = 'iops'   # <-- change this to any KPI you want

# ---------------------------------------------------------
# Aggregate KPI by month
# ---------------------------------------------------------
df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()

kpi_trend = df.groupby('month')[kpi].mean().reset_index()

# ---------------------------------------------------------
# Plot KPI Trend Line
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(data=kpi_trend, x='month', y=kpi, marker='o')

plt.title(f"Trend of {kpi} Over Time")
plt.xlabel("Time (Monthly)")
plt.ylabel(f"{kpi} Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("KPI Trend Line generated successfully!")
