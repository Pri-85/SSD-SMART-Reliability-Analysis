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
# File paths (update if needed)
# ---------------------------------------------------------
before_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\cleaned_SSD_dataset.csv"
after_path  = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step2-cleaned_SSD_dataset.csv"

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
df_before = pd.read_csv(before_path)
df_after  = pd.read_csv(after_path)

print("Datasets loaded successfully!")

# ---------------------------------------------------------
# 1. Missingness Bar Chart
# ---------------------------------------------------------
missing_before = df_before.isnull().sum()
missing_after = df_after.isnull().sum()

missing_df = pd.DataFrame({
    "Before Preprocessing": missing_before,
    "After Preprocessing": missing_after
})

plt.figure(figsize=(14, 6))
missing_df.plot(kind="bar", figsize=(14,6))
plt.title("Missing Values Before vs After Preprocessing")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Missingness Heatmap
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(df_before.isnull(), cbar=False, ax=axes[0])
axes[0].set_title("Missingness Heatmap - Before")

sns.heatmap(df_after.isnull(), cbar=False, ax=axes[1])
axes[1].set_title("Missingness Heatmap - After")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Boxplots Before vs After
# ---------------------------------------------------------
numeric_cols = df_after.select_dtypes(include=['float64', 'int64']).columns

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(data=df_before[numeric_cols], ax=axes[0])
axes[0].set_title("Boxplot - Before Preprocessing")
axes[0].tick_params(axis='x', rotation=90)

sns.boxplot(data=df_after[numeric_cols], ax=axes[1])
axes[1].set_title("Boxplot - After Preprocessing")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

print("All visualizations generated successfully!")
