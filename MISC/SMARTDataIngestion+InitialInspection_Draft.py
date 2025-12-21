"""
SSD SMART Dataset – Data Ingestion and Initial Inspection Script
Author: Priya
Purpose:
    This script performs the first phase of the SSD reliability analysis pipeline:
    Data Ingestion and Initial Exploratory Inspection. It loads the raw SMART dataset,
    standardizes column names, identifies metadata vs. telemetry attributes, converts
    numeric fields, evaluates missingness, and generates summary tables and visualizations.
    This step establishes the structural understanding required for normalization,
    feature engineering, and predictive modeling in later phases.

Workflow:
    1. Import required libraries (Pandas, NumPy, Seaborn, Matplotlib).
    2. Load the raw SMART dataset from CSV/Excel.
    3. Clean and standardize column names for consistency.
    4. Categorize columns into metadata, SMART health attributes, usage counters,
       and latency/performance metrics.
    5. Convert numeric fields (including scientific notation) to proper numeric types.
    6. Generate dataset-level summaries (shape, unique devices, attribute counts).
    7. Analyze missingness patterns and create an attribute overview table.
    8. Produce correlation heatmaps for telemetry attributes.
    9. Generate an automated EDA profiling report (HTML).

Tools Used:
    - Python 3.x
    - Pandas for data ingestion and cleaning
    - NumPy for numerical operations
    - Seaborn & Matplotlib for visualization
    - ydata-profiling for automated EDA reporting

Output:
    - Cleaned DataFrame in memory
    - Missingness summary
    - Attribute overview table
    - Correlation heatmap
    - Automated HTML EDA report saved to /reports/

This script is part of the reproducible SSD reliability benchmarking framework.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ============================================================
# 2. LOAD DATASET
# ============================================================
df = pd.read_csv("data/raw/ssd_smart_snapshot.csv")
print("✅ Dataset Loaded:", df.shape)
print(df.head())

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("%", "percent")
)

print("\n✅ Cleaned Column Names:")
print(df.columns.tolist())

# ============================================================
# 4. IDENTIFY COLUMN GROUPS
# ============================================================
metadata_cols = [
    "system_ma", "system_co", "cpu_name", "model_num", "serial_num",
    "system_id", "pc_vendor", "pc_name", "firmware", "nand_type",
    "nvme_cap", "namespaces", "namespace", "ieee_oui", "nvme_vent",
    "controller", "local_time"
]

smart_cols = [
    "temperatu", "temperatu_1", "available_s", "available_s_percentage",
    "percentage", "power_cyc", "power_on", "unsafe_sh",
    "media_and_error", "infor", "infor_warning", "c_critical"
]

usage_cols = [
    "data_units", "data_units_host", "read_host", "write_controller",
    "max_i/o", "i/o_pa", "log_page_s"
]

latency_cols = ["read_late", "write_late"]

metadata_cols = [c for c in metadata_cols if c in df.columns]
smart_cols = [c for c in smart_cols if c in df.columns]
usage_cols = [c for c in usage_cols if c in df.columns]
latency_cols = [c for c in latency_cols if c in df.columns]

print("\n✅ Column Groups Identified:")
print("Metadata:", metadata_cols)
print("SMART:", smart_cols)
print("Usage:", usage_cols)
print("Latency:", latency_cols)

# ============================================================
# 5. CONVERT NUMERIC COLUMNS
# ============================================================
numeric_cols = smart_cols + usage_cols + latency_cols

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n✅ Numeric Conversion Complete")
print(df[numeric_cols].describe())

# ============================================================
# 6. BASIC DATASET SUMMARY
# ============================================================
print("\n✅ Basic Dataset Summary:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

if "serial_num" in df.columns:
    print("Unique SSDs:", df["serial_num"].nunique())

# ============================================================
# 7. MISSINGNESS ANALYSIS
# ============================================================
missing_summary = (
    df.isna().mean().sort_values(ascending=False) * 100
).to_frame("missing_percent")

print("\n✅ Missingness Summary:")
print(missing_summary.head(20))

# ============================================================
# 8. ATTRIBUTE OVERVIEW TABLE
# ============================================================
telemetry_cols = smart_cols + usage_cols + latency_cols

attribute_overview = pd.DataFrame({
    "attribute": telemetry_cols,
    "non_null_count": df[telemetry_cols].notna().sum().values,
    "missing_percent": df[telemetry_cols].isna().mean().values * 100,
    "dtype": [df[c].dtype for c in telemetry_cols]
})

print("\n✅ Attribute Overview:")
print(attribute_overview)

# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================
plt.figure(figsize=(12, 8))
sns.heatmap(df[telemetry_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of SMART Telemetry")
plt.show()

# ============================================================
# 10. AUTOMATED EDA REPORT (HTML)
# ============================================================
profile = ProfileReport(df, title="SMART Dataset Profile", explorative=True)
profile.to_file("reports/smart_dataset_profile.html")

print("\n✅ Automated EDA Report Generated: reports/smart_dataset_profile.html")
