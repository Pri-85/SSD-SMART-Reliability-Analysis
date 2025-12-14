# ============================================================
# SSD SMART Dataset – Ingestion, Inspection, and Smart SSD Analysis
# Author: Priya
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# ------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------

INPUT_PATH = r"C:\Users\venki\Desktop\MS_Data_Anayltics\MIS581-Capstone\Test\MIS581_SSD_dataset.csv"

CLEANED_CSV_PATH = "cleaned_SSD_dataset.csv"
NUMERIC_SUMMARY_PATH = "numeric_summary_stats.csv"
CORR_MATRIX_PATH = "correlation_matrix_numeric.csv"
CORR_LATENCY_PATH = "correlation_latency_focus.csv"
CORR_POWER_PATH = "correlation_power_focus.csv"
CORR_RW_PATH = "correlation_rw_focus.csv"
MISSINGNESS_PATH = "missingness_summary.csv"
ATTR_OVERVIEW_PATH = "attribute_overview.csv"
HEATMAP_ALL_PATH = "corr_heatmap_numeric.png"
HEATMAP_LATENCY_PATH = "corr_heatmap_latency_focus.png"
HEATMAP_VENDOR_LAT_TREND_PATH = "latency_trend_by_vendor.png"
EDA_REPORT_PATH = "SSD_SMART_EDA_Report.html"

HOST_INTERFACE_MBPS = 375.0  # reference throughput


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, low_memory=False)
print("\n✅ Dataset loaded:", df.shape)


# ------------------------------------------------------------
# 2. Standardize column names
# ------------------------------------------------------------

df.columns = (
    df.columns.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("-", "_")
              .str.replace("/", "_")
)

print("\n✅ Column names standardized")


# ------------------------------------------------------------
# 3. SMART Telemetry Grouping
# ------------------------------------------------------------

def contains_any(col, keywords):
    return any(k in col for k in keywords)

metadata_cols = [c for c in df.columns if contains_any(c, [
    "system_manufacturer", "system_configuration", "cpu_name",
    "model_number", "serial_number", "system_id",
    "pci_vendor_id", "pci_vendor_subsystem_id",
    "firmware_revision", "ieee_oui", "vendor", "ff"
])]

capacity_cols = [c for c in df.columns if contains_any(c, [
    "nvme_capacity", "unallocated", "namespace_size",
    "namespace_count", "controller_count", "nvme_version",
    "nand_type", "capacity"
])]

time_cols = [c for c in df.columns if contains_any(c, [
    "local_time", "timestamp", "date", "time", "year"
])]

temp_cols = [c for c in df.columns if contains_any(c, [
    "temperature_warning", "temperature_critical", "temperature_current",
    "warning_comp_temperature_time", "critical_comp_temperature_time"
])]

wear_health_cols = [c for c in df.columns if contains_any(c, [
    "available_spare", "available_spare_threshold", "percentage_used",
    "media_and_data_integrity_errors", "error_information_log_entries"
])]

workload_rw_cols = [c for c in df.columns if contains_any(c, [
    "data_units_read", "data_units_written",
    "host_read_commands", "host_write_commands",
    "iops", "throughput"
])]

power_cols = [c for c in df.columns if contains_any(c, [
    "power_cycles", "power_on_hours", "power", "energy", "watt"
])]

latency_cols = [c for c in df.columns if contains_any(c, [
    "read_latency", "write_latency", "latency"
])]

reliability_event_cols = [c for c in df.columns if contains_any(c, [
    "unsafe_shutdowns", "controller_busy_time"
])]

feature_flag_cols = [c for c in df.columns if contains_any(c, [
    "optional_features", "log_page_support", "max_i_o_pages"
])]

print("\n✅ SMART telemetry groups identified")


# ------------------------------------------------------------
# 4. Convert numeric fields
# ------------------------------------------------------------

non_numeric_like = set(metadata_cols + time_cols)
numeric_candidates = [c for c in df.columns if c not in non_numeric_like]

for col in numeric_candidates:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n✅ Numeric conversion complete")


# ------------------------------------------------------------
# 5. Missingness + Attribute Overview
# ------------------------------------------------------------

missing_summary = df.isna().mean().sort_values(ascending=False)
missing_summary.to_frame("missing_fraction").to_csv(MISSINGNESS_PATH)

attribute_overview = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.values,
    "missing_pct": df.isna().mean().values,
    "unique_values": df.nunique().values
})
attribute_overview.to_csv(ATTR_OVERVIEW_PATH, index=False)

print("\n✅ Missingness + attribute overview saved")


# ------------------------------------------------------------
# 6. Numeric Summary Statistics
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=[np.number])

basic_desc = numeric_df.describe().T

extra_stats = pd.DataFrame(index=numeric_df.columns)
extra_stats["missing_pct"] = numeric_df.isna().mean()
extra_stats["zero_pct"] = (numeric_df == 0).mean()
extra_stats["skew"] = numeric_df.skew()
extra_stats["kurtosis"] = numeric_df.kurtosis()

numeric_summary = basic_desc.join(extra_stats)
numeric_summary.to_csv(NUMERIC_SUMMARY_PATH)

print("\n✅ Numeric summary statistics saved")


# ------------------------------------------------------------
# 7. Correlation Matrices
# ------------------------------------------------------------

corr_all = numeric_df.corr()
corr_all.to_csv(CORR_MATRIX_PATH)

plt.figure(figsize=(18, 14))
sns.heatmap(corr_all, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap – All Numeric Fields")
plt.tight_layout()
plt.savefig(HEATMAP_ALL_PATH)
plt.close()

print("\n✅ Full correlation heatmap saved")


# Latency-focused correlation
latency_focus_cols = list(set(latency_cols + workload_rw_cols + power_cols + wear_health_cols))
latency_focus_df = numeric_df[latency_focus_cols].dropna(axis=1, how="all")

corr_latency = latency_focus_df.corr()
corr_latency.to_csv(CORR_LATENCY_PATH)

plt.figure(figsize=(14, 10))
sns.heatmap(corr_latency, cmap="coolwarm", center=0)
plt.title("Latency-Focused Correlation")
plt.tight_layout()
plt.savefig(HEATMAP_LATENCY_PATH)
plt.close()

print("\n✅ Latency-focused correlation heatmap saved")


# ------------------------------------------------------------
# 8. Vendor-wise Latency Trends (Year Basis)
# ------------------------------------------------------------

vendor_candidates = [c for c in metadata_cols if "vendor" in c or "manufacturer" in c]
vendor_col = vendor_candidates[0] if vendor_candidates else None

year_col = None
for c in time_cols:
    try:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        year_col = c
        break
    except Exception:
        continue

if year_col is not None:
    df["year"] = df[year_col].dt.year

if vendor_col and latency_cols and "year" in df.columns:
    lat_col = latency_cols[0]

    trend_df = (
        df.dropna(subset=[vendor_col, "year", lat_col])
          .groupby([vendor_col, "year"])[lat_col]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=trend_df, x="year", y=lat_col, hue=vendor_col, marker="o")
    plt.title(f"Latency Trend by Vendor Over Years ({lat_col})")
    plt.ylabel("Average Latency (ms)")
    plt.tight_layout()
    plt.savefig(HEATMAP_VENDOR_LAT_TREND_PATH)
    plt.close()

    print("\n✅ Vendor latency trend saved")
else:
    print("\n⚠️ Vendor latency trend not generated (missing vendor/year/latency)")


# ------------------------------------------------------------
# 9. Smart SSD Offload & Energy Metrics
# ------------------------------------------------------------

if "power_cycles" in numeric_df.columns and "host_read_commands" in numeric_df.columns:
    df["host_read_cmds_per_power_cycle"] = (
        df["host_read_commands"] / df["power_cycles"].replace(0, np.nan)
    )

if "power_on_hours" in numeric_df.columns and "media_and_data_integrity_errors" in numeric_df.columns:
    df["errors_per_power_on_hour"] = (
        df["media_and_data_integrity_errors"] / df["power_on_hours"].replace(0, np.nan)
    )

if "percentage_used" in numeric_df.columns and "data_units_written" in numeric_df.columns:
    df["wear_per_data_written"] = (
        df["percentage_used"] / df["data_units_written"].replace(0, np.nan)
    )

print("\n✅ Smart SSD offload/energy metrics computed")


# ------------------------------------------------------------
# 10. Safe HTML Profiling Report (Minimal Mode)
# ------------------------------------------------------------

from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    minimal=True,          # avoids heavy computations
    explorative=False,     # keeps it stable
)

profile.to_file("SSD_SMART_EDA_Report.html")
print("\n✅ Safe HTML EDA report generated")


# ------------------------------------------------------------
# 11. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(CLEANED_CSV_PATH, index=False)
print("\n✅ Cleaned dataset saved")
