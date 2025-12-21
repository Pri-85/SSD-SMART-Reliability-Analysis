# =============================================================================
# SSD SMART Dataset – Ingestion, Cleaning, Inspection, and Smart SSD Analysis
# Author: Priya Pooja Hariharan
# Script Version: 1.3
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 
# ==============================================================================

import os
import random
import string

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend

import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from ydata_profiling import ProfileReport

plt.style.use("seaborn-v0_8")

# ------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------

INPUT_PATH = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\Step3-synthetic_smart_data_V1.1.csv"

CLEANED_CSV_PATH = "cleaned_SSD_dataset.csv"
NUMERIC_SUMMARY_PATH = "numeric_summary_stats.csv"
CORR_MATRIX_PATH = "correlation_matrix_numeric.csv"
CORR_LATENCY_PATH = "correlation_latency_focus.csv"
MISSINGNESS_PATH = "missingness_summary.csv"
ATTR_OVERVIEW_PATH = "attribute_overview.csv"
HEATMAP_ALL_PATH = "corr_heatmap_numeric.png"
HEATMAP_LATENCY_PATH = "corr_heatmap_latency_focus.png"
HEATMAP_VENDOR_LAT_TREND_PATH = "latency_trend_by_vendor.png"
EDA_REPORT_PATH = "SSD_SMART_EDA_Report.html"

# Visualization outputs (Option B & C)
HEATMAP_THERMAL_WORKLOAD_PATH = "corr_thermal_vs_workload.png"
HEATMAP_RELIABILITY_WORKLOAD_PATH = "corr_reliability_vs_workload.png"
BOXPLOT_THERMAL_BY_WORKLOAD_PATH = "boxplot_thermal_by_workload.png"
BAR_HEALTH_BY_POWER_BIN_PATH = "bar_health_by_power_bins.png"

# Categorical plots outputs
CATEGORICAL_FREQ_DIR = "categorical_frequency_plots"
CATEGORICAL_BOX_DIR = "categorical_boxplots"

os.makedirs(CATEGORICAL_FREQ_DIR, exist_ok=True)
os.makedirs(CATEGORICAL_BOX_DIR, exist_ok=True)

HOST_INTERFACE_MBPS = 375.0  # reference throughput


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, low_memory=False)
print("\nDataset loaded:", df.shape)


# ------------------------------------------------------------
# 1B. Inject synthetic system metadata + inspection logic
# ------------------------------------------------------------

# Template records (manufacturer, configuration, CPU)
template_records = [
    ("HPE",        "HPE NF5280M6",                 "Intel"),
    ("Inspur",     "Inspur ThinkSystem SR650",     "Intel"),
    ("Fujitsu",    "Fujitsu UCS C240",            "Intel"),
    ("SuperMicro", "SuperMicro ThinkSystem SR650", "AMD"),
    ("Inspur",     "Inspur NF5280M6",             "AMD"),
    ("Dell",       "Dell ThinkSystem SR650",      "AMD"),
    ("Dell",       "Dell ThinkSystem SR650",      "AMD"),
    ("SuperMicro", "SuperMicro ProLiant DL380",   "Intel"),
    ("Lenovo",     "Lenovo R740",                 "AMD"),
    ("HPE",        "HPE UCS C240",                "Intel"),
]

manufacturers = list({m for m, _, _ in template_records})

def random_serial():
    return "ABCD" + "".join(random.choices(string.digits, k=4)) + "EFGH"

def random_sysid():
    return "D3V" + "".join(random.choices(string.digits, k=3)) + "VAB"

def random_model_number():
    prefix = random.choice(["NV", "PM", "XG", "MT"])
    return prefix + "".join(random.choices(string.digits, k=4))

# manufacturer → allowed configs / cpus
config_map = {}
cpu_map = {}
for m, cfg, cpu in template_records:
    config_map.setdefault(m, set()).add(cfg)
    cpu_map.setdefault(m, set()).add(cpu)

system_manufacturer = []
system_configuration = []
cpu_name = []
model_number = []
serial_number = []
system_id = []

for _ in range(len(df)):
    m = random.choice(manufacturers)
    cfg = random.choice(list(config_map[m]))
    cpu = random.choice(list(cpu_map[m]))

    system_manufacturer.append(m)
    system_configuration.append(cfg)
    cpu_name.append(cpu)
    model_number.append(random_model_number())
    serial_number.append(random_serial())
    system_id.append(random_sysid())

df["system_manufacturer"] = system_manufacturer
df["system_configuration"] = system_configuration
df["cpu_name"] = cpu_name
df["model_number"] = model_number
df["serial_number"] = serial_number
df["system_id"] = system_id

print("\nSynthetic system metadata injected")

# Metadata inspection
inspection_report = {}

inspection_report["duplicate_model_numbers"] = df["model_number"].duplicated().sum()
inspection_report["duplicate_serial_numbers"] = df["serial_number"].duplicated().sum()
inspection_report["duplicate_system_ids"] = df["system_id"].duplicated().sum()

invalid_config = df[
    ~df.apply(
        lambda r: r["system_configuration"] in config_map.get(r["system_manufacturer"], set()),
        axis=1,
    )
]
inspection_report["invalid_config_rows"] = len(invalid_config)

invalid_cpu = df[
    ~df.apply(
        lambda r: r["cpu_name"] in cpu_map.get(r["system_manufacturer"], set()),
        axis=1,
    )
]
inspection_report["invalid_cpu_rows"] = len(invalid_cpu)

inspection_report["manufacturer_distribution"] = df["system_manufacturer"].value_counts().to_dict()
inspection_report["cpu_distribution"] = df["cpu_name"].value_counts().to_dict()

pd.DataFrame.from_dict(inspection_report, orient="index").to_csv("metadata_inspection_report.csv")

print("\nMetadata inspection completed. Summary:")
for k, v in inspection_report.items():
    print(f"  {k}: {v}")


# ------------------------------------------------------------
# 2. Standardize column names
# ------------------------------------------------------------

df.columns = (
    df.columns.astype(str)
              .str.strip()
              .str.replace(" ", "_")
              .str.replace("-", "_")
              .str.replace("/", "_")
              .str.lower()
)

print("\nColumn names standardized")


# ------------------------------------------------------------
# 3. SMART telemetry grouping
# ------------------------------------------------------------

def contains_any(col, keywords):
    return any(k in col for k in keywords)

# Metadata
metadata_cols = [c for c in df.columns if contains_any(c, [
    "system_manufacturer", "system_configuration", "cpu_name",
    "model_number", "serial_number", "system_id",
    "pci_vendor_id", "pci_vendor_subsystem_id",
    "firmware_revision", "ieee_oui", "vendor", "ff"
])]

# Capacity / geometry
capacity_cols = [c for c in df.columns if contains_any(c, [
    "nvme_capacity", "unallocated", "namespace_size",
    "namespace_count", "controller_count", "nvme_version",
    "nand_type", "capacity"
])]

# Time-related columns
time_cols = [c for c in df.columns if contains_any(c, [
    "local_time", "timestamp", "date", "time", "datetime"
])]

# Thermal and health
temp_cols = [c for c in df.columns if contains_any(c, [
    "temperature_warning", "temperature_critical", "temperature_current"
])]

health_cols = [c for c in df.columns if contains_any(c, [
    "available_spare", "available_spare_threshold", "percentage_used"
])]

# Feature flags
feature_flag_cols = [c for c in df.columns if contains_any(c, [
    "optional_features", "log_page_support", "max_i_o_pages"
])]

# Workload / usage
workload_rw_cols = [c for c in df.columns if contains_any(c, [
    "data_units_read", "data_units_written",
    "host_read_commands", "host_write_commands"
])]

# Power / lifetime
power_cols = [c for c in df.columns if contains_any(c, [
    "power_cycles", "power_on_hours"
])]

# Latency
latency_cols = [c for c in df.columns if contains_any(c, [
    "read_latency", "write_latency", "latency"
])]

# Reliability events
reliability_event_cols = [c for c in df.columns if contains_any(c, [
    "unsafe_shutdowns", "controller_busy_time",
    "media_and_data_integrity_errors",
    "error_information_log_entries"
])]

# Non-numeric forced exclusions
force_non_numeric = [
    "ff",
    "warning_comp_temperature_time",
    "critical_comp_temperature_time"
]

print("\nSMART telemetry groups identified")


# ------------------------------------------------------------
# 4. Date handling (Local Time -> year, month, treated as categorical)
# ------------------------------------------------------------

local_time_col = "local_time" if "local_time" in df.columns else None

if local_time_col:
    df[local_time_col] = pd.to_datetime(df[local_time_col], errors="coerce", utc=True)
    df["year"] = df[local_time_col].dt.year
    df["month"] = df[local_time_col].dt.month
    time_cols.extend(["year", "month"])
    print("\nParsed Local Time and extracted year, month")
else:
    print("\nLocal Time column not found; year/month not extracted")


# ------------------------------------------------------------
# 5. Convert numeric fields (exclude metadata, time, forced non-numeric, year, month)
# ------------------------------------------------------------

non_numeric_like = set(metadata_cols + time_cols + force_non_numeric + ["year", "month"])
numeric_candidates = [c for c in df.columns if c not in non_numeric_like]

for col in numeric_candidates:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nNumeric conversion complete")


# ------------------------------------------------------------
# 6. Missingness analysis
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

print("\nMissingness and attribute overview saved")

missing_indicators = df[numeric_candidates].isna().astype(int)
missing_corr = missing_indicators.corr()  # in-memory only


# ------------------------------------------------------------
# 7. MICE imputation (remove all-NaN numeric columns first)
# ------------------------------------------------------------

numeric_candidates = [
    c for c in numeric_candidates
    if df[c].notna().sum() > 0
]

numeric_df = df[numeric_candidates]

imputer = IterativeImputer(max_iter=10, random_state=42)
numeric_imputed = pd.DataFrame(
    imputer.fit_transform(numeric_df),
    columns=numeric_candidates
)

df[numeric_candidates] = numeric_imputed

print("\nMICE imputation completed for numeric fields")


# ------------------------------------------------------------
# 8. Domain validation
# ------------------------------------------------------------

for col in numeric_candidates:
    df[col] = df[col].clip(lower=0)

if "temperature_current" in df.columns:
    df["temperature_current"] = df["temperature_current"].clip(0, 120)

if "percentage_used" in df.columns:
    df["percentage_used"] = df["percentage_used"].clip(0, 100)
if "available_spare" in df.columns:
    df["available_spare"] = df["available_spare"].clip(0, 100)
if "available_spare_threshold" in df.columns:
    df["available_spare_threshold"] = df["available_spare_threshold"].clip(0, 100)

print("\nDomain validation applied to imputed values")


# ------------------------------------------------------------
# 9. Numeric summary statistics
# ------------------------------------------------------------

numeric_df = df[numeric_candidates]

basic_desc = numeric_df.describe().T

extra_stats = pd.DataFrame(index=numeric_df.columns)
extra_stats["missing_pct"] = df[numeric_candidates].isna().mean()
extra_stats["zero_pct"] = (numeric_df == 0).mean()
extra_stats["skew"] = numeric_df.skew()
extra_stats["kurtosis"] = numeric_df.kurtosis()

numeric_summary = basic_desc.join(extra_stats)
numeric_summary.to_csv(NUMERIC_SUMMARY_PATH)

print("\nNumeric summary statistics saved")


# ------------------------------------------------------------
# 10. Correlation matrices
# ------------------------------------------------------------

corr_all = numeric_df.corr()
corr_all.to_csv(CORR_MATRIX_PATH)

plt.figure(figsize=(18, 14))
sns.heatmap(corr_all, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap – All Numeric Fields")
plt.tight_layout()
plt.savefig(HEATMAP_ALL_PATH)
plt.close()

print("\nFull numeric correlation heatmap saved")

latency_focus_cols = list(set(latency_cols + workload_rw_cols + power_cols + health_cols + temp_cols))
latency_focus_cols = [c for c in latency_focus_cols if c in numeric_df.columns]
latency_focus_df = numeric_df[latency_focus_cols].copy()

if not latency_focus_df.empty:
    corr_latency = latency_focus_df.corr()
    corr_latency.to_csv(CORR_LATENCY_PATH)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_latency, cmap="coolwarm", center=0)
    plt.title("Latency-Focused Correlation")
    plt.tight_layout()
    plt.savefig(HEATMAP_LATENCY_PATH)
    plt.close()

    print("\nLatency-focused correlation heatmap saved")
else:
    print("\nNo latency-focused numeric columns available; skipping latency correlation")


# ------------------------------------------------------------
# 11. Vendor-wise latency trends
# ------------------------------------------------------------

vendor_candidates = [c for c in metadata_cols if "vendor" in c or "manufacturer" in c]
vendor_col = vendor_candidates[0] if vendor_candidates else None

lat_col = next((c for c in latency_cols if c in df.columns), None)

if vendor_col and lat_col:
    df_vendor_lat = df[[vendor_col, lat_col, "year"]].dropna()

    if not df_vendor_lat.empty:

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_vendor_lat, x=vendor_col, y=lat_col)
        plt.title(f"Latency Distribution by Vendor ({lat_col})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("latency_boxplot_by_vendor.png")
        plt.close()

        vendor_mean = df_vendor_lat.groupby(vendor_col)[lat_col].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=vendor_mean, x=vendor_col, y=lat_col)
        plt.title(f"Mean Latency by Vendor ({lat_col})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("latency_bar_by_vendor.png")
        plt.close()

        if df_vendor_lat["year"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_vendor_lat, x="year", y=lat_col)
            plt.title(f"Latency Distribution by Year ({lat_col})")
            plt.tight_layout()
            plt.savefig("latency_boxplot_by_year.png")
            plt.close()

            year_mean = df_vendor_lat.groupby("year")[lat_col].mean().reset_index()

            plt.figure(figsize=(10, 6))
            sns.barplot(data=year_mean, x="year", y=lat_col)
            plt.title(f"Mean Latency by Year ({lat_col})")
            plt.tight_layout()
            plt.savefig("latency_bar_by_year.png")
            plt.close()

            print("\nVendor/year latency boxplots and bar charts saved")
        else:
            print("\nOnly one year present — skipping year-based plots")

    else:
        print("\nVendor-latency dataset empty; skipping vendor latency plots")

else:
    print("\nVendor or latency column missing; skipping vendor latency plots")


# ------------------------------------------------------------
# 12. Smart SSD metrics
# ------------------------------------------------------------

if "power_cycles" in df.columns and "host_read_commands" in df.columns:
    df["host_read_cmds_per_power_cycle"] = (
        df["host_read_commands"] / df["power_cycles"].replace(0, np.nan)
    )

print("\nSmart SSD utilization metric computed (host_read_cmds_per_power_cycle)")


# ------------------------------------------------------------
# 13. Visualizations – Option B (grouped, auto-binned)
# ------------------------------------------------------------

thermal_health_cols = [c for c in temp_cols + health_cols if c in df.columns]
workload_reliability_cols = [c for c in workload_rw_cols + power_cols + ["unsafe_shutdowns"] if c in df.columns]

def add_bins(series, bins=4):
    s = series.dropna()

    if s.empty:
        return pd.Series(["bin_0"] * len(series), index=series.index)

    if s.nunique() == 1:
        return pd.Series(["bin_0"] * len(series), index=series.index)

    unique_vals = s.nunique()
    effective_bins = min(bins, unique_vals)

    try:
        return pd.qcut(series, q=effective_bins, duplicates="drop")
    except Exception:
        pass

    try:
        return pd.cut(series, bins=effective_bins)
    except Exception:
        return pd.Series(["bin_0"] * len(series), index=series.index)


if thermal_health_cols and workload_reliability_cols:
    primary_thermal = next(
        (c for c in ["temperature_current", "temperature_warning", "temperature_critical"] if c in thermal_health_cols),
        thermal_health_cols[0]
    )
    primary_workload = workload_reliability_cols[0]

    df_plot = df[[primary_thermal, primary_workload]].dropna()
    if not df_plot.empty:
        df_plot["workload_bin"] = add_bins(df_plot[primary_workload], bins=4)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_plot, x="workload_bin", y=primary_thermal)
        plt.title(f"{primary_thermal} by {primary_workload} bins")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(BOXPLOT_THERMAL_BY_WORKLOAD_PATH)
        plt.close()

        print("\nBoxplot (thermal vs workload bins) saved")

    if "percentage_used" in df.columns and "power_on_hours" in df.columns:
        df_bar = df[["percentage_used", "power_on_hours"]].dropna()
        if not df_bar.empty:
            df_bar["power_on_bin"] = add_bins(df_bar["power_on_hours"], bins=4)
            health_by_power = df_bar.groupby("power_on_bin")["percentage_used"].mean().reset_index()

            plt.figure(figsize=(10, 6))
            sns.barplot(data=health_by_power, x="power_on_bin", y="percentage_used")
            plt.title("Mean Percentage Used by Power On Hours bins")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(BAR_HEALTH_BY_POWER_BIN_PATH)
            plt.close()

            print("\nBar chart (percentage_used by power_on_hours bins) saved")
else:
    print("\nOption B grouped plots not generated (missing thermal/health or workload columns)")


# ------------------------------------------------------------
# 14. Visualizations – Option C (correlation-style heatmaps)
# ------------------------------------------------------------

thermal_health_numeric = [c for c in thermal_health_cols if c in numeric_df.columns]
workload_reliability_numeric = [c for c in workload_reliability_cols if c in numeric_df.columns]

if thermal_health_numeric and workload_reliability_numeric:
    corr_subset = numeric_df[thermal_health_numeric + workload_reliability_numeric].corr()
    corr_th = corr_subset.loc[thermal_health_numeric, workload_reliability_numeric]

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_th, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation – Thermal/Health vs Workload/Reliability")
    plt.tight_layout()
    plt.savefig(HEATMAP_THERMAL_WORKLOAD_PATH)
    plt.close()

    print("\nCorrelation heatmap (thermal/health vs workload/reliability) saved")

    reliability_cols_sub = [c for c in reliability_event_cols if c in numeric_df.columns]
    if reliability_cols_sub:
        corr_subset2 = numeric_df[reliability_cols_sub + workload_reliability_numeric].corr()
        corr_rel = corr_subset2.loc[reliability_cols_sub, workload_reliability_numeric]

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_rel, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation – Reliability Events vs Workload")
        plt.tight_layout()
        plt.savefig(HEATMAP_RELIABILITY_WORKLOAD_PATH)
        plt.close()

        print("\nCorrelation heatmap (reliability vs workload) saved")
else:
    print("\nOption C correlation-style plots not generated (missing numeric thermal/health or workload/reliability)")


# ------------------------------------------------------------
# 15. Categorical analysis (Option C2 – all categoricals)
# ------------------------------------------------------------

explicit_categorical = [
    "ff",
    "system_manufacturer",
    "system_configuration",
    "cpu_name",
    "model_number",
    "firmware_revision",
    "nand_type",
    "nvme_capacity",
    "year",
    "month",
]

all_categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

for c in explicit_categorical:
    if c in df.columns and c not in all_categorical_cols:
        all_categorical_cols.append(c)

for col in all_categorical_cols:
    value_counts = df[col].value_counts(dropna=False)
    if value_counts.empty:
        continue

    freq_path = os.path.join(CATEGORICAL_FREQ_DIR, f"{col}_frequency.csv")
    value_counts.to_csv(freq_path, header=["count"])

    top_counts = value_counts.head(50)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_counts.index.astype(str), y=top_counts.values)
    plt.title(f"Frequency Distribution – {col}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = os.path.join(CATEGORICAL_FREQ_DIR, f"{col}_frequency.png")
    plt.savefig(plot_path)
    plt.close()

print("\nCategorical frequency plots and tables saved")

for cat_col in all_categorical_cols:
    for num_col in numeric_candidates:
        pair_df = df[[cat_col, num_col]].dropna()
        if pair_df.empty:
            continue

        freq = pair_df[cat_col].value_counts()
        if len(freq) > 20:
            top_categories = freq.index[:19]
            pair_df[cat_col] = np.where(
                pair_df[cat_col].isin(top_categories),
                pair_df[cat_col],
                "Other"
            )

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=pair_df, x=cat_col, y=num_col)
        plt.title(f"{num_col} by {cat_col}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        box_path = os.path.join(CATEGORICAL_BOX_DIR, f"{num_col}_by_{cat_col}.png")
        plt.savefig(box_path)
        plt.close()

print("\nCategorical-vs-numeric boxplots saved")


# ------------------------------------------------------------
# 16. Safe HTML profiling
# ------------------------------------------------------------

profile = ProfileReport(
    df,
    minimal=True,
    explorative=False,
)

profile.to_file(EDA_REPORT_PATH)
print("\nSafe HTML EDA report generated")


# ------------------------------------------------------------
# 17. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(CLEANED_CSV_PATH, index=False)
print("\nCleaned dataset saved")
