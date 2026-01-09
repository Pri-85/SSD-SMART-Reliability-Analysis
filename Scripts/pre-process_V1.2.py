"""
Step1: pre-process.py

Author: Priya Pooja Hariharan
Script Version: 1.2
# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 
Preprocessing pipeline for synthetic SSD SMART logs.

Extracts ONLY the fields required for the SSD SMART Dataset  attributes:
timestamp, ff, model_number, drive_firmware_revision, nand_type,
nvme_capacity_tb, overprovisioning_ratio, composite_temperature_c,
data_units_read, data_units_written, host_read_commands, host_write_commands,
avg_queue_depth, iops, bandwidth_read_gbps, bandwidth_write_gbps,
io_completion_time_ms, power_cycles, power_on_hours, controller_busy_time,
percentage_used, wear_level_avg, wear_level_max, endurance_estimate_remaining,
unsafe_shutdowns, background_scrub_time_pct, gc_active_time_pct,
media_errors, error_information_log_entries, bad_block_count_grown,
pcie_correctable_errors, pcie_uncorrectable_errors,
workload_type, queue_depth, workload_block_size_kb
"""

import re
import pandas as pd
from pathlib import Path
import numpy as np

RAW_ROOT = Path(r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Raw_V1.2\SSD_Drive_Structure")

# ---------------------------------------------------------
# Helper: Extract value from text via regex
# ---------------------------------------------------------
def extract_value(pattern, text, cast=str):
    match = re.search(pattern, text)
    if match:
        try:
            return cast(match.group(1))
        except Exception:
            return match.group(1)
    return None

# ---------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------

# Capacity buckets for normalization (updated)
CAPACITY_SKU_BUCKETS = [2, 4, 8, 15, 25, 30, 60, 128, 256]

def round_capacity_to_sku(tb_value):
    if tb_value is None:
        return None
    try:
        return min(CAPACITY_SKU_BUCKETS, key=lambda x: abs(x - tb_value))
    except Exception:
        return None

def simplify_nand_type(value):
    if not isinstance(value, str):
        return None
    v = value.upper().strip()

    if "SLC" in v:
        return "SLC"
    if "MLC" in v:
        return "MLC"
    if "TLC" in v:
        return "TLC"
    if "QLC" in v:
        return "QLC"

    return None

def normalize_block_size(value):
    """Convert '4K', '16K', '512B' → numeric KB."""
    if not isinstance(value, str):
        return None
    v = value.upper().strip()

    if v.endswith("K"):
        return float(v.replace("K", ""))
    if v.endswith("KB"):
        return float(v.replace("KB", ""))
    if v.endswith("B"):
        try:
            b = float(v.replace("B", ""))
            return round(b / 1024.0, 3)
        except Exception:
            return None
    return None

def normalize_workload_type(value):
    if not isinstance(value, str):
        return None
    v = value.lower()
    if "seq" in v:
        return "sequential"
    if "rand" in v:
        return "random"
    if "mixed" in v:
        return "mixed"
    return "other"

def extract_form_factor(text):
    """Extract form factor from SMART log text."""
    ff_patterns = ["E3.s", "E1.s", "U.2", "U.3", "M.2"]
    for ff in ff_patterns:
        if ff in text:
            return ff
    return "Other"

# ---------------------------------------------------------
# Parse a single SMART log file
# ---------------------------------------------------------
def parse_smart_log(filepath: Path) -> dict:
    with filepath.open("r", errors="ignore") as f:
        text = f.read()

    record = {}

    # Timestamp
    record["timestamp"] = extract_value(r"Timestamp:\s+([\d\-T:Z]+)", text)

    # Form factor
    record["ff"] = extract_form_factor(text)

    # Device identity
    record["model_number"] = extract_value(r'Model Number:\s+"([^"]+)"', text)
    record["drive_firmware_revision"] = extract_value(r'Firmware Revision:\s+"([^"]+)"', text)
    record["nand_type"] = extract_value(r'NAND Type:\s+([A-Za-z0-9 ]+)', text)

    # Capacity
    record["nvme_capacity_tb"] = extract_value(r"Total NVM Capacity:\s+([\d\.]+)", text, float)
    record["overprovisioning_ratio"] = extract_value(r"Overprovisioning Ratio:\s+(\d+)", text, int)

    # Thermal – composite temperature (e.g. "301.7 K (28.7°C)")
    record["composite_temperature_c"] = extract_value(
        r"Composite Temperature:\s+[\d\.]+\s*K\s*\(([\d\.]+)°C\)",
        text,
        float
    )

    # Workload counters
    record["data_units_read"] = extract_value(
        r"Data Units Read:\s+([\d,]+)", text, lambda x: int(x.replace(",", ""))
    )
    record["data_units_written"] = extract_value(
        r"Data Units Written:\s+([\d,]+)", text, lambda x: int(x.replace(",", ""))
    )
    record["host_read_commands"] = extract_value(
        r"Host Read Commands:\s+([\d,]+)", text, lambda x: int(x.replace(",", ""))
    )
    record["host_write_commands"] = extract_value(
        r"Host Write Commands:\s+([\d,]+)", text, lambda x: int(x.replace(",", ""))
    )

    # IO Activity
    record["avg_queue_depth"] = extract_value(r"Avg Queue Depth:\s+([\d\.]+)", text, float)
    record["iops"] = extract_value(r"IOPS.*:\s+([\d,]+)", text, lambda x: int(x.replace(",", "")))
    record["bandwidth_read_gbps"] = extract_value(r"Bandwidth \(Read\):\s+([\d\.]+)", text, float)
    record["bandwidth_write_gbps"] = extract_value(r"Bandwidth \(Write\):\s*([\d\.]+)", text, float)

    # IO Completion
    record["io_completion_time_ms"] = extract_value(
        r"Average IO Completion Time:\s+([\d\.]+)", text, float
    )

    # Usage
    record["power_cycles"] = extract_value(r"Power Cycles:\s+(\d+)", text, int)
    record["power_on_hours"] = extract_value(r"Power On Hours:\s+(\d+)", text, int)
    record["controller_busy_time"] = extract_value(r"Controller Busy Time:\s+(\d+)", text, int)

    # Wear
    record["percentage_used"] = extract_value(r"Percentage Used:\s+(\d+)", text, int)
    record["wear_level_avg"] = extract_value(r"Wear Leveling Count \(Avg/Max\):\s+(\d+)", text, int)
    record["wear_level_max"] = extract_value(r"Wear Leveling Count \(Avg/Max\):\s+\d+ / (\d+)", text, int)
    record["endurance_estimate_remaining"] = extract_value(
        r"Endurance Estimate Remaining:\s+(\d+)", text, int
    )

    # Reliability Risk
    record["unsafe_shutdowns"] = extract_value(r"Unsafe Shutdowns:\s+(\d+)", text, int)
    record["background_scrub_time_pct"] = extract_value(
        r"Background Scrub Time:\s+([\d\.]+)", text, float
    )
    record["gc_active_time_pct"] = extract_value(
        r"GC Active Time:\s+([\d\.]+)", text, float
    )

    # Fault Indicators
    record["media_errors"] = extract_value(r"Media Errors:\s+(\d+)", text, int)
    record["error_information_log_entries"] = extract_value(
        r"Number of Error Info Log Entries:\s+(\d+)", text, int
    )
    record["bad_block_count_grown"] = extract_value(
        r"Bad Block Count \(Grown\):\s+(\d+)", text, int
    )
    record["pcie_correctable_errors"] = extract_value(
        r"PCIe Correctable Errors:\s+(\d+)", text, int
    )
    record["pcie_uncorrectable_errors"] = extract_value(
        r"PCIe Uncorrectable Errors:\s+(\d+)", text, int
    )

    # Workload characteristics
    record["workload_block_size"] = extract_value(r"Workload Block Size:\s+(.+)", text)
    record["workload_type"] = extract_value(r"Workload\s*Type\s*:\s*([^\n]+)", text)
    record["queue_depth"] = extract_value(r"Queue Depth:\s+(.+)", text)

    return record

# ---------------------------------------------------------
# Load all logs
# ---------------------------------------------------------
def load_all_logs(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*.smart"):
        rows.append(parse_smart_log(path))
    for path in root.rglob("*.smart.nvme"):
        rows.append(parse_smart_log(path))
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Clean + Normalize
# ---------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").drop_duplicates()

    # Ensure important columns exist
    if "composite_temperature_c" not in df.columns:
        df["composite_temperature_c"] = pd.NA
    if "nand_type" not in df.columns:
        df["nand_type"] = pd.NA

    # Basic normalization
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["nand_type"] = df["nand_type"].astype(str).str.strip().apply(simplify_nand_type)
    df["nvme_capacity_tb"] = df["nvme_capacity_tb"].apply(round_capacity_to_sku)
    df["workload_block_size_kb"] = df["workload_block_size"].apply(normalize_block_size)
    df["workload_type"] = df["workload_type"].apply(normalize_workload_type)

    if "queue_depth" in df.columns:
        df["queue_depth"] = (
            df["queue_depth"].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")
        )

    # ------------------------------------------------------------------
    # NAND-type–aware scaling for read/write counters
    # ------------------------------------------------------------------
    for col in ["data_units_read", "data_units_written"]:
        if col not in df.columns:
            df[col] = np.nan

    def scale_rw(row):
        base_r = row["data_units_read"]
        base_w = row["data_units_written"]

        if pd.isna(base_r):
            base_r = 0
        if pd.isna(base_w):
            base_w = 0

        nand = row["nand_type"]
        cap = row["nvme_capacity_tb"]

        factor = 1.0
        if nand == "QLC":
            if cap is not None and cap >= 30:
                factor = 8.0
            else:
                factor = 4.0
        elif nand == "MLC":
            if cap is not None and cap >= 30:
                factor = 4.0
            else:
                factor = 2.0

        return pd.Series({
            "data_units_read": int(base_r * factor),
            "data_units_written": int(base_w * factor)
        })

    df[["data_units_read", "data_units_written"]] = df.apply(scale_rw, axis=1)

    # ------------------------------------------------------------------
    # Capacity-aware error injection for low-capacity drives
    # ------------------------------------------------------------------
    for col in ["media_errors", "pcie_correctable_errors",
                "pcie_uncorrectable_errors", "bad_block_count_grown"]:
        if col not in df.columns:
            df[col] = np.nan

    def inject_errors(row):
        cap = row["nvme_capacity_tb"]
        if cap in [2, 4, 8]:
            err_factor = 5
        elif cap in [15, 25]:
            err_factor = 2
        else:
            err_factor = 1

        return pd.Series({
            "media_errors": int((0 if pd.isna(row["media_errors"]) else row["media_errors"]) * err_factor),
            "pcie_correctable_errors": int((0 if pd.isna(row["pcie_correctable_errors"]) else row["pcie_correctable_errors"]) * err_factor),
            "pcie_uncorrectable_errors": int((0 if pd.isna(row["pcie_uncorrectable_errors"]) else row["pcie_uncorrectable_errors"]) * err_factor),
            "bad_block_count_grown": int((0 if pd.isna(row["bad_block_count_grown"]) else row["bad_block_count_grown"]) * err_factor),
        })

    df[["media_errors", "pcie_correctable_errors",
        "pcie_uncorrectable_errors", "bad_block_count_grown"]] = df.apply(inject_errors, axis=1)

    # ------------------------------------------------------------------
    # Workload-type–aware performance adjustment (seq vs random)
    # ------------------------------------------------------------------
    for col in ["bandwidth_read_gbps", "bandwidth_write_gbps", "iops", "io_completion_time_ms"]:
        if col not in df.columns:
            df[col] = np.nan

    def adjust_perf(row):
        bw_r = row["bandwidth_read_gbps"]
        bw_w = row["bandwidth_write_gbps"]
        iops = row["iops"]
        lat = row["io_completion_time_ms"]

        if pd.isna(bw_r):
            bw_r = 0
        if pd.isna(bw_w):
            bw_w = 0
        if pd.isna(iops):
            iops = 0
        if pd.isna(lat):
            lat = 0

        wt = row["workload_type"]

        if wt == "sequential":
            return pd.Series({
                "bandwidth_read_gbps": bw_r * 1.5,
                "bandwidth_write_gbps": bw_w * 1.5,
                "iops": int(iops * 0.6),
                "io_completion_time_ms": lat * 0.7
            })
        elif wt == "random":
            return pd.Series({
                "bandwidth_read_gbps": bw_r * 0.6,
                "bandwidth_write_gbps": bw_w * 0.6,
                "iops": int(iops * 1.4),
                "io_completion_time_ms": lat * 1.3
            })
        else:
            return pd.Series({
                "bandwidth_read_gbps": bw_r,
                "bandwidth_write_gbps": bw_w,
                "iops": int(iops),
                "io_completion_time_ms": lat
            })

    df[["bandwidth_read_gbps", "bandwidth_write_gbps",
        "iops", "io_completion_time_ms"]] = df.apply(adjust_perf, axis=1)

    # Final column order
    cols = [
        "timestamp", "ff", "model_number", "drive_firmware_revision", "nand_type",
        "nvme_capacity_tb", "overprovisioning_ratio", "composite_temperature_c",
        "data_units_read", "data_units_written", "host_read_commands",
        "host_write_commands", "avg_queue_depth", "iops", "bandwidth_read_gbps",
        "bandwidth_write_gbps", "io_completion_time_ms", "power_cycles",
        "power_on_hours", "controller_busy_time", "percentage_used",
        "wear_level_avg", "wear_level_max", "endurance_estimate_remaining",
        "unsafe_shutdowns", "background_scrub_time_pct", "gc_active_time_pct",
        "media_errors", "error_information_log_entries", "bad_block_count_grown",
        "pcie_correctable_errors", "pcie_uncorrectable_errors",
        "workload_type", "queue_depth", "workload_block_size_kb"
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[cols]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
def save_output(df: pd.DataFrame):
    out = Path(r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.2\Step1-processed_smart_dataset_V1.2.csv")
    df.to_csv(out, index=False)
    print(f"Saved cleaned dataset → {out.resolve()}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Scanning SMART logs...")
    df_raw = load_all_logs(RAW_ROOT)
    print(f"Loaded {len(df_raw)} logs")
    df_clean = clean_dataframe(df_raw)
    save_output(df_clean)
