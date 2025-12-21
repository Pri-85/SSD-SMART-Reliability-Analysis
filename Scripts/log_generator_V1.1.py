"""
SSD Synthetic SMART Log Generator
---------------------------------

This script generates large-scale synthetic NVMe SMART logs for SSD reliability
research, benchmarking, and machine learning pipelines. It produces realistic
SMART/Health Information logs in text formats (.smart, .smart.nvme) and
supports a wide range of SSD form factors, capacities, workloads, and telemetry
attributes.

Key Features:
- Generates detailed SMART logs modeled after NVMe Log Page 0x02, including:
  * Device identity, thermal health, workload profile, I/O activity,
    usage history, wear indicators, reliability metrics, fault indicators,
    latency metrics, and workload characteristics.
- Supports multiple SSD form factors (E3.s, U.2, U.3, M.2, E1.s).
- Produces unique serial numbers and firmware revisions (e.g., F21000.1).
- Randomizes NAND type across SLC, MLC, TLC, QLC.
- Supports an extended capacity SKU set:
    1T7, 3T2, 3T8, 6T4, 7T6, 12T5, 15T, 25T6, 30T7, 60T, 128T, 256T
- Reads queue depths, IO types, and workload sizes from external .txt files
  for flexible workload modeling.
- Constrains timestamps to a defined 6-month window (2025-02-11 to 2025-07-30).
- Uses multiprocessing to efficiently generate thousands of logs in parallel.
- Organizes output into a structured directory hierarchy:
      SSD_Drive_Structure/<FormFactor>/<Capacity>/<IOType>/<LogFiles>

Usage examples:
    python log_generator.py --num-files 200
    python log_generator.py --num-files 200 --inject-errors

Arguments:
    --num-files       Number of logs per drive per workload type.
    --inject-errors   Enables random error-related variability.
"""

import random
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import datetime

# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SSD NVMe SMART log generator")
    parser.add_argument(
        '--inject-errors',
        action='store_true',
        help='Enable random error-related variability (degraded logs)'
    )
    parser.add_argument(
        '--num-files',
        type=int,
        default=100,
        help='Number of files per drive per IO type'
    )
    return parser.parse_args()


args = parse_args()
INJECT_ERRORS = args.inject_errors
file_count_per_type = args.num_files

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def load_list_from_file(path, default_list):
    """Load a list of values from a text file, fallback to default_list if missing/empty."""
    p = Path(path)
    if not p.exists():
        return default_list
    with p.open() as f:
        values = [line.strip() for line in f if line.strip()]
    return values if values else default_list


def random_timestamp(start_date, end_date):
    """
    Return a random timestamp between two datetime.date objects
    in ISO 8601 UTC format.
    """
    start = datetime.datetime.combine(start_date, datetime.time.min)
    end = datetime.datetime.combine(end_date, datetime.time.max)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    ts = start + datetime.timedelta(seconds=random_seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Time window for synthetic logs
START_DATE = datetime.date(2025, 2, 11)
END_DATE = datetime.date(2025, 7, 30)

# Form factors and capacities
drive_form_factors = ['E3.s', 'U.2', 'U.3', 'M.2', 'E1.s']

# Exact capacity SKUs (symbolic labels)
drive_capacities = [
    '1T7',
    '3T2',
    '3T8',
    '6T4',
    '7T6',
    '12T5',
    '15T',
    '25T6',
    '30T7',
    '60T',
    '128T',
    '256T'
]

# Map SKU to realistic TB ranges
capacity_tb_ranges = {
    '1T7':   (1.6, 1.8),
    '3T2':   (3.1, 3.3),
    '3T8':   (3.7, 3.9),
    '6T4':   (6.3, 6.5),
    '7T6':   (7.5, 7.7),
    '12T5':  (12.4, 12.6),
    '15T':   (14.8, 15.2),
    '25T6':  (25.5, 25.7),
    '30T7':  (30.6, 30.8),
    '60T':   (59.0, 61.0),
    '128T':  (127.0, 129.0),
    '256T':  (255.0, 257.0),
}

# Workload & IO parameters from external files if present
queue_depths = load_list_from_file("queue_depths.txt", ["1", "4", "8", "16", "32"])
io_types = load_list_from_file(
    "io_types.txt",
    [
        "Sequential Read",
        "Random Read",
        "Sequential Write",
        "Random Write",
        "Mixed Read/Write"
    ]
)
workload_sizes = load_list_from_file(
    "workload_sizes.txt",
    ["4K", "512B", "8K", "16K"]
)

file_extensions = ['.smart', '.smart.nvme']
root_dir = Path('SSD_Drive_Structure')

# NAND types
nand_types = ["SLC 3D NAND", "MLC 3D NAND", "TLC 3D NAND", "QLC 3D NAND"]

# ---------------------------------------------------------
# ADVANCED SMART LOG GENERATOR
# ---------------------------------------------------------
def generate_advanced_smart_log(capacity_sku, form_factor, block_size, io_type, qd):
    """
    Generate a detailed SMART / Health Information log emulating NVMe-style output,
    including latency, workload characteristics, and constrained timestamps.
    """

    # ---------- Device Identity ----------
    model_prefix = random.choice(["XG", "PM", "MT", "NV"])
    model_number = f"{model_prefix}-DC{random.randint(8000, 9999)}-{form_factor.replace('.', '').upper()}"

    serial_number = (
        f"SN{random.randint(10, 99)}"
        f"{random.randint(1000, 9999)}"
        f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
        f"{random.randint(100, 999)}"
    )
    firmware_revision = f"F{random.randint(10000, 99999)}.{random.randint(0, 9)}"

    nand_type = random.choice(nand_types)

    # ---------- Capacity ----------
    tb_low, tb_high = capacity_tb_ranges.get(capacity_sku, (3.1, 3.3))
    total_tb = round(random.uniform(tb_low, tb_high), 2)
    formatted_tb = round(total_tb * random.uniform(0.92, 0.97), 2)
    overprov = int((total_tb - formatted_tb) / total_tb * 100)

    # ---------- Timestamp ----------
    timestamp = random_timestamp(START_DATE, END_DATE)

    # ---------- Thermal Health ----------
    temps = [random.randint(28, 40) for _ in range(4)]
    composite_k = temps[0] + 273

    # ---------- Workload Profile ----------
    data_read = random.randint(10_000_000, 200_000_000)
    data_written = random.randint(8_000_000, 150_000_000)
    read_ratio = random.randint(40, 70)
    write_ratio = 100 - read_ratio

    host_read_cmds = random.randint(500_000_000, 2_000_000_000)
    host_write_cmds = int(host_read_cmds * random.uniform(0.6, 0.9))

    avg_io_read = round(random.uniform(4.0, 32.0), 1)
    avg_io_write = round(random.uniform(4.0, 32.0), 1)

    # ---------- I/O Activity ----------
    avg_qd = float(qd) if isinstance(qd, str) else float(qd)
    iops = random.randint(10_000, 120_000)
    bw_read = round(random.uniform(1.0, 6.0), 1)
    bw_write = round(random.uniform(1.0, 5.0), 1)

    # ---------- Usage History ----------
    power_cycles = random.randint(1, 50)
    power_on_hours = random.randint(1000, 8000)
    busy_minutes = random.randint(100, 1000)

    # ---------- Wear Level ----------
    pct_used = random.randint(1, 10)
    wl_avg = random.randint(1, 5)
    wl_max = wl_avg + random.randint(0, 3)
    endurance_remaining = 100 - pct_used
    available_spare = max(90, 100 - pct_used)
    spare_threshold = 10

    # ---------- Reliability Risk ----------
    unsafe = random.randint(0, 3)
    scrub = round(random.uniform(0.1, 1.0), 1)
    gc_time = round(random.uniform(1.0, 6.0), 1)

    # ---------- Fault Indicators ----------
    media_errors = random.randint(0, 3) if INJECT_ERRORS else 0
    error_log_entries = 0 if media_errors == 0 else random.randint(1, 20)
    grown_bad = random.randint(0, 5)
    pcie_corr = random.randint(0, 50)
    pcie_uncorr = random.randint(0, 2) if INJECT_ERRORS else 0

    # ---------- SMART Alert Flags ----------
    critical_warning = "0x00"

    # ---------- Energy Modeling ----------
    power_state_trans = random.randint(10, 100)
    avg_power_5m = round(random.uniform(5.0, 10.0), 1)
    avg_power_1h = round(random.uniform(5.0, 10.0), 1)
    peak_power = round(random.uniform(8.0, 12.0), 1)
    idle_power = round(random.uniform(0.5, 2.0), 1)

    # ---------- Latency Metrics (NVMe-ish) ----------
    read_lat_avg = round(random.uniform(50, 120), 2)        # µs
    read_lat_p99 = round(read_lat_avg * random.uniform(2.0, 3.5), 2)
    read_lat_max = round(read_lat_avg * random.uniform(4.0, 8.0), 2)

    write_lat_avg = round(random.uniform(60, 150), 2)       # µs
    write_lat_p99 = round(write_lat_avg * random.uniform(2.0, 3.5), 2)
    write_lat_max = round(write_lat_avg * random.uniform(4.0, 8.0), 2)

    # ---------- IO Completion Time ----------
    io_completion_time = round(random.uniform(0.05, 0.50), 3)  # ms

    # ---------- Workload Characteristics ----------
    workload_block_size = block_size  # e.g., "4K", "512B"
    workload_desc = io_type           # e.g., "Sequential Read"

    # ---------- Final SMART Log Text ----------
    return f"""SMART / Health Information Log (0x02)
Timestamp:                               {timestamp}

-----------------------------------------
Device Identity (Nominal)
-----------------------------------------
Model Number:                            "{model_number}"
Drive Type:                               {form_factor}
Serial Number:                            "{serial_number}"
Firmware Revision:                        "{firmware_revision}"
NAND Type:                                {nand_type}
Namespace Count:                          1

-----------------------------------------
Storage Size (Ratio)
-----------------------------------------
Total NVM Capacity:                       {total_tb:.2f} TB
Formatted Capacity:                       {formatted_tb:.2f} TB
Overprovisioning Ratio:                   {overprov} %

-----------------------------------------
Thermal Health (Ratio)
-----------------------------------------
Composite Temperature:                    {composite_k} K ({temps[0]}°C)
Temperature Sensor 1:                     {temps[0]}°C
Temperature Sensor 2:                     {temps[1]}°C
Temperature Sensor 3:                     {temps[2]}°C
Temperature Sensor 4:                     {temps[3]}°C

-----------------------------------------
Workload Profile (Ratio)
-----------------------------------------
Data Units Read:                          {data_read:,} units
Data Units Written:                       {data_written:,} units
Read/Write Ratio:                         {read_ratio}R / {write_ratio}W
Host Read Commands:                       {host_read_cmds:,}
Host Write Commands:                      {host_write_cmds:,}
Avg IO Size (Read):                       {avg_io_read} KB
Avg IO Size (Write):                      {avg_io_write} KB

-----------------------------------------
I/O Activity (Ratio)
-----------------------------------------
Avg Queue Depth:                          {avg_qd}
IOPS (5 min avg):                         {iops:,}
Bandwidth (Read):                         {bw_read} GB/s
Bandwidth (Write):                        {bw_write} GB/s

-----------------------------------------
Latency Metrics (NVMe Spec)
-----------------------------------------
Read Latency (Avg):                       {read_lat_avg} µs
Read Latency (P99):                       {read_lat_p99} µs
Read Latency (Max):                       {read_lat_max} µs
Write Latency (Avg):                      {write_lat_avg} µs
Write Latency (P99):                      {write_lat_p99} µs
Write Latency (Max):                      {write_lat_max} µs

-----------------------------------------
IO Timing (NVMe Spec)
-----------------------------------------
Average IO Completion Time:               {io_completion_time} ms

-----------------------------------------
Workload Characteristics
-----------------------------------------
Workload Block Size:                      {workload_block_size}
Workload Type:                            {workload_desc}
Queue Depth:                              {qd}

-----------------------------------------
Usage History (Ratio)
-----------------------------------------
Power Cycles:                             {power_cycles}
Power On Hours:                           {power_on_hours}
Controller Busy Time:                     {busy_minutes} minutes

-----------------------------------------
Wear Level (Ratio)
-----------------------------------------
Percentage Used:                          {pct_used} %
Wear Leveling Count (Avg/Max):            {wl_avg} / {wl_max}
Endurance Estimate Remaining:             {endurance_remaining} %

-----------------------------------------
Remaining Endurance (Ratio)
-----------------------------------------
Available Spare:                          {available_spare} %
Available Spare Threshold:                {spare_threshold} %

-----------------------------------------
Reliability Risk (Ratio)
-----------------------------------------
Unsafe Shutdowns:                         {unsafe}
Background Scrub Time:                    {scrub} %
GC Active Time:                           {gc_time} %

-----------------------------------------
Fault Indicators (Ratio)
-----------------------------------------
Media Errors:                             {media_errors}
Number of Error Info Log Entries:         {error_log_entries}
Bad Block Count (Grown):                  {grown_bad}
PCIe Correctable Errors:                  {pcie_corr}
PCIe Uncorrectable Errors:                {pcie_uncorr}

-----------------------------------------
SMART Alert Flags (Binary/Nominal)
-----------------------------------------
Critical Warning:                         {critical_warning}
  [bit 0] Spare Below Threshold:          0
  [bit 1] Temperature Threshold:          0
  [bit 2] Reliability Degraded:           0
  [bit 3] Read Only Mode:                 0
  [bit 4] Volatile Memory Backup Failed:  0

-----------------------------------------
Energy Modeling (Ratio)
-----------------------------------------
Power State:                              PS1 (Active)
Power State Transitions:                  {power_state_trans}
Avg Power (5 min):                        {avg_power_5m} W
Avg Power (1 hr):                         {avg_power_1h} W
Peak Power:                               {peak_power} W
Idle Power:                               {idle_power} W

-----------------------------------------
Capability Flags (Nominal)
-----------------------------------------
Namespace Features:                       Thin Provisioning Supported
Atomic Write Units:                       16 KB
Metadata Capabilities:                    Extended + Separate
Security Features:                        TCG Opal + Sanitize
Endurance Group Support:                  Yes
Telemetry Host-Initiated:                Supported
Telemetry Controller-Initiated:           Supported
"""


# ---------------------------------------------------------
# GENERATOR SELECTOR
# ---------------------------------------------------------
def get_generator(ext):
    # For now, both .smart and .smart.nvme use the same content style
    if ext in ('.smart', '.smart.nvme'):
        return generate_advanced_smart_log
    else:
        raise ValueError(f"Unknown extension: {ext}")


# ---------------------------------------------------------
# TASK EXECUTION
# ---------------------------------------------------------
def create_task(args):
    path, ext, cap, form_factor, block_size, read_type, qd = args
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = get_generator(ext)

    content = generator(cap, form_factor, block_size, read_type, qd)
    with open(path, 'w') as f:
        f.write(content)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == '__main__':
    tasks = []

    for form in drive_form_factors:
        for cap in drive_capacities:
            for _ in range(file_count_per_type):
                block_size = random.choice(workload_sizes)
                read_type = random.choice(io_types)
                qd = random.choice(queue_depths)

                for ext in file_extensions:
                    file_index = random.randint(1, 9999)
                    safe_read_type = read_type.replace(" ", "")
                    file_name = f"{block_size}_{safe_read_type}_qd{qd}_{file_index:04d}{ext}"
                    dir_path = root_dir / form / cap / read_type.replace(" ", "_")
                    tasks.append((dir_path / file_name, ext, cap, form, block_size, read_type, qd))

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(create_task, tasks), total=len(tasks), desc='Generating SSD logs'))

    print("Log generation complete.")
