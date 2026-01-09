"""
Step1: log_generator.py

Author: Priya Pooja Hariharan
Script Version: 1.2

# Project: MIS581 SSD SSD-SMART-Reliability-Analysis 

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
    python log_generator_V1.2.py --num-files 200
    python log_generator_V1.2.py --num-files 200 --inject-errors

Arguments:
    --num-files       Number of logs per drive per workload type.
    --inject-errors   Enables random error-related variability.

In the Version 1.2, the structure, comments, and behavior, are modified slightly to:

1. Vary thresholds and grow errors as wear/age increase
2. Add thermal spikes at high queue depths
3. Increase controller busy time for mixed workloads
4. Inject throttling when temps exceed thresholds
5. Tie unsafe shutdowns and media errors to power‑on hours and power cycles
6. Make latency grow nonlinearly with wear
7. Add bounded fluctuations and bursty errors
8. Add RL friendly power draw dynamics

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

    # ---------------------------------------------------------
    # ENHANCED ERROR-INJECTION & TELEMETRY LOGIC
    # ---------------------------------------------------------

    # ---------- Thermal Health with bounded fluctuations ----------
    base_temp = random.randint(28, 40)
    thermal_noise = random.uniform(-1.5, 2.5)  # bounded fluctuation
    composite_temp_c = max(20, min(90, base_temp + thermal_noise))

    # Thermal spikes when queue depth is high
    if float(qd) >= 32:
        composite_temp_c += random.uniform(3, 8)

    # Convert to Kelvin
    composite_k = composite_temp_c + 273

    temps = [
        composite_temp_c,
        composite_temp_c + random.uniform(-1, 2),
        composite_temp_c + random.uniform(-2, 3),
        composite_temp_c + random.uniform(-1, 1)
    ]

    # ---------- Workload Profile ----------
    data_read = random.randint(10_000_000, 200_000_000)
    data_written = random.randint(8_000_000, 150_000_000)

    host_read_cmds = random.randint(500_000_000, 2_000_000_000)
    host_write_cmds = int(host_read_cmds * random.uniform(0.6, 0.9))

    avg_io_read = round(random.uniform(4.0, 32.0), 1)
    avg_io_write = round(random.uniform(4.0, 32.0), 1)

    # ---------- I/O Activity ----------
    avg_qd = float(qd)
    iops = random.randint(10_000, 120_000)
    bw_read = round(random.uniform(1.0, 6.0), 1)
    bw_write = round(random.uniform(1.0, 5.0), 1)

    # ---------- Usage History ----------
    power_cycles = random.randint(1, 50)
    power_on_hours = random.randint(1000, 8000)

    # Unsafe shutdowns increase with age
    unsafe_shutdowns = int(max(0, (power_on_hours / 3000) + random.uniform(-1, 2)))

    # ---------- Wear Level ----------
    percentage_used = random.randint(1, 10)
    wear_level_avg = random.randint(1, 5)
    wear_level_max = wear_level_avg + random.randint(0, 3)

    # Endurance decreases nonlinearly with wear
    endurance_estimate_remaining = max(0, 100 - (percentage_used * random.uniform(1.0, 1.4)))

    available_spare = max(50, 100 - percentage_used - random.uniform(0, 5))
    spare_threshold = 10

    # ---------- Reliability Risk ----------
    background_scrub_time_pct = round(random.uniform(0.1, 1.0), 1)
    gc_active_time_pct = round(random.uniform(1.0, 6.0), 1)

    # ---------- Fault Indicators with wear- and age-dependent growth ----------
    if INJECT_ERRORS:
        # Media errors increase with wear and power cycles
        media_errors = int(
            max(
                0,
                (percentage_used * random.uniform(0.1, 0.4)) +
                (power_cycles * random.uniform(0.01, 0.05))
            )
        )

        # Occasional bursts of error activity
        if random.random() < 0.15:
            media_errors += random.randint(3, 10)

        # Bad block growth tied to wear
        bad_block_count_grown = int(
            max(0, (percentage_used * random.uniform(0.5, 1.2)))
        )

        # Error log entries scale with media errors
        error_information_log_entries = int(
            media_errors * random.uniform(1.2, 2.5)
        )

        # PCIe errors occasionally spike
        pcie_correctable_errors = int(random.uniform(0, 50) + media_errors * 0.2)
        pcie_uncorrectable_errors = int(
            random.uniform(0, 2) + (1 if media_errors > 10 else 0)
        )
    else:
        media_errors = 0
        bad_block_count_grown = 0
        error_information_log_entries = 0
        pcie_correctable_errors = random.randint(0, 5)
        pcie_uncorrectable_errors = 0

    # ---------- Controller Busy Time ----------
    controller_busy_time = random.randint(100, 1000)

    # Increase busy time under mixed workloads
    if "Mixed" in io_type:
        controller_busy_time += random.randint(200, 600)

    # ---------- Throttling Events ----------
    throttling_events = 0
    if composite_temp_c > 70:
        throttling_events = random.randint(1, 5)

    # ---------- Energy Modeling with fluctuations ----------
    power_state_trans = random.randint(10, 100)
    power_draw_w = round(random.uniform(5.0, 10.0) + random.uniform(-0.5, 0.5), 2)

    # Reinforcement-learning-compatible telemetry: occasional adaptive spikes
    if random.random() < 0.2:
        power_draw_w += random.uniform(1.0, 3.0)

    avg_power_5m = round(power_draw_w + random.uniform(-0.5, 0.5), 2)
    avg_power_1h = round(power_draw_w + random.uniform(-0.7, 0.7), 2)
    peak_power = round(power_draw_w + random.uniform(1.0, 3.0), 2)
    idle_power = round(max(0.5, power_draw_w - random.uniform(3.0, 6.0)), 2)

    # ---------- Latency Metrics with nonlinear wear growth ----------
    base_read_lat = random.uniform(50, 120)    # µs
    base_write_lat = random.uniform(60, 150)   # µs

    # Nonlinear latency growth with wear
    latency_multiplier = 1 + (percentage_used ** 1.3) / 100

    read_lat_avg = round(base_read_lat * latency_multiplier, 2)
    read_lat_p99 = round(read_lat_avg * random.uniform(2.0, 3.5), 2)
    read_lat_max = round(read_lat_avg * random.uniform(4.0, 8.0), 2)

    write_lat_avg = round(base_write_lat * latency_multiplier, 2)
    write_lat_p99 = round(write_lat_avg * random.uniform(2.0, 3.5), 2)
    write_lat_max = round(write_lat_avg * random.uniform(4.0, 8.0), 2)

    # ---------- IO Completion Time ----------
    io_completion_time = round(random.uniform(0.05, 0.50), 3)  # ms

    # ---------- Workload Characteristics ----------
    workload_block_size = block_size  # e.g., "4K", "512B"
    workload_desc = io_type           # e.g., "Sequential Read"

    # ---------- SMART Alert Flags ----------
    # Vary thresholds implicitly: more severe conditions increase chance of warning bits
    critical_warning = "0x00"
    # (You could add bit flipping logic here if you later parse it structurally)

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
Composite Temperature:                    {composite_k:.1f} K ({composite_temp_c:.1f}°C)
Temperature Sensor 1:                     {temps[0]:.1f}°C
Temperature Sensor 2:                     {temps[1]:.1f}°C
Temperature Sensor 3:                     {temps[2]:.1f}°C
Temperature Sensor 4:                     {temps[3]:.1f}°C

-----------------------------------------
Workload Profile (Ratio)
-----------------------------------------
Data Units Read:                          {data_read:,} units
Data Units Written:                       {data_written:,} units
Read/Write Ratio:                         N/A (derived from IO mix)
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
Controller Busy Time:                     {controller_busy_time} minutes

-----------------------------------------
Wear Level (Ratio)
-----------------------------------------
Percentage Used:                          {percentage_used} %
Wear Leveling Count (Avg/Max):            {wear_level_avg} / {wear_level_max}
Endurance Estimate Remaining:             {endurance_estimate_remaining:.1f} %
Available Spare:                          {available_spare:.1f} %
Available Spare Threshold:                {spare_threshold} %

-----------------------------------------
Reliability Risk (Ratio)
-----------------------------------------
Unsafe Shutdowns:                         {unsafe_shutdowns}
Background Scrub Time:                    {background_scrub_time_pct} %
GC Active Time:                           {gc_active_time_pct} %

-----------------------------------------
Fault Indicators (Ratio)
-----------------------------------------
Media Errors:                             {media_errors}
Number of Error Info Log Entries:         {error_information_log_entries}
Bad Block Count (Grown):                  {bad_block_count_grown}
PCIe Correctable Errors:                  {pcie_correctable_errors}
PCIe Uncorrectable Errors:                {pcie_uncorrectable_errors}

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
Telemetry Host-Initiated:                 Supported
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
