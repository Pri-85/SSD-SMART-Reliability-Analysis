"""
SSD Synthetic SMART Log Generator
---------------------------------

This script generates large-scale synthetic NVMe SMART logs for SSD reliability
research, benchmarking, and machine learning pipelines. It produces realistic
SMART/Health Information logs in multiple formats (.smart, .smart.nvme) and
supports a wide range of SSD form factors, capacities, workloads, and telemetry
attributes.

Key Features:
- Generates detailed SMART logs modeled after NVMe Log Page 0x02, including:
  * Device identity, thermal health, workload profile, I/O activity,
    usage history, wear indicators, reliability metrics, and fault indicators.
- Supports multiple SSD form factors (E3.s, U.2, U.3, M.2, E1.s).
- Produces unique serial numbers and firmware revisions (e.g., F21000.1).
- Reads queue depths, IO types, and workload sizes from external .txt files
  for flexible workload modeling.
- Allows optional random error injection to simulate degraded or failing drives.
- Uses multiprocessing to efficiently generate thousands of logs in parallel.
- Organizes output into a structured directory hierarchy:
      SSD_Drive_Structure/<FormFactor>/<Capacity>/<IOType>/<LogFiles>

Usage:
    python log_generator.py --num-files 1000
    python log_generator.py --num-files 1000 --inject-errors

Arguments:
    --num-files       Number of logs per drive per workload type.
    --inject-errors   Enables random SMART error injection.

This script is designed for reproducible research workflows, synthetic dataset
generation, and large-scale SSD telemetry simulation.
"""
import random
import datetime

def generate_advanced_smart_log(capacity, form_factor):
    """Generate the new detailed SMART log format."""

    # ---------- Unique Model + Serial + Firmware ----------
    model_prefix = random.choice(["XG", "PM", "MT", "NV"])
    model_number = f"{model_prefix}-DC{random.randint(8000,9999)}-{form_factor.replace('.', '').upper()}"

    serial_number = f"SN{random.randint(10,99)}{random.randint(1000,9999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100,999)}"

    firmware_revision = f"F{random.randint(10000,99999)}.{random.randint(0,9)}"

    # ---------- Capacity ----------
    total_tb = round(random.uniform(1.6, 60.0), 2)
    formatted_tb = round(total_tb * random.uniform(0.92, 0.97), 2)
    overprov = int((total_tb - formatted_tb) / total_tb * 100)

    # ---------- Temperatures ----------
    temps = [random.randint(28, 40) for _ in range(4)]
    composite_k = temps[0] + 273

    # ---------- Workload ----------
    data_read = random.randint(10_000_000, 200_000_000)
    data_written = random.randint(8_000_000, 150_000_000)
    read_ratio = random.randint(40, 70)
    write_ratio = 100 - read_ratio

    host_read_cmds = random.randint(500_000_000, 2_000_000_000)
    host_write_cmds = int(host_read_cmds * random.uniform(0.6, 0.9))

    avg_io_read = round(random.uniform(4.0, 32.0), 1)
    avg_io_write = round(random.uniform(4.0, 32.0), 1)

    # ---------- IO Activity ----------
    avg_qd = round(random.uniform(1.0, 32.0), 1)
    max_qd = 64
    iops = random.randint(10_000, 120_000)
    bw_read = round(random.uniform(1.0, 6.0), 1)
    bw_write = round(random.uniform(1.0, 5.0), 1)

    # ---------- Usage ----------
    power_cycles = random.randint(1, 50)
    power_on_hours = random.randint(1000, 8000)
    busy_minutes = random.randint(100, 1000)

    # ---------- Wear ----------
    pct_used = random.randint(1, 10)
    wl_avg = random.randint(1, 5)
    wl_max = wl_avg + random.randint(0, 3)
    endurance_remaining = 100 - pct_used

    # ---------- Reliability ----------
    unsafe = random.randint(0, 3)
    scrub = round(random.uniform(0.1, 1.0), 1)
    gc_time = round(random.uniform(1.0, 6.0), 1)

    # ---------- Fault Indicators ----------
    grown_bad = random.randint(0, 5)
    pcie_corr = random.randint(0, 50)

    # ---------- Timestamp ----------
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---------- Final Log ----------
    return f"""
SMART / Health Information Log (0x02)
Timestamp:                               {timestamp}

-----------------------------------------
Device Identity (Nominal)
-----------------------------------------
Model Number:                            "{model_number}"
Drive Type:                               {form_factor}
Serial Number:                            "{serial_number}"
Firmware Revision:                        "{firmware_revision}"
NAND Type:                                TLC 3D NAND
Namespace Count:                          1

-----------------------------------------
Storage Size (Ratio)
-----------------------------------------
Total NVM Capacity:                       {total_tb} TB
Formatted Capacity:                       {formatted_tb} TB
Overprovisioning Ratio:                   {overprov} %

-----------------------------------------
Thermal Health (Ratio)
-----------------------------------------
Composite Temperature:                    {composite_k} K ({temps[0]}°C)
Temperature Sensor 1:                     {temps[0]}°C
Temperature Sensor 2:                     {temps[1]}°C
Temperature Sensor 3:                     {temps[2]}°C
Temperature Sensor 4:                     {temps[3]}°C
Thermal Throttle Events:                  {random.randint(0,5)}
Thermal Throttle Duration:                {random.randint(1,20)} minutes

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
Max Queue Depth:                          {max_qd}
IOPS (5 min avg):                         {iops:,}
Bandwidth (Read):                         {bw_read} GB/s
Bandwidth (Write):                        {bw_write} GB/s

-----------------------------------------
Usage History (Ratio)
-----------------------------------------
Power Cycles:                             {power_cycles}
Power On Hours:                           {power_on_hours}
Controller Busy Time:                     {busy_minutes} minutes
Host Throttle Events:                     0
Device Throttle Events:                   {random.randint(0,3)}

-----------------------------------------
Wear Level (Ratio)
-----------------------------------------
Percentage Used:                          {pct_used} %
Wear Leveling Count (Avg/Max):            {wl_avg} / {wl_max}
Endurance Estimate Remaining:             {endurance_remaining} %

-----------------------------------------
Remaining Endurance (Ratio)
-----------------------------------------
Available Spare:                          {100 - pct_used} %
Available Spare Threshold:                10 %

-----------------------------------------
Reliability Risk (Ratio)
-----------------------------------------
Unsafe Shutdowns:                         {unsafe}
Background Scrub Time:                    {scrub} %
GC Active Time:                           {gc_time} %

-----------------------------------------
Fault Indicators (Ratio)
-----------------------------------------
Media Errors:                             0
Number of Error Info Log Entries:         0
Bad Block Count (Grown):                  {grown_bad}
Program Fail Count:                       0
Erase Fail Count:                         0
PCIe Correctable Errors:                  {pcie_corr}
PCIe Uncorrectable Errors:                0

-----------------------------------------
SMART Alert Flags (Binary/Nominal)
-----------------------------------------
Critical Warning:                         0x00
  [bit 0] Spare Below Threshold:          0
  [bit 1] Temperature Threshold:          0
  [bit 2] Reliability Degraded:           0
  [bit 3] Read Only Mode:                 0
  [bit 4] Volatile Memory Backup Failed:  0

-----------------------------------------
Energy Modeling (Ratio)
-----------------------------------------
Power State:                              PS1 (Active)
Power State Transitions:                  {random.randint(10,100)}
Avg Power (5 min):                        {round(random.uniform(5.0,10.0),1)} W
Avg Power (1 hr):                         {round(random.uniform(5.0,10.0),1)} W
Peak Power:                               {round(random.uniform(8.0,12.0),1)} W
Idle Power:                               {round(random.uniform(0.5,2.0),1)} W

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
