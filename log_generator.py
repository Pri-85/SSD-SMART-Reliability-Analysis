import os
import random
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ========== CLI ARGUMENT PARSING ==========
def parse_args():
    parser = argparse.ArgumentParser(description="SSD NVMe benchmark log generator")
    parser.add_argument('--inject-errors', action='store_true', help='Enable random error injection in generated logs')
    parser.add_argument('--num-files', type=int, default=100, help='Number of files per drive per type')
    return parser.parse_args()

args = parse_args()
INJECT_ERRORS = args.inject_errors
file_count_per_type = args.num_files

# ========== CONSTANTS ==========
drive_form_factors = ['E3.s', 'U.2']
drive_capacities = ['3T2', '3T8', '6T4', '12T5', '15T', '25T', '60T', '7T6', '1T7']
read_types = ['Random_Read', 'Sequential_Read']
block_sizes = ['512K','2K', '4K', '8K', '16K', '32K', '64K', '128K']
queue_depths = [1, 2, 4, 8, 16, 32]
file_extensions = ['.smart', '.smart.nvme']
root_dir = Path('SSD_Drive_Structure')

# ========== CAPACITY PROFILES ==========
endurance_profiles = {
    '1T7':  {'read_pb': (3.8, 4.1), 'power_hours': (500, 1000), 'busy_time': (20000, 30000), 'temp_thresh': (70, 75), 'host_cmds': (2e11, 3e11), 'spare_thresh': 9},
    '3T2':  {'read_pb': (4.0, 4.3), 'power_hours': (1000, 1500), 'busy_time': (25000, 35000), 'temp_thresh': (72, 77), 'host_cmds': (2.5e11, 3.5e11), 'spare_thresh': 9},
    '3T8':  {'read_pb': (4.2, 4.6), 'power_hours': (1200, 1600), 'busy_time': (28000, 38000), 'temp_thresh': (73, 78), 'host_cmds': (3e11, 4e11), 'spare_thresh': 9},
    '6T4':  {'read_pb': (4.5, 5.0), 'power_hours': (1500, 2000), 'busy_time': (32000, 42000), 'temp_thresh': (75, 80), 'host_cmds': (3.5e11, 4.5e11), 'spare_thresh': 8},
    '7T6':  {'read_pb': (4.7, 5.2), 'power_hours': (1700, 2200), 'busy_time': (35000, 45000), 'temp_thresh': (76, 81), 'host_cmds': (4e11, 5e11), 'spare_thresh': 8},
    '12T5': {'read_pb': (5.0, 5.4), 'power_hours': (2000, 2600), 'busy_time': (40000, 50000), 'temp_thresh': (77, 82), 'host_cmds': (4.5e11, 5.5e11), 'spare_thresh': 8},
    '15T':  {'read_pb': (5.2, 5.6), 'power_hours': (2400, 3000), 'busy_time': (45000, 55000), 'temp_thresh': (78, 83), 'host_cmds': (5e11, 6e11), 'spare_thresh': 7},
    '25T':  {'read_pb': (5.4, 5.8), 'power_hours': (2600, 3200), 'busy_time': (50000, 60000), 'temp_thresh': (79, 84), 'host_cmds': (5.5e11, 6.5e11), 'spare_thresh': 7},
    '60T':  {'read_pb': (5.8, 6.3), 'power_hours': (3000, 4000), 'busy_time': (55000, 65000), 'temp_thresh': (80, 85), 'host_cmds': (6e11, 7e11), 'spare_thresh': 6},
}

def get_capacity_bytes(cap):
    cap_map = {
        '1T7': (1.7 * 1024**4),
        '3T2': (3.2 * 1024**4),
        '3T8': (3.8 * 1024**4),
        '6T4': (6.4 * 1024**4),
        '7T6': (7.6 * 1024**4),
        '12T5': (12.5 * 1024**4),
        '15T': (15 * 1024**4),
        '25T': (25 * 1024**4),
        '60T': (60 * 1024**4),
    }
    return int(cap_map.get(cap, 3.2 * 1024**4))

def capacity_str(cap_bytes):
    tb = cap_bytes / 1e12
    if tb < 10:
        return f"{tb:.2f} TB"
    return f"{tb:.1f} TB"

def inject_errors():
    if not INJECT_ERRORS or random.random() > 0.15:
        return {'critical_warning': '0x00', 'media_errors': 0, 'error_log_entries': 0, 'temp_warning_time': 0, 'critical_temp_time': 0}
    return {
        'critical_warning': random.choice(['0x01', '0x02', '0x04']),
        'media_errors': random.randint(1, 5),
        'error_log_entries': random.randint(10, 100),
        'temp_warning_time': random.randint(5, 30),
        'critical_temp_time': random.randint(1, 10)
    }

def sample_profile(cap):
    p = endurance_profiles.get(cap)
    read_pb = round(random.uniform(*p['read_pb']), 2)
    write_pb = round(read_pb + random.uniform(0.2, 0.5), 2)
    power_hours = random.randint(*p['power_hours'])
    busy_time = random.randint(*p['busy_time'])
    temp_thresh = random.randint(*p['temp_thresh'])
    host_reads = int(random.uniform(*p['host_cmds']))
    host_writes = int(host_reads * random.uniform(0.8, 1.1))
    spare_thresh = p['spare_thresh']
    return read_pb, write_pb, power_hours, busy_time, temp_thresh, host_reads, host_writes, spare_thresh


def generate_model_number():
    """Randomly generate a model number ending with E3.s or U.2"""
    prefix = random.choice(['XG', 'PM', 'MT', 'NV'])
    digits = f"{random.randint(1000,9999)}"
    suffix = random.choice(['E3.s', 'U.2'])
    return f"{prefix}{digits}-{suffix}"

def generate_smartctl_smart(capacity):
    profile = endurance_profiles.get(capacity, endurance_profiles['3T2'])
    cap_bytes = get_capacity_bytes(capacity)
    readable_cap = capacity_str(cap_bytes)
    temps = [random.randint(38, 64) for _ in range(3)]
    local_time = f"Tue Dec 24 18:{random.randint(10,59)}:{random.randint(10,59)} 2024 MST"
    power_cycles = random.randint(3, 15)
    power_on_hours = random.randint(*profile['power_hours'])
    busy_time = random.randint(*profile['busy_time'])
    read_pb = round(random.uniform(*profile['read_pb']), 2)
    write_pb = round(read_pb + random.uniform(0.2, 0.5), 2)
    read_units = int(read_pb * 2e9)
    write_units = int(write_pb * 2e9)
    host_read_cmds = int(random.uniform(*profile['host_cmds']))
    host_write_cmds = int(host_read_cmds * random.uniform(0.8, 1.1))
    percentage_used = random.randint(5, 15)
    available_spare = 100
    unsafe_shutdowns = random.randint(0, 5)
    temp_thresh = profile['temp_thresh']
    spare_thresh = profile['spare_thresh']

    return f"""smartctl 7.2 2020-12-30 r5155 [x86_64-linux] (local build)
Copyright (C) 2002-20, Bruce Allen, Christian Franke, www.smartmontools.org

=== START OF INFORMATION SECTION ===
Model Number:                       {generate_model_number()}
Serial Number:                      ABCD{random.randint(1000,9999)}EFGH
Firmware Version:                   D3V{random.randint(100,999)}VAB
PCI Vendor ID:                      0x1234
PCI Vendor Subsystem ID:            0x1182
IEEE OUI Identifier:                0x00b538
Total NVM Capacity:                 {cap_bytes:,d} [{readable_cap}]
Unallocated NVM Capacity:           0
Controller ID:                      0
NVMe Version:                       2.0
Number of Namespaces:               16
Namespace 1 Size/Capacity:          {cap_bytes:,d} [{readable_cap}]
Namespace 1 Formatted LBA Size:     512
Namespace 1 IEEE EUI-64:            0026b7 282b2ba6c5
Local Time is:                      {local_time}
Firmware Updates (0x17):            3 Slots, Slot 1 R/O, no Reset required
Optional Admin Commands (0x045e):   Format Frmw_DL NS_Mngmt Self_Test MI_Snd/Rec *Other*
Optional NVM Commands (0x0057):     Comp Wr_Unc DS_Mngmt Sav/Sel_Feat Timestmp
Log Page Attributes (0x7e):         Cmd_Eff_Lg Ext_Get_Lg Telmtry_Lg Pers_Ev_Lg *Other*
Maximum Data Transfer Size:         256 Pages
Warning  Comp. Temp. Threshold:     {temp_thresh[0]} Celsius
Critical Comp. Temp. Threshold:     {temp_thresh[1]} Celsius
Namespace 1 Features (0x1e):        NA_Fields Dea/Unw_Error No_ID_Reuse NP_Fields

Supported Power States
St Op     Max   Active     Idle   RL RT WL WT  Ent_Lat  Ex_Lat
 0 +    25.00W   25.00W       -    0  0  0  0      100     100
 1 +    24.00W   24.00W       -    1  1  1  1       10      10
 2 +    23.00W   23.00W       -    2  2  2  2       10      10
 3 +    22.00W   22.00W       -    3  3  3  3       10      10
 4 +    21.00W   21.00W       -    4  4  4  4       10      10
 5 +    20.00W   20.00W       -    5  5  5  5       10      10
 6 +    19.00W   19.00W       -    6  6  6  6       10      10
 7 +    18.00W   18.00W       -    7  7  7  7       10      10
 8 +    17.00W   17.00W       -    8  8  8  8       10      10
 9 +    16.00W   16.00W       -    9  9  9  9       10      10
10 +    15.00W   15.00W       -   10 10 10 10       10      10
11 +    14.00W   14.00W       -   11 11 11 11       10      10
12 +    13.00W   13.00W       -   12 12 12 12       10      10
13 +    12.00W   12.00W       -   13 13 13 13       10      10

Supported LBA Sizes (NSID 0x1)
Id Fmt  Data  Metadt  Rel_Perf
 0 +     512       0         0
 1 -    4096       0         0

=== START OF SMART DATA SECTION ===
SMART overall-health self-assessment test result: PASSED

SMART/Health Information (NVMe Log 0x02)
Critical Warning:                   0x00
Temperature:                        {temps[0]} Celsius
Available Spare:                    {available_spare}%
Available Spare Threshold:          {spare_thresh}%
Percentage Used:                    {percentage_used}%
Data Units Read:                    {read_units} [{read_pb:.2f} PB]
Data Units Written:                 {write_units} [{write_pb:.2f} PB]
Host Read Commands:                 {host_read_cmds}
Host Write Commands:                {host_write_cmds}
Controller Busy Time:               {busy_time}
Power Cycles:                       {power_cycles}
Power On Hours:                     {power_on_hours}
Unsafe Shutdowns:                   {unsafe_shutdowns}
Media and Data Integrity Errors:    0
Error Information Log Entries:      0
Warning  Comp. Temperature Time:    0
Critical Comp. Temperature Time:    0
Temperature Sensor 1:               {temps[0]} Celsius
Temperature Sensor 2:               {temps[1]} Celsius
Temperature Sensor 3:               {temps[2]} Celsius

Error Information (NVMe Log 0x01, 16 of 256 entries)
No Errors Logged
"""

def generate_nvme_content(cap):
    t = [random.randint(40, 60) for _ in range(3)]
    rpb, wpb, ph, bt, tt, hr, hw, st = sample_profile(cap)
    ru = int(rpb * 2e9); wu = int(wpb * 2e9)
    err = inject_errors()
    return f"""Model Number: {generate_model_number()}
SMART/Health Information Log
critical_warning : {err['critical_warning']}
temperature : {t[0]} °C
available_spare : 100%
available_spare_threshold : {st}%
percentage_used : {random.randint(5, 15)}%
Data Units Read : {ru} ({rpb:.2f} PB)
Data Units Written : {wu} ({wpb:.2f} PB)
host_read_commands : {hr}
host_write_commands : {hw}
controller_busy_time : {bt}
power_on_hours : {ph}
media_errors : {err['media_errors']}
num_err_log_entries : {err['error_log_entries']}
Warning Temperature Time : {err['temp_warning_time']}
Critical Composite Temperature Time: {err['critical_temp_time']}
Temperature Sensor 1 : {t[0]} °C
Temperature Sensor 2 : {t[1]} °C
Temperature Sensor 3 : {t[2]} °C
"""

def get_generator(ext):
    if ext == '.smart':
        return lambda cap, *_: generate_smartctl_smart(cap)
    elif ext == '.smart.nvme':
        return lambda cap, *_: generate_nvme_content(cap)
    else:
        raise ValueError(f"Unknown extension: {ext}")

def create_task(args):
    path, ext, cap, block_size, read_type, qd = args
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = get_generator(ext)
    with open(path, 'w') as f:
        f.write(generator(cap, block_size, read_type, qd))

if __name__ == '__main__':
    tasks = []
    for form in drive_form_factors:
        for cap in drive_capacities:
            for _ in range(file_count_per_type):
                block_size = random.choice(block_sizes)
                read_type = random.choice(read_types)
                qd = random.choice(queue_depths)
                for ext in file_extensions:
                    file_index = random.randint(1, 9999)
                    file_name = f"{block_size}_{read_type}_qd{qd}_{file_index:04d}{ext}"
                    dir_path = root_dir / form / cap / read_type
                    tasks.append((dir_path / file_name, ext, cap, block_size, read_type, qd))
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(create_task, tasks), total=len(tasks), desc='Generating SSD logs'))
    print("Log generation complete.")
