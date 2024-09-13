#! /usr/bin/env python3
from pyulog import ULog
import pandas as pd
import sys

def ulog_to_pandas(data_set):
    data_dict = {}
    for field in data_set.field_data:
        data_dict[field.field_name] = data_set.data[field.field_name]
    return pd.DataFrame(data_dict)

ulog_filename = sys.argv[1]

# ULOG processing
ulog = ULog(ulog_filename)

for dataset in ["vehicle_local_position", "vehicle_gps_position", "vehicle_attitude", "vehicle_global_position"]:
    df = ulog_to_pandas(ulog.get_dataset(dataset))

    usecs_since_start = df["timestamp"]
    start_usec, stop_usec = (usecs_since_start.iloc[0], usecs_since_start.iloc[-1])
    dur_usecs = stop_usec - start_usec
    num_samples = len(usecs_since_start)-1

    sample_rate = num_samples/dur_usecs * 1e6

    print(f"# {dataset}")
    print(f"sample_rate: {sample_rate} Hz, columns: {df.columns}")
    print(df)
    print()

print('# all data')
print(' '.join(set([dl.name for dl in ulog.data_list])))
