#! /usr/bin/env python3
from pyulog import ULog
import numpy as np
import pandas as pd
import rerun as rr
from scipy.spatial.transform import Rotation as R
import sys
from uuid import uuid4
import load_floz
from config_loader import ConfigLoader
import os

def ulog_to_pandas(data_set):
    data_dict = {}
    for field in data_set.field_data:
        data_dict[field.field_name] = data_set.data[field.field_name]
    return pd.DataFrame(data_dict)

conf_filename = sys.argv[1]
conf = ConfigLoader(conf_filename)
rerun_filename = conf.out_filename('.rrd')

ulog_filename = conf.get_filename('ulog_file')
floz_filename = conf.get_filename('floz_file')
mask_filename = conf.get_filename('mask_file')

os.makedirs(conf.output_dir, exist_ok=True)

if mask_filename is not None:
    mask_df = pd.read_csv(mask_filename)
else:
    mask_df = None

# resample FLO data to this interval
flo_resample_rate = conf['flo_resample_rate']
resample_str = '%dus'%(int(1/flo_resample_rate * 1000000.0))


# ULOG processing
ulog = ULog(ulog_filename)
local_pos_data = ulog.get_dataset("vehicle_local_position")
vehicle_gps_position = ulog.get_dataset("vehicle_gps_position") # This data is purely used for correcting timestamps.
vehicle_attitude = ulog.get_dataset("vehicle_attitude")

gps_utc = vehicle_gps_position.data["time_utc_usec"]
gps_t = vehicle_gps_position.data["timestamp"]
utc_offset_usec = gps_utc[0] - gps_t[0]

vehicle_local_position_df = ulog_to_pandas(local_pos_data)
vehicle_local_position_df = vehicle_local_position_df[vehicle_local_position_df['ref_timestamp']!=0]

local_pos_data_t = np.array(vehicle_local_position_df["timestamp"])  # usecs since system start
vehicle_attitude_t = vehicle_attitude.data["timestamp"]  # usecs since system start

local_pos_data_t = pd.to_datetime(local_pos_data_t * 1e-6 + utc_offset_usec * 1e-6, unit="s").tz_localize('UTC')
vehicle_attitude_t= pd.to_datetime(vehicle_attitude_t * 1e-6 + utc_offset_usec * 1e-6, unit="s").tz_localize('UTC')

assert len(vehicle_local_position_df["ref_lat"].unique())==1
assert len(vehicle_local_position_df["ref_lon"].unique())==1
assert len(vehicle_local_position_df["ref_alt"].unique())==1
ref_lat = vehicle_local_position_df.iloc[0]["ref_lat"]
ref_lon = vehicle_local_position_df.iloc[0]["ref_lon"]
ref_alt = vehicle_local_position_df.iloc[0]["ref_alt"]
vehicle_local_position_df["north"] = vehicle_local_position_df["x"]
vehicle_local_position_df["east"] = vehicle_local_position_df["y"]
vehicle_local_position_df["down"] = vehicle_local_position_df["z"]
vehicle_local_position_df["reftime"] = local_pos_data_t
vehicle_attitude_df = ulog_to_pandas(vehicle_attitude)
vehicle_attitude_df["reftime"] = vehicle_attitude_t

px4_combined_df = pd.merge(vehicle_attitude_df, vehicle_local_position_df, how='outer', left_on="timestamp_sample", right_on="timestamp_sample")
px4_combined_df.dropna(inplace=True)

# FLOZ processing
floz = load_floz.load_floz(floz_filename)

# Resample in time
tracking_state_df = floz['tracking_state_df']
tracking_state_df['reftime'] = tracking_state_df['processed_timestamp']
tracking_state_df.set_index('processed_timestamp', inplace=True, verify_integrity=True, drop=True)
tracking_state_df = tracking_state_df.resample(resample_str).mean()

motor_positions_df = floz['motor_positions_df']
motor_positions_df['reftime'] = motor_positions_df['local']
motor_positions_df.set_index('local', inplace=True, verify_integrity=True, drop=True)
motor_positions_df = motor_positions_df.resample(resample_str).mean()

flo_combined_df = pd.merge(tracking_state_df, motor_positions_df, how='outer', left_on="processed_timestamp", right_on = "local")
flo_combined_df.set_index('reftime_x', inplace=True, verify_integrity=True, drop=False)
del tracking_state_df, motor_positions_df

# use rolling median on distance estimates
flo_combined_df['distance_smoothed'] = flo_combined_df['est_dist'].rolling(5).median()

# start tracking
tracking_time_start = conf.get('tracking_time_start')
if tracking_time_start is not None:
    tracking_time_start = pd.to_datetime(tracking_time_start)
    flo_combined_df = flo_combined_df[tracking_time_start <= flo_combined_df['reftime_x']]

# end tracking
tracking_time_end = conf.get('tracking_time_end')
if tracking_time_end is not None:
    tracking_time_end = pd.to_datetime(tracking_time_end)
    flo_combined_df = flo_combined_df[flo_combined_df['reftime_x'] <= tracking_time_end]

null_pan = flo_combined_df[flo_combined_df['pan_enc'].isnull()]
if len(null_pan) > 0:
    first_bad_time = null_pan.iloc[0]['reftime_x']
    print('*'*80)
    print('FLO missing valid pan encoder data from:', first_bad_time)
    print('WARNING: it is suggested to end your data analysis prior to this time.')
    print('You will likely encounter problems with this data.')
    print('*'*80)

print('FLO data time range (prior to masking):', flo_combined_df.iloc[0]['reftime_x'], ' - ', flo_combined_df.iloc[-1]['reftime_x'])

if mask_df is not None:
    dur = pd.Timedelta(seconds=0)
    total_dur = pd.to_datetime(flo_combined_df['reftime_x'].max()) - pd.to_datetime(flo_combined_df['reftime_x'].min())
    for i,row in mask_df.iterrows():
        dur += pd.to_datetime(row['mask_end']) - pd.to_datetime(row['mask_start'])
        idxs = flo_combined_df[(row['mask_start'] <= flo_combined_df['reftime_x']) & (flo_combined_df['reftime_x'] <= row['mask_end'])].index
        for idx in idxs:
            flo_combined_df.at[idx, 'distance_smoothed'] = np.nan
    print(f"{dur} data masked. Total duration: {total_dur}.")

print("Done loading data, now saving to RRD.")

recording_id = conf.get('recording_id', uuid4())
if recording_id is None:
    raise ValueError("Are you setting recording_id to None? Don't do this as it prevents building a new UUID.")
print(f'rerun recording_id: {recording_id}')
rr.init("export_floz_ulog_to_rrd", recording_id = recording_id)
rr.save(rerun_filename)

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis

# rr.log("world/asset", rr.Asset3D(path="odm_texturing/odm_textured_model_geo.obj"))

image_plane_distance = 0.1

RR_FPV_PATH = "world/copter/fpv_cam"
RR_GIMBAL_PATH = "world/copter/gimbal"
print(f'rerun entity path for FPV cam: {RR_FPV_PATH}')
print(f'rerun entity path for gimbal: {RR_GIMBAL_PATH}')

draw_copter = True
if draw_copter:
    # Do not draw copter origin until we have copter pose.
    rr.set_time(timeline="wall_clock", timestamp=px4_combined_df.reftime_x.min().timestamp())

    draw_fpv_cam = True
    if draw_fpv_cam:
        rr.log(RR_FPV_PATH, rr.Transform3D(
                translation = (0.1, 0.0, -0.02),
                rotation=rr.Quaternion(xyzw=R.from_euler('YZ', (90.0,-90.0), degrees=True).as_quat()),
                relation=rr.TransformRelation.ChildFromParent,
            ))

        rr.log(RR_FPV_PATH, rr.Pinhole(
                resolution=[1920, 1080], # for goggles
                focal_length=250.0, # this is not a real calibration, that remains to be done
                image_plane_distance=image_plane_distance,
            ))

# gimbal motor positions
base_gimbal_rot = R.from_euler('YZ', (90.0,-90.0), degrees=True)

pan0 = flo_combined_df.iloc[0]['pan_enc']
tilt0 = flo_combined_df.iloc[0]['tilt_enc']

def nearest_flo_row(reftime):
    nearest_match = abs(flo_combined_df['reftime_x'] - reftime).idxmin()
    flo_row = flo_combined_df.loc[nearest_match]
    return flo_row


def compute_stuff(row):
    result = {}

    result['distance'] = row.distance_smoothed

    # compute gimbal transform based on motor encoder values
    gimbal_offset = R.from_euler('ZY', (-row.pan_enc, -row.tilt_enc))
    result['gimbal_rotation'] = gimbal_offset*base_gimbal_rot

    result['gimbal_translation'] = (0.05, 0.0, -0.2)

    result['pan_enc'] = row.pan_enc
    result['tilt_enc'] = row.tilt_enc

    return result

bee_traj_data = {'east':[],'north':[], 'up':[], 'reftime':[]}
copter_traj_data = {'east':[],'north':[], 'up':[], 'reftime':[], 'qx': [], 'qy': [], 'qz': [], 'qw':[]}

if True:
    # create copter transform
    for (i,row) in px4_combined_df.iterrows():
        rr.set_time(timeline="wall_clock", timestamp=row.reftime_x.timestamp())
        translation = (row["east"], row["north"], -row["down"])
        # Convert from PX4 convention to scipy convention
        rot1 = R.from_quat((row["q[1]"], -row["q[2]"], -row["q[3]"], row["q[0]"]))
        rot2 = R.from_euler('Z', 90.0, degrees=True) # good but need to rotate around X by 180 deg
        rotation = rot2*rot1
        rr.log("world/copter", rr.Transform3D(
                    translation=translation,
                    rotation=rr.Quaternion(xyzw=rotation.as_quat()),
                    relation=rr.TransformRelation.ChildFromParent,
                ))

        # TODO: would interpolation be better than nearest?
        flo_row = nearest_flo_row(row.reftime_x)

        time_delta = (flo_row.reftime_x - row.reftime_x).total_seconds()
        if abs(time_delta) > conf['timestamp_tolerance_dt']:
            continue
        flo_computed = compute_stuff(flo_row)

        # compute world coords of bee

        bee_gimbal_coords = (0.0, 0.0, flo_computed["distance"])
        bee_copter_coords = flo_computed['gimbal_rotation'].as_matrix() @ bee_gimbal_coords + flo_computed['gimbal_translation']
        bee_world_coords = rotation.as_matrix() @ bee_copter_coords + translation

        # TODO: use camera calibration of IR tracking cam and coords of bee in
        # IR cam to project more accurate 3D ray.

        rr.log("world/bee", rr.Points3D([bee_world_coords]))
        bee_traj_data['east'].append(bee_world_coords[0])
        bee_traj_data['north'].append(bee_world_coords[1])
        bee_traj_data['up'].append(bee_world_coords[2])
        # use PX4 timestamps
        bee_traj_data['reftime'].append(row.reftime_x)

        qx, qy, qz, qw = rotation.as_quat()
        copter_traj_data['east'].append(translation[0])
        copter_traj_data['north'].append(translation[1])
        copter_traj_data['up'].append(translation[2])
        copter_traj_data['qx'].append(qx)
        copter_traj_data['qy'].append(qy)
        copter_traj_data['qz'].append(qz)
        copter_traj_data['qw'].append(qw)
        # use PX4 timestamps
        copter_traj_data['reftime'].append(row.reftime_x)

bee_traj_df = pd.DataFrame(bee_traj_data)
bee_traj_df.dropna(inplace=True)
bee_traj_df['ref_lat'] = ref_lat
bee_traj_df['ref_lon'] = ref_lon
bee_traj_df['ref_alt'] = ref_alt

bee_traj_fname = conf.out_filename('-bee.csv')
bee_traj_df.to_csv(bee_traj_fname,sep=',',index=False)
print(f'saved {bee_traj_fname}')

copter_traj_df = pd.DataFrame(copter_traj_data)
copter_traj_df.dropna(inplace=True)

copter_traj_fname = conf.out_filename('-copter.csv')
copter_traj_df.to_csv(copter_traj_fname,sep=',',index=False)
print(f'saved {copter_traj_fname}')

for processed_timestamp,row in flo_combined_df.iterrows():
    # distance estimates ----------------------
    rr.set_time(timeline="wall_clock", timestamp=row.reftime_x.timestamp())
    flo_computed = compute_stuff(row)

    distance_path = "distance\ to\ target\ \[m\]"
    rr.log(distance_path, rr.Scalars(flo_computed["distance"]))

    # create gimbal transform relative to copter ---------------

    rr.log(RR_GIMBAL_PATH, rr.Transform3D(
                translation=flo_computed["gimbal_translation"],
                rotation=rr.Quaternion(xyzw=flo_computed["gimbal_rotation"].as_quat()),
            ))

    rr.log(RR_GIMBAL_PATH, rr.Pinhole(
            resolution=[1920, 1200], # for movie20240712_134113.505531834_Basler-40116283.mp4
            focal_length=500.0, # this is not a real calibration, that remains to be done
            image_plane_distance=image_plane_distance*0.75,
        ))

    # log scalar timeseries of motor encoders
    rr.log("gimbal/pan\ \[rad\]", rr.Scalars(flo_computed['pan_enc']))
    rr.log("gimbal/tilt\ \[rad\]", rr.Scalars(flo_computed['tilt_enc']))

rr.disconnect()
print(f"{rerun_filename} saved")
