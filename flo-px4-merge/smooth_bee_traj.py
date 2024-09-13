import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import adskalman.adskalman as adskalman
import imageio.v3 as iio
import os
import sys
import yaml
import pyned2lla
import math
from config_loader import ConfigLoader
import os

D2R = math.pi/180.0
R2D = 180.0/math.pi

conf_filename = sys.argv[1]
conf = ConfigLoader(conf_filename)

# Set this to True to view an interactive single plot.
show_final_only = True

if show_final_only:
    from matplotlib import rcParams
    rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial'] # lucid: ttf-mscorefonts-installer

if not show_final_only:
    os.makedirs('topview-10fps-raw')

def column(arr):
    """convert 1D array-like to a 2D vertical array

    >>> column((1,2,3))

    array([[1],
           [2],
           [3]])
    """
    arr = np.array(arr)
    assert arr.ndim == 1
    a2 = arr[:, np.newaxis]
    return a2


# Create a 6-dimensional state space model:
# (x, y, z, xvel, yvel, zvel).
dt = conf['kalman_smoothing_dt']

# This is F in wikipedia language.
motion_model = np.array([[1.0, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         ])

# This is Q in wikipedia language. For a constant velocity form,
# it must take this specific form to be correct. The
# only free parameter here is `motion_noise_scale`.
motion_noise_scale = 100.0
observation_noise_covariance = 2.0*np.eye(3)

T33 = dt ** 3 / 3.0
T22 = dt ** 2 / 2.0
T = dt
motion_noise_covariance = motion_noise_scale*np.array(
    [
        [T33, 0, 0, T22, 0, 0],
        [0, T33, 0, 0, T22, 0],
        [0, 0, T33, 0, 0, T22],
        [T22, 0, 0, T, 0, 0],
        [0, T22, 0, 0, T, 0],
        [0, 0, T22, 0, 0, T],
    ]
)

# orthophoto = iio.imread('self-orthophoto-maker.png')

bee_traj_df = pd.read_csv(conf.out_filename('-bee.csv'))
assert len(bee_traj_df["ref_lat"].unique())==1
assert len(bee_traj_df["ref_lon"].unique())==1
assert len(bee_traj_df["ref_alt"].unique())==1
ref_lat = bee_traj_df.iloc[0]["ref_lat"]
ref_lon = bee_traj_df.iloc[0]["ref_lon"]
ref_alt = bee_traj_df.iloc[0]["ref_alt"]
reftime0 = pd.to_datetime(bee_traj_df.iloc[0]["reftime"])

copter_traj_df = pd.read_csv(conf.out_filename('-copter.csv'))
timeseries_svg = conf.out_filename('-timeseries.svg')
out_gpx = conf.out_filename('.gpx')

def gpx_format_dt(ts: pd.Timestamp) -> str:
    # e.g. 2024-06-27T19:08:55.501Z
    return ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def smooth_traj(traj_df):
    reftime = pd.to_datetime(traj_df['reftime'])
    traj_df['reftime'] = reftime
    traj_df.set_index('reftime', inplace=True, verify_integrity=True, drop=False)

    # target FPS
    fps = 1.0/dt
    resample_str = '%dus'%(int(1/fps * 1000000.0))
    traj_df = traj_df.resample(resample_str).mean()

    traj_df['since_start'] = np.arange(len(traj_df)) / fps


    observation = np.array([traj_df['east'].values,
                            traj_df['north'].values,
                            traj_df['up'].values,
    ]).T

    observation_model = np.zeros((3,6))
    observation_model[:3,:3] = np.eye(3)



    ## ------------------


    # Run kalman filter on the noisy observations.
    y = observation
    F = motion_model
    H = observation_model
    Q = motion_noise_covariance
    R = observation_noise_covariance
    initx = np.array(list(observation[0,:]) + [0,0,0])
    initV = 0.1*np.eye(6)

    xfilt, Vfilt = adskalman.kalman_smoother(y, F, H, Q, R, initx, initV)

    # ## ------------------


    traj_df['east_f'] = xfilt[:,0]
    traj_df['north_f'] = xfilt[:,1]
    traj_df['up_f'] = xfilt[:,2]
    return (traj_df, xfilt)

bee_traj_df, bee_smoothed = smooth_traj(bee_traj_df)
copter_traj_df, copter_smoothed = smooth_traj(copter_traj_df)

dx = bee_smoothed[1:,:] - bee_smoothed[:-1,:]

# export GPX
if out_gpx:
    gpx_time = gpx_format_dt(bee_traj_df.iloc[0].reftime)
    wgs84 = pyned2lla.wgs84()
    # east_ref_meters, north_ref_meters, z_ref_meters = pyned2lla.ned2lla(ref_lat, ref_lon, ref_alt, 0, 0, 0, wgs84)

    geo_coords = []
    for row_idx, bee_row in  bee_traj_df.iterrows():
        # Perform the coordinate transformation
        lat_rad, lon_rad, alt = pyned2lla.ned2lla(ref_lat*D2R, ref_lon*D2R, ref_alt, bee_row['north_f'], bee_row['east_f'], -bee_row['up_f'], wgs84)
        timestamp = gpx_format_dt(reftime0 + pd.to_timedelta(bee_row["since_start"], unit='seconds' ))
        geo_coords.append((lat_rad*R2D, lon_rad*R2D, alt, timestamp))
    geo_coords_arr = np.array(geo_coords)
    bee_traj_df['smooth_lat'] = geo_coords_arr[:,0]
    bee_traj_df['smooth_lon'] = geo_coords_arr[:,1]
    bee_traj_df['smooth_ele'] = geo_coords_arr[:,2]

    minlat = bee_traj_df['smooth_lat'].min()
    maxlat = bee_traj_df['smooth_lat'].max()
    minlon = bee_traj_df['smooth_lon'].min()
    maxlon = bee_traj_df['smooth_lon'].max()

    gpx_head = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    version="1.1" creator="Polar"
    xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
    <metadata>
        <author>
            <name>Polar</name>
        </author>
        <time>{gpx_time}</time>
        <bounds minlat="{minlat}" minlon="{minlon}" maxlat="{maxlat}" maxlon="{maxlon}"></bounds>
    </metadata>
    <trk>
        <trkseg>
"""
    with open(out_gpx, mode="w") as fd:
        fd.write(gpx_head)
        for pt in geo_coords:
            (lat, lon, alt, timestamp) = pt
            gpx_pt = f"""            <trkpt lat="{lat}" lon="{lon}">
                <ele>{alt}</ele>
                <time>{timestamp}</time>
            </trkpt>
"""
            fd.write(gpx_pt)
        gpx_tail = """        </trkseg>
    </trk>
</gpx>
"""
        fd.write(gpx_tail)


# displacement
displacement = np.sqrt(np.sum(dx[:,:3]**2, axis=1))
traj_len = displacement.sum()

speed_horiz = np.sqrt(np.sum(dx[:,3:5]**2, axis=1)) / dt

print("trajectory duration: %.1f s, length %.1f m"%(bee_traj_df['since_start'].iloc[-1],traj_len))

bee_offset = None
for fno in range(len(copter_traj_df['east_f'])):

    copter_row = copter_traj_df.iloc[fno]

    if bee_offset is None:
        bee_row = bee_traj_df.iloc[0]
        if bee_row.reftime == copter_row.reftime:
            bee_offset = fno
            bee_idx = 0
        else:
            bee_row = None
            bee_idx = None
    else:
        bee_idx = fno-bee_offset

    if bee_idx is not None:
        if bee_idx >= len(bee_traj_df):
            bee_idx = len(bee_traj_df)-1
        bee_row = bee_traj_df.iloc[bee_idx]

    if show_final_only:
        # Skip to last frame
        if fno < len(copter_traj_df['east_f'])-1:
            continue

    if show_final_only:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")

    if False:
        ortho_pix_h, ortho_pix_w, _ = orthophoto.shape
        ortho_l = -38
        ortho_b = -44
        ortho_w = 60
        ortho_h = ortho_w / ortho_pix_w * ortho_pix_h
        ax.imshow(orthophoto, extent=[ortho_l, ortho_l + ortho_w, ortho_b, ortho_b + ortho_h])

    if bee_idx is not None:
        ax.plot(bee_traj_df['east'][:bee_idx], bee_traj_df['north'][:bee_idx], 'wo', label='obs', markersize=3)
        ax.plot(bee_traj_df['east_f'][:bee_idx], bee_traj_df['north_f'][:bee_idx], 'k-', label='smooth', linewidth=3)
        ax.plot(bee_traj_df['east_f'][:bee_idx], bee_traj_df['north_f'][:bee_idx], 'w-', label='smooth', linewidth=2)

    ax.plot(copter_traj_df['east_f'][:fno], copter_traj_df['north_f'][:fno], 'k-', label='smooth', linewidth=3)
    ax.plot(copter_traj_df['east_f'][:fno], copter_traj_df['north_f'][:fno], 'y-', label='smooth', linewidth=2)

    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Position (m)')
    # ax.set_xlim([-10, 15])
    # ax.set_ylim([-30, 5])

    if show_final_only:
        plt.savefig(conf.out_filename('-topview.svg'), transparent=True)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis('off')
        ax.text(0, 0, "%s"%copter_row.reftime, size=9, color='white', transform=ax.transAxes)

        ax.set_frame_on(False)
        ax.patch.set_visible(False)
        fig.patch.set_visible(False)
        plt.savefig('topview-10fps-raw/topview_%04d.png'%fno, dpi=500)
        plt.close()

if False:
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax = axes[0]
    ax.plot(bee_traj_df['since_start'], bee_traj_df['east'], 'r.', label='obs')
    ax.plot(bee_traj_df['since_start'], bee_traj_df['east_f'], 'b-', label='smooth')
    ax.set_ylabel('Position East (m)')

    ax = axes[1]
    ax.plot(bee_traj_df['since_start'], bee_traj_df['north'], 'r.', label='obs')
    ax.plot(bee_traj_df['since_start'], bee_traj_df['north_f'], 'b-', label='smooth')
    ax.set_ylabel('Position North (m)')

    ax = axes[2]
    ax.plot(bee_traj_df['since_start'], bee_traj_df['up'], 'r.', label='obs')
    ax.plot(bee_traj_df['since_start'], bee_traj_df['up_f'], 'b-', label='smooth')
    ax.set_ylabel('Altitude (m)')

    axes[-1].set_xlabel('Time (s)')

if show_final_only:
    alt_offset = -bee_traj_df['up_f'].min() + 0.5

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax = axes[0]
    ax.plot(bee_traj_df['since_start'][:-1], speed_horiz, '-')
    ax.set_ylabel('Horizontal Speed (m/s)')
    # ax.set_ylim([0,10])

    ax = axes[1]
    ax.plot(bee_traj_df['since_start'], bee_traj_df['up_f'] + alt_offset, '-')
    ax.set_ylabel('Altitude (m)')
    # ax.set_ylim([0,6])

    ax = axes[2]
    ax.plot(bee_traj_df['since_start'][:-1], displacement.cumsum(), '-')
    ax.set_ylabel('Accumulated Displacement (m)')

    axes[-1].set_xlabel('Time (s)')
    # axes[-1].set_xlim([0,175])
    plt.savefig(timeseries_svg)

if show_final_only:
    plt.show()
