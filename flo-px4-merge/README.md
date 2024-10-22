# FLO PX4 merge

Merge data from FLO and PX4 flight logs to allow georeferencing of data saved by FLO.

Pan, tilt and distance estimates from FLO are converted from their intrinsic
quadcopter-centric to georeferenced coordinates first by converting them to the
local coordinate frame used by the quadcopter’s flight controller. Local
coordinates are then converted to global coordinates using the PX4 reference
point (local NED frame origin) in global (GPS / WGS84) frame.

## data acquisition

Ideally, the PX4 flight controller is logging at high rate but the defaults are
relatively slow.

As a solution, set the PX4 parameter
[`SDLOG_PROFILE`](https://docs.px4.io/main/en/advanced_config/parameter_reference.html#SDLOG_PROFILE)
to a value of 4 (high rate).

Alternatively, following [these
instructions](https://docs.px4.io/main/en/dev_log/logging.html), high data rate
logs can be enabled by setting `etc/logging/logger_topics.txt` to the following:

```
vehicle_local_position
vehicle_gps_position
vehicle_attitude
```

However, this disables all other logging, which makes the logs largely useless
for many other purposes. Thus, we suggest changing the `SDLOG_PROFILE` parameter
as described above.

## setup

```
conda create -n px4-log-analysis python=3.11
pip install -r requirements.txt
```

## run on demo data

Download `movie20240712_134113.505531834_Basler-40116283.mp4` and `DJI_0010.mp4`
from the `Fig4-Movie2-Apis` folder in the `flo-dryad.zip` file at
https://doi.org/10.5061/dryad.bvq83bkjr and place them in this folder, alongside
`Makefile`.

This will generate several output files, including rerun `.rrd` files, in the
`output/` directory:

```
conda activate px4-log-analysis
make
```

To view the output files, run rerun:

```
rerun output/*.rrd
```
