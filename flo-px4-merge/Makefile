default: output/20240712-134130.gpx

output/20240712-134130.rrd: merge_ulog_floz.py demo-data/20240712-134130.yaml
	python merge_ulog_floz.py demo-data/20240712-134130.yaml

output/20240712-134130.gpx: output/20240712-134130.rrd smooth_bee_traj.py demo-data/20240712-134130.yaml
	python smooth_bee_traj.py demo-data/20240712-134130.yaml
