default: output/20240712-134130.gpx output/20240712-134130.rrd output/DJI_0010-codec-copy.h264.rrd output/movie20240712_134113.505531834_Basler-40116283.mp4.rrd

output/20240712-134130.rrd: merge_ulog_floz.py demo-data/20240712-134130.yaml
	python merge_ulog_floz.py demo-data/20240712-134130.yaml

output/20240712-134130.gpx: output/20240712-134130.rrd smooth_bee_traj.py demo-data/20240712-134130.yaml
	python smooth_bee_traj.py demo-data/20240712-134130.yaml

output/DJI_0010-codec-copy.h264: DJI_0010.MOV
	ffmpeg -i DJI_0010.MOV -vcodec copy output/DJI_0010-codec-copy.h264

output/DJI_0010-codec-copy.h264.rrd: output/DJI_0010-codec-copy.h264
	video2rrd --start-time "2024-07-12 11:41:31.609Z" --framerate 60.0 --input output/DJI_0010-codec-copy.h264 --recording-id b7ba4662-2fd5-49c6-8fb9-f961a8f596b4  --entity-path world/copter/fpv_cam --output output/DJI_0010-codec-copy.h264.rrd

output/movie20240712_134113.505531834_Basler-40116283.mp4.rrd:
	video2rrd --input movie20240712_134113.505531834_Basler-40116283.mp4 --recording-id b7ba4662-2fd5-49c6-8fb9-f961a8f596b4 --entity-path world/copter/gimbal --output output/movie20240712_134113.505531834_Basler-40116283.mp4.rrd
