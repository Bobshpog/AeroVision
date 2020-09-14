# This module will take the raw files supplied by the pi cameras, IR cameras and IMU and transforms it to a unified format
# Input/Output for each instance of continuous movement:

# Input:
# IR - Supplies CSV (Example in Onedrive), module should support both 120hz and 240 hz and provide datetime of start
# IMU - Supplies txt with 6 Dimensions of movement in txt format with timestamps  (Example in Onedrive)
# Cameras - video named <mac>_<datetime of start>.mp4
# 
# Folder structure:
#                  /data/<lab or wind tunnel>/vid_<datetime>/manifest_vid_<datetime>
#                  /data/lab/vid_<datetime>/ir.csv
#                  /data/lab/vid_<datetime>/cam<id>/cam<id>_<frame_num>.png
#                  /data/lab/vid_<datetime>/imu<id>.txt
#
# Output:
# manifest_vid_<datetime>:
# |IR|---|IR||CAM0(links to files)|---|CAM99|IMU0|---|IMU99|
# We synchronize the diffrenet frequencies by saving the highest one and replicaing previous entries in between slow device updates

import csv
from pathlib import Path


class Config:
    num_points=28
    num_cams=1
    num_imus=1
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>

# input folder must include: an input file for each device:
# cam<id>_<timestamp>.mp4
# imu<id>_<timestamp>.txt
# ir.csv
def unify(path_to_input_folder, path_to_data_folder, cam_num, imu_num=1):
    input_folder=Path(path_to_input_folder)
    output_folder=Path(path_to_data_folder)
    with open(input_folder /"ir.csv","r") as ir_csv, open(output_folder / "manifest.csv","w") as manifest_csv:
        pass

with open('data/data_samples/IR Lab conditions.csv') as csvfile:
    content=csv.DictReader(csvfile)
    for i in content:
        print (i)
print(df)