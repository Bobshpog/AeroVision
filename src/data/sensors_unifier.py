# This module will take the raw files supplied by the pi cameras, IR cameras and IMU and transforms it to a unified format
# Input/Output for each instance of continuous movement:

# Input:
# IR - Supplies CSV (Example in Onedrive), module should support both 120hz and 240 hz and provide datetime of start
# IMU - Supplies txt with 6 Dimensions of movement in txt format with timestamps  (Example in Onedrive)
# Cameras - video named <mac>_<datetime of start>.mp4
# Manifest - CSV called /data/manifest.csv
# 
# Folder structure:
#                  /data/manifest.csv - mapping between mac addresses of pis and names
#                  /data/<lab or wind tunnel>/vid_<datetime>/manifest_vid_<datetime>
#                  /data/lab/vid_<datetime>/ir.csv
#                  /data/lab/vid_<datetime>/cam<id>/cam<id>_<frame_num>.png
#                  /data/lab/vid_<datetime>/imu<id>.txt
#
# Output:
# manifest_vid_<datetime>:
# |IR|---|IR||CAM0(links to files)|---|CAM99|IMU0|---|IMU99|
# We synchronize the diffrenet frequencies by saving the highest one and replicaing previous entries in between slow device updates
