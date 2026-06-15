import cv2
import msgpack
import numpy as np
import os
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

msgfile = "/home/jb/workspace/360video/msgfiles/0811_1stfloor/FWH_1stfloor_NL_08112025_2_original.msg" ## msgfile from original stella vslam
video_path = "/home/jb/workspace/360video/videofiles/FWH_1stfloor_NL_08112025_2.mp4" ### your original video directory
path = "/home/jb/workspace/360video/graph_codes/00_final/1stfloor_door_corner_result/" ### this directory will have all outputs. MUST remember this directory. This one will be used in the following python codes.

os.makedirs(path, exist_ok=True)

with open(msgfile, "rb") as f:
    u = msgpack.Unpacker(f)
    msg = u.unpack()

keyframes_data= msg["keyframes"]
keyframe_numbers = list(map(int,keyframes_data.keys()))
keyframe_numbers.sort()
timestamps =[]
timeline = []

for keyframe in keyframes_data.values():
    timestamps.append(keyframe["ts"])

initial_timestamp = timestamps[0]

# For conversion of timestamp to frame index
def convert_unix_timestamp_to_frame_index(unix_timestamp, initial_unix_timestamp, fps):

    time_difference = unix_timestamp - initial_unix_timestamp
    frame_index = int(time_difference * fps)
    return frame_index

vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)

keyfrm_points = []

i = 0
for key in keyframe_numbers:
    
    value = keyframes_data[str(key)]

    trans_cw = np.matrix(value["trans_cw"]).T
    rot_cw = R.from_quat(value["rot_cw"]).as_matrix()

    for_rot_cw = value["rot_cw"]
    rot_wc = rot_cw.T
    trans_wc = - rot_wc * trans_cw
    
    keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))

    frame_index = convert_unix_timestamp_to_frame_index(value['ts'], initial_timestamp, fps)
    timeline.append([frame_index,i])
    i+=1

np.savetxt((os.path.join(path,"00_timestamps.csv")), timeline, delimiter=",", fmt="%.0f")
np.savetxt((os.path.join(path,"00_keyframes.csv")), keyfrm_points, delimiter=",")
print("Finished")