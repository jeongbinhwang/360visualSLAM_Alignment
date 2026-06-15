import json
import csv, os
import pandas as pd
import msgpack
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

msgdata = "/home/jb/workspace/360video/msgfiles/0811_1stfloor/FWH_1stfloor_NL_08112025_2_original.msg" ## same msgfile with "03.py"
jsonDIR = "/home/jb/workspace/360video/graph_codes/GSAM_result/00_threshold_3030/FWH_1stfloor_NL_08112025_2/equirectangular/" ##this is the segmentaion results from "01.py". This directory should same as "01.py"  file's line 182.
path = "/home/jb/workspace/360video/graph_codes/00_final/1stfloor_door_corner_result/" ## same directory with "03.py"

landmarkCSV = os.path.join(path, "01_landmark_details.csv")
keyframe_excel = os.path.join(path,"01_keyframe_to_original.xlsx")

jsonfiles = [
    os.path.join(jsonDIR,p) for p in os.listdir(jsonDIR)
    if os.path.splitext(p)[-1] in [".json"]
]
jsonfiles.sort()

keyframe_data = pd.read_excel(keyframe_excel)

k_original = keyframe_data['keyframe'].tolist()
k_keyframe = keyframe_data['original'].tolist()

landmarkDATA = pd.read_csv(landmarkCSV)

landmark_as_door = []
landmark_as_door_dict = {}

for jsonfile in jsonfiles:
    if "summary" in jsonfile:
        continue
    original_detectionNO = int(jsonfile.split('/')[-1].replace(".json",""))
    print("Original Frame ID: ",original_detectionNO)
    if original_detectionNO in k_original:
        keyframeNO = k_keyframe[k_original.index(original_detectionNO)]
        candidate_feature = landmarkDATA[landmarkDATA['keyframe_id'] == keyframeNO].values.tolist()

        with open(jsonfile, "r") as f:
            data = json.load(f)
        
        # detection_results = data['labels']
        detection_results = {}
        for dt in data:
            detection_results[str(dt['id'])] = dt['corners']
            # detection_results[str(dt['id'])] = dt['corners']

        ids = detection_results.keys()
        
        for oid in ids:
            target_points = detection_results[oid]


            for i in range(len(candidate_feature)):
                x = float(candidate_feature[i][3])
                y = float(candidate_feature[i][4])

                for j in range(len(target_points)):
                    x_diff = x - float(target_points[j][0])
                    y_diff = y - float(target_points[j][1])

                    distance = (x_diff*x_diff + y_diff*y_diff)**0.5
                    if distance < 5:
                        landmarkid_door = int(candidate_feature[i][0])
                        if j == 0:
                            print(f"Door {oid}'s Left Top corner is Landmark {landmarkid_door}")
                        elif j == 1:
                            print(f"Door {oid}'s Right Top corner is Landmark {landmarkid_door}")
                        elif j == 2:
                            print(f"Door {oid}'s Right Bottom corner is Landmark {landmarkid_door}")
                        else:
                            print(f"Door {oid}'s Left Bottom corner is Landmark {landmarkid_door}")
                        landmark_as_door.append(landmarkid_door)
                        
                        if oid in landmark_as_door_dict.keys():
                            landmark_as_door_dict[oid].append(landmarkid_door)
                        else:
                            landmark_as_door_dict[oid] = [landmarkid_door]

print("Landmark IDs which have corresponding door corners. ",landmark_as_door)

with open(msgdata, "rb") as f:
    u = msgpack.Unpacker(f)
    msg = u.unpack()

keyframes_data= msg["keyframes"]
keyframe_numbers = list(map(int,keyframes_data.keys()))
keyframe_numbers.sort()
landmarks_data = msg["landmarks"]
landmark_points = []

## for visualization - keyframes
keyfrm_points = []
for key in keyframe_numbers:
    
    value = keyframes_data[str(key)]

    trans_cw = np.matrix(value["trans_cw"]).T
    rot_cw = R.from_quat(value["rot_cw"]).as_matrix()

    rot_wc = rot_cw.T
    trans_wc = - rot_wc * trans_cw
    
    keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))

translate_keyframe = np.matrix(keyframes_data[str(keyframe_numbers[0])]["trans_cw"]).T
rotation_keyframe = R.from_quat(keyframes_data[str(keyframe_numbers[0])]["rot_cw"]).as_matrix()

for landmark in landmark_as_door:
    landmark_actual = rotation_keyframe.T * np.matrix(landmarks_data[str(int(landmark))]["pos_w"]).T - rotation_keyframe.T * translate_keyframe
    landmark_points.append([float(landmark_actual[0][0]),float(landmark_actual[1][0]),float(landmark_actual[2][0])])
np.savetxt((os.path.join(path,"02_landmarks_door_mask.csv")),landmark_points,delimiter=",")

keyfrm_points = np.array(keyfrm_points)
landmark_points = np.array(landmark_points)

# kmeans = KMeans(n_clusters=2)
# labels = kmeans.fit_predict(landmark_points)

# landmark_points_label0 = landmark_points[labels == 0]

dbscan_landmarks = DBSCAN(eps=10, min_samples=1)
labels = dbscan_landmarks.fit_predict(landmark_points)
print("DBSCAN labels: ",set(labels))
landmark_points_label0 = landmark_points[labels == 0]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(keyfrm_points[:, 0], keyfrm_points[:, 1], keyfrm_points[:, 2], c='b', marker='o')
# ax.scatter(landmark_points[:, 0], landmark_points[:, 1], landmark_points[:, 2], c='g', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.axis('equal')
# plt.show()

# print(landmark_as_door_dict)

# keyfrm_points PointCloud 
pcd_keyfrm = o3d.geometry.PointCloud()
pcd_keyfrm.points = o3d.utility.Vector3dVector(keyfrm_points)
pcd_keyfrm.paint_uniform_color([0, 0, 1])  # blue (RGB)

# landmark_points PointCloud 
pcd_landmark = o3d.geometry.PointCloud()
pcd_landmark.points = o3d.utility.Vector3dVector(landmark_points)
pcd_landmark.paint_uniform_color([0, 1, 0])  # green (RGB)

# pcd_landmark = o3d.geometry.PointCloud()
# pcd_landmark.points = o3d.utility.Vector3dVector(landmark_points)
# pcd_landmark.paint_uniform_color([1, 0, 0])  # green (RGB)

o3d.visualization.draw_geometries([pcd_keyfrm, pcd_landmark], window_name="Keyframes and Landmarks", point_show_normal=False, width=800, height=600)