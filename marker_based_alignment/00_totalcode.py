import cv2
import msgpack
import numpy as np
import os, random
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import csv
import ifcopenshell
import ifcopenshell.geom
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
import time

# ijk = 6

# msgfile = msgfiles[ijk-3]
msgfile = "/home/jb/workspace/360video/msgfiles/0811_1stfloor/FWH_1stfloor_NL_08112025_2_4_markers_1e6_not_using_4th.msg" ## need msgfile from stella vslam (Shitikantha's changed code).
video_path = "/home/jb/workspace/360video/videofiles/FWH_1stfloor_NL_08112025_2.mp4"  ##original video file.



path = os.path.join("/home/jb/workspace/360video/timestamp_result/01142026_1stfloor/", msgfile.split("/")[-1].replace(".msg","")) ## where you want to save the results.
ifc_path = "/home/jb/workspace/IFCdata/FWH/FWH_Architectural.ifc" ##BIM model 

transformation_method_list = ['basic', 'umeyama', 'ransac']
transformation_method = transformation_method_list[1] ## you can choose tranformation method what you want.

floor = 1 ## choose your floor for the real life marker coordinates

markernumber = 4
##############################################################################################################
##################################### Define real-life marker coordinates#####################################
##############################################################################################################
# ## 1st floor 0811
# if ijk == 3:
#     marker_coordinates = [
#         [44.136, 71.005, 2.026],
#         [13.695, 22.581, 1.852],
#         [48.994, 38.549, 1.694]
#     ]
#     markerIDs = [2,4,6]
# elif ijk == 4:
#     marker_coordinates = [
#         [44.136, 71.005, 2.026],
#         [29.478, 52.560, 2.015],
#         [29.291, 12.799, 2.005],
#         [48.994, 38.549, 1.694]
#     ]
#     markerIDs = [2,3,5,6]
# elif ijk == 5:
#     marker_coordinates = [
#         [44.136, 71.005, 2.026],
#         [29.478, 52.560, 2.015],
#         [13.695, 22.581, 1.852],
#         [29.291, 12.799, 2.005],
#         [48.994, 38.549, 1.694]
#     ]
#     markerIDs = [2,3,4,5,6]
# elif ijk == 6:
#     marker_coordinates = [
#         [60.341, 60.562, 2.025],
#         [44.136, 71.005, 2.026],
#         [29.478, 52.560, 2.015],
#         [13.695, 22.581, 1.852],
#         [29.291, 12.799, 2.005],
#         [48.994, 38.549, 1.694]
#     ]
#     markerIDs = [1,2,3,4,5,6]
if floor == 1:
    ## 1st floor 0811
    # marker_coordinates = [
    #     [60.341, 60.562, 2.025],
    #     [44.136, 71.005, 2.026],
    #     [29.478, 52.560, 2.015],
    #     [13.695, 22.581, 1.852],
    #     [29.291, 12.799, 2.005],
    #     [48.994, 38.549, 1.694]
    # ]
    marker_coordinates = [
        [44.136, 71.005, 2.026],
        [29.478, 52.560, 2.015],
        [29.291, 12.799, 2.005],
        [48.994, 38.549, 1.694]
    ]

elif floor == 2:
    ## 2nd floor 0724
    marker_coordinates = [
        [47.313, 76.238, 1.887],
        [60.341, 59.056, 1.708],
        [25.921, 9.440, 1.721],
        [15.191, 25.605, 1.714],
        [10.008, -2.397, 1.664],
        [26.201, 42.959, 1.771]
    ]

elif floor == 3:
    ## 3rd floor 0610
    # marker_coordinates = [
    #     [47.323, 76.232, 2.002],
    #     [52.005, 44.403, 1.761],
    #     [24.525, 7.214, 1.753],
    #     [15.279, 25.601, 1.798]
    # ]
    ## 3rd floor 0724
    marker_coordinates = [
        [47.314, 76.237, 2.012],
        [52.011, 44.413, 1.769],
        [24.531, 7.223, 1.782],
        [15.273, 25.592, 1.714],
        [11.623, 0.093, 1.664],
        [32.393, 52.887, 1.771]
    ]

elif floor == 4:
    ## 4th floor 0721
    marker_coordinates = [
        [47.306, 76.242, 2.057],
        [56.366, 51.461, 1.750],
        [26.025, 9.606, 1.707],
        [14.816, 29.530, 1.602],
        [11.629, 0.102, 1.734],
        [28.318, 46.392, 1.718]
    ]
else:
    print("Floor Definition is Wrong.")
    exit()

##############################################################################################################
##############################################################################################################
##############################################################################################################

marker_coordinates = marker_coordinates[:markernumber]
real_life_markers = np.array(marker_coordinates)

os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path,"images"), exist_ok=True)

with open(msgfile, "rb") as f:
    u = msgpack.Unpacker(f)
    msg = u.unpack()

keyframes_data= msg["keyframes"]
keyframe_numbers = list(map(int,keyframes_data.keys()))
keyframe_numbers.sort()
landmarks_data = msg["landmarks"]
timestamps =[]
timeline = []
markers_data = msg["markers"]

for keyframe in keyframes_data.values():
    timestamps.append(keyframe["ts"])

# for initial timestamp 
initial_timestamp = timestamps[0]

# For conversion of timestamp to frame index
def convert_unix_timestamp_to_frame_index(unix_timestamp, initial_unix_timestamp, fps):
    # Calculate time difference in seconds
    time_difference = unix_timestamp - initial_unix_timestamp
    frame_index = int(time_difference * fps)

    return frame_index


vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)

keyfrm_points = []
keyframepoints = []
euler_angles_list = []
rot_cw_list = []

for key in keyframe_numbers:
    
    value = keyframes_data[str(key)]
    #key has the keyframe number

    # get conversion from camera to world
    trans_cw = np.matrix(value["trans_cw"]).T
    # print(value["rot_cw"])
    rot_cw = R.from_quat(value["rot_cw"]).as_matrix()

    for_rot_cw = value["rot_cw"]
    # compute conversion from world to camera
    rot_wc = rot_cw.T
    trans_wc = - rot_wc * trans_cw
    # Euler angles calculation
    euler_angles = R.from_matrix(rot_wc).as_euler('xyz', degrees=True)
    euler_angles_list.append((euler_angles[0], euler_angles[1], euler_angles[2]))
    #print((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
    
    keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
    keyframepoints.append([trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]])

    frame_index = convert_unix_timestamp_to_frame_index(value['ts'], initial_timestamp, fps)
    timeline.append(frame_index)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = vidcap.read()

    if ret:
        #print(f"Found frame at Unix timestamp {keyfrm['ts']}, Frame Index: {frame_index}")
        cv2.imwrite(os.path.join(path,'images',(str(key).zfill(6)+".jpg")),frame)
        # Use the frame as needed
    else:
        print(f"Error: Frame not found for Unix timestamp {value['ts']}")
    # print(for_rot_cw)
np.savetxt((os.path.join(path,"timestamps.csv")), timeline, delimiter=",")
np.savetxt((os.path.join(path,"keyframes.csv")), keyfrm_points, delimiter=",")
np.savetxt((os.path.join(path,"euler_angles.csv")), euler_angles_list, delimiter=",")
print("Finished")
now = time.time()
# Markers
markerIDs_basic = [1,2,3,4,5,6]


if floor == 1:
    markerIDs = [2,3,5,6]
else:
    markerIDs = markerIDs_basic[:markernumber]
markerpoints = []

#extract marker centers
for markerID in markerIDs:
    target_marker = markers_data[str(markerID)]

    marker_center = [
        float(target_marker["corners_pos_w_"][0][0]),
        float(target_marker["corners_pos_w_"][0][1]),
        float(target_marker["corners_pos_w_"][0][2])
    ]

    marker_center = np.matrix(marker_center).T

    rot_cw_marker = R.from_quat(keyframes_data[str(keyframe_numbers[0])]["rot_cw"]).as_matrix()
    trans_cw_marker = np.matrix(keyframes_data[str(keyframe_numbers[0])]["trans_cw"]).T

    rot_wc_marker = rot_cw_marker.T
    trans_wc_marker = - rot_wc_marker @ trans_cw_marker

    marker_coordinate = rot_wc_marker @ marker_center + trans_wc_marker

    markerpoints.append([float(marker_coordinate[0][0]),float(marker_coordinate[1][0]),float(marker_coordinate[2][0])])
    print("Marker ID: ",markerID)
    print("Marker Coordinate: ", [float(marker_coordinate[0][0]),float(marker_coordinate[1][0]),float(marker_coordinate[2][0])])

# Landmark points
landmark_points = []

for lm in landmarks_data.values():
    translate_keyframe = np.matrix(keyframes_data[str(keyframe_numbers[-1])]["trans_cw"]).T
    rotation_keyframe = R.from_quat(keyframes_data[str(keyframe_numbers[-1])]["rot_cw"]).as_matrix()

    landmark_actual = rotation_keyframe.T * np.matrix(lm["pos_w"]).T - rotation_keyframe.T * translate_keyframe
    landmark_points.append([float(landmark_actual[0][0]),float(landmark_actual[1][0]),float(landmark_actual[2][0])])
np.savetxt((os.path.join(path,"landmarks.csv")),landmark_points,delimiter=",")

def umeyama_similarity(src, dst, with_scaling=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    N, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / N
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(dim)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1

    R = U @ D @ Vt

    if with_scaling:
        var_src = (src_c**2).sum() / N
        scale = np.trace(np.diag(S) @ D) / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    T = np.eye(dim+1)
    T[:dim, :dim] = scale * R
    T[:dim, -1] = t

    return T, scale, R, t

def find_robust_transform(src_pts, dst_pts, alpha=2.0):
    # 1) 초기 변환
    T0, s0, R0, t0 = umeyama_similarity(src_pts, dst_pts, with_scaling=True)
    pred0 = (s0 * (R0 @ src_pts.T)).T + t0

    res = np.linalg.norm(pred0 - dst_pts, axis=1)
    print("Initial residuals:", res)

    med = np.median(res)
    mad = np.median(np.abs(res - med))
    thresh = med + alpha * mad
    mask_inlier = (res < thresh)

    if mask_inlier.sum() < 3:
        print("Not enough inliers ({}); using all markers".format(mask_inlier.sum()))
        return T0, s0, R0, t0


    T1, s1, R1, t1 = umeyama_similarity(src_pts[mask_inlier], dst_pts[mask_inlier], with_scaling=True)
    pred1 = (s1 * (R1 @ src_pts.T)).T + t1
    res1 = np.linalg.norm(pred1 - dst_pts, axis=1)
    print("Refined residuals:", res1)

    return T1, s1, R1, t1

def ransac_similarity(src_pts, dst_pts, 
                      n_iter=100, n_min=3, threshold=1.0, random_seed=None):

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    N = src_pts.shape[0]
    best_inliers = []
    best_model = None

    for i in range(n_iter):
        idx = random.sample(range(N), n_min)
        try:
            T_i, _, _, _ = umeyama_similarity(src_pts[idx], dst_pts[idx], with_scaling=True)
        except np.linalg.LinAlgError:
            continue

        pred = apply_transform_umeyama(src_pts, T_i)
        residuals = np.linalg.norm(pred - dst_pts, axis=1)

        inlier_mask = residuals < threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > len(best_inliers):
            best_inliers = inlier_mask
            best_model = T_i

    if best_model is None or best_inliers.sum() < n_min:
        raise RuntimeError("RANSAC failed to find a valid model")

    T_refined, scale, R, t = umeyama_similarity(
        src_pts[best_inliers], dst_pts[best_inliers], with_scaling=True
    )
    print(f"RANSAC inliers: {best_inliers.sum()}/{N}")
    print("Inlier indices:", np.where(best_inliers)[0])

    return T_refined, scale, R, t, best_inliers

def find_similarity_transformation(source_points, target_points):
    assert source_points.shape == target_points.shape
    N = source_points.shape[0]

    # Center the points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Compute covariance matrix
    H = source_centered.T @ target_centered / N

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection (det(R) = -1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_source = np.var(source_centered, axis=0).sum()  # total variance
    scale = np.sum(S) / var_source

    # Compute translation
    t = centroid_target - scale * R @ centroid_source

    # Create homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T, scale, R, t

def apply_transform_umeyama(pts, T):
    hp = np.hstack((pts, np.ones((len(pts),1))))
    return (T @ hp.T).T[:, :3]

def apply_transformation_similiarity(points, T):
    # points: (N, 3)
    N = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((N, 1))))  # (N, 4)
    transformed = (T @ homogeneous_points.T).T  # (N, 4)
    return transformed[:, :3]  # remove homogeneous coordinate

transformed_csvDIR = os.path.join(path, "keyframes_transformed.csv")
detected_markers = np.array(markerpoints)

if transformation_method == 'basic':
    T, scale, R, t = find_similarity_transformation(detected_markers, real_life_markers)
    source_points = pd.read_csv(os.path.join(path,'keyframes.csv'), header=None).values  # (N, 3)
    transformed_points = apply_transformation_similiarity(source_points, T)
    transformed_markers = apply_transformation_similiarity(detected_markers, T)
    pd.DataFrame(transformed_points).to_csv(transformed_csvDIR, header=False, index=False)

elif transformation_method == 'umeyama':
    T, scale, R, t = find_robust_transform(detected_markers, real_life_markers, alpha=2.0)
    source_points = pd.read_csv(os.path.join(path,'keyframes.csv'), header=None).values
    transformed_points = apply_transform_umeyama(source_points, T)
    transformed_markers = apply_transform_umeyama(detected_markers, T)
    pd.DataFrame(transformed_points).to_csv(
        transformed_csvDIR,
        header=False, index=False
    )

elif transformation_method == 'ransac':
    T_ransac, scale, R, t, inliers = ransac_similarity(
        detected_markers, real_life_markers,
        n_iter=200, n_min=3, threshold=2.0, random_seed=42
    )
    source_points = pd.read_csv(os.path.join(path,'keyframes.csv'), header=None).values
    transformed_points = apply_transform_umeyama(source_points, T_ransac)
    transformed_markers = apply_transform_umeyama(detected_markers, T_ransac)
    pd.DataFrame(transformed_points).to_csv(transformed_csvDIR, header=False, index=False)
    print("Saved transformed points to:", transformed_csvDIR)

else: 
    print("Transformation Method is Wrong. Will be processed by using basic transformation method.")
    T, scale, R, t = find_similarity_transformation(detected_markers, real_life_markers)
    source_points = pd.read_csv(os.path.join(path,'keyframes.csv'), header=None).values  # (N, 3)
    transformed_points = apply_transformation_similiarity(source_points, T)
    transformed_markers = apply_transformation_similiarity(detected_markers, T)
    pd.DataFrame(transformed_points).to_csv(transformed_csvDIR, header=False, index=False)

transformed_markers = transformed_markers.squeeze().tolist()

print("Whole time, ",time.time()-now)
exit()
# print("transformed_markers: ", transformed_markers)

# IFC
model = ifcopenshell.open(ifc_path)

# Geometry
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_PYTHON_OPENCASCADE, True)

exclude_guids = {"1kCsIarjLCq8yHmVMBUpKt", "0E18YStXvDAQkkOoQUUVgV","0E18YStXvDAQkkOoQUUVhg"}

if floor == 1:
    level_type = 'level 1'
    level_type_list = []
    level_threshold = 0 ##these heights should be manually measured by the user from the IFC model.
elif floor == 2:
    level_type = 'level 2'
    level_type_list = []
    level_threshold = 8.128 ##these heights should be manually measured by the user from the IFC model.
elif floor == 3:
    level_type = 'level 3'
    level_type_list = ['level 2']
    level_threshold = 15.24 ##these heights should be manually measured by the user from the IFC model.
elif floor == 4:
    level_type = 'level 4'
    level_type_list = ['level 2']
    level_threshold = 20.117 ##these heights should be manually measured by the user from the IFC model.
else:
    print("Check Level.")


level_type_list.append(level_type)
print("level_type_list", level_type_list)

# 1. Level storey
storeys = model.by_type("IfcBuildingStorey")
target_storey = []
for s in storeys:
    if s.Name.lower() in level_type_list:
        target_storey.append(s)

if len(target_storey) == 0:
    raise ValueError(level_type + " storey not found in IFC model.")

level1_products = []
for rel in model.by_type("IfcRelContainedInSpatialStructure"):
    if rel.RelatingStructure in target_storey:
        level1_products.extend(rel.RelatedElements)

z0_edges = []
for product in level1_products:
    if product.GlobalId in exclude_guids:
        continue
    try:
        shape = ifcopenshell.geom.create_shape(settings, product)
        occ_shape = shape.geometry

        exp_edge = TopExp_Explorer(occ_shape, TopAbs_EDGE)
        while exp_edge.More():
            edge = exp_edge.Current()
            exp_vertex = TopExp_Explorer(edge, TopAbs_VERTEX)
            points = []
            while exp_vertex.More():
                vertex = exp_vertex.Current()
                pnt = BRep_Tool.Pnt(vertex)
                coord = (pnt.X(), pnt.Y(), pnt.Z())
                points.append(coord)
                exp_vertex.Next()

            if len(points) == 2:
                z_coords = [pt[2] for pt in points]
                if all(abs(z-level_threshold) < 0.1 for z in z_coords):
                    z0_edges.append(points)
            exp_edge.Next()
    except:
        continue

## transformed csv file
with open(transformed_csvDIR) as file:
    transformed_csv_data = []
    for line in file.readlines():
        tmp_line = line.split('\t')[0].split(',')
        transformed_csv_data.append([float(tmp_line[0]),float(tmp_line[1])])
print(len(transformed_csv_data))

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111)

final_edges = []

for edge in z0_edges:
    x = [edge[0][0], edge[1][0]]
    y = [edge[0][1], edge[1][1]]
    ax.plot(x, y, 'k-', alpha=0.15)
    final_edges.append([edge[0][0], edge[1][0], edge[0][1], edge[1][1]])

# with open(gtfile) as gfile:
#     gt_data = []
#     for line in gfile.readlines():
#         tmp_line = line.split('\t')[0].split(',')
#         gt_data.append([float(tmp_line[0]),float(tmp_line[1])])

# x_coords_gt = [ptgt[0] for ptgt in gt_data]
# y_coords_gt = [ptgt[1] for ptgt in gt_data]

# ax.scatter(x_coords_gt, y_coords_gt, c='r', marker='o', s = 3, label = "Ground Truth")

x_coords = [pt[0] for pt in transformed_csv_data]
y_coords = [pt[1] for pt in transformed_csv_data]

# targetname = 'OpenVSLAM with Loop Closure'
targetname = msgfile.split("/")[-1].replace(".msg","")
ax.scatter(x_coords, y_coords, c='b', marker='o', s = 3, label = targetname)

# x_coords_marker = [pt1[0] for pt1 in transformed_markers]
# y_coords_marker = [pt1[1] for pt1 in transformed_markers]

# ax.scatter(x_coords_marker, y_coords_marker, c='r', marker='o', s = 15, label = 'Reference Point (Estimated)')

# x_coords_marker_r = [pt2[0] for pt2 in marker_coordinates]
# y_coords_marker_r = [pt2[1] for pt2 in marker_coordinates]

# ax.scatter(x_coords_marker_r, y_coords_marker_r, c='g', marker='o', s = 15, label = 'Reference Point (Actual)')

diff = 0
print("transformed_markers ",transformed_markers)
print("marker_coordinates ",marker_coordinates)
for m in range(len(transformed_markers)):
    for n in range(3):
        diff += (transformed_markers[m][n] - marker_coordinates[m][n]) ** 2
RMSE = (diff / len(transformed_markers)) ** 0.5
print("RMSE of markers are ", RMSE)

# ax.set_title("Z=0 Edges from IFC Model")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis('equal')
ax.legend()
plt.tight_layout()
plt.show()
