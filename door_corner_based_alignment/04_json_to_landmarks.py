import json
import csv, os
from collections import defaultdict
import pandas as pd

json_path = "/home/jb/workspace/360video/msgfiles/0811_1stfloor/FWH_1stfloor_NL_08112025_2_original.json" ## what you converted json file from "02.py"
path = "/home/jb/workspace/360video/graph_codes/00_final/1stfloor_door_corner_result/" ## same directory with "03.py"

os.makedirs(path, exist_ok=True)
timestamp = os.path.join(path, "00_timestamps.csv")

timestamp_df = pd.read_csv(timestamp, header=None)
timestamp_data = timestamp_df[[1,0]].values.tolist()

detection_to_original_df = pd.DataFrame(timestamp_data, columns = ['original', 'keyframe'])
detection_to_original_df.to_excel(os.path.join(path, '01_keyframe_to_original.xlsx'),index=False)

with open(json_path, "r") as f:
    data = json.load(f)

keyframes = data["keyframes"]
landmarks = data["landmarks"]
landmark_observations = defaultdict(list)

keyframe_numbers = list(map(int,keyframes.keys()))
keyframe_numbers.sort()

# looking every keyframes, extract keypoints which are matched with landmarks
for kf_id in keyframe_numbers:
    kf = keyframes[str(kf_id)]
    keypoints = kf["undist_keypts"]
    lm_ids = kf["lm_ids"]

    for i, lm_id in enumerate(lm_ids):
        if lm_id == -1:
            continue  # if this keypoint does not have landmark, pass

        pt = keypoints[i]["pt"]
        landmark_observations[str(lm_id)].append({
            "keyframe_id": kf_id,
            "keypoint_idx": i,
            "2d_point_x": pt[0],
            "2d_point_y": pt[1]
        })

output_csv = os.path.join(path, "01_landmark_details.csv")

with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["landmark_id", "keyframe_id", "keypoint_idx", "2d_point_x", "2d_point_y"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for lm_id, obs_list in landmark_observations.items():
        for obs in obs_list:
            row = {
                "landmark_id": lm_id,
                "keyframe_id": obs["keyframe_id"],
                "keypoint_idx": obs["keypoint_idx"],
                "2d_point_x": obs["2d_point_x"],
                "2d_point_y": obs["2d_point_y"]
            }
            writer.writerow(row)

print(f"All landmark info is stored in '{output_csv}'")