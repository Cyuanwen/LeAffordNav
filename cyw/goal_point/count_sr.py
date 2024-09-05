'''
统计成功率
'''
import pickle
from pathlib import Path
import os
import json

# 对比来看,原本的策略更好一些
pkl_dir = "cyw/datasets/place_dataset/val/heuristic_agent_esc_yolo_place_cyw/multi_thread/place_waypoint_6.pkl"
# pkl_dir = "cyw/datasets/place_dataset_debug/val/heuristic_agent_place_initial/place_waypoint.pkl"

parent_dir = str(Path(pkl_dir).resolve().parent)

with open(pkl_dir,"rb") as f:
    pkl_data_s = pickle.load(f)
total_count = 0
total_success = 0 
sr_dict = {}
for pkl_data in pkl_data_s:
    scene_id = pkl_data["scene_id"]
    episode_id = pkl_data['episode_id']
    for skill_waypoint_data in pkl_data["skill_waypoint_data"]:
        recep_position = skill_waypoint_data['recep_position']
        view_point_count = 0
        recep_seccess = 0
        for each_view_point_data in skill_waypoint_data["each_view_point_data"]:
            view_point_count += 1
            recep_seccess += each_view_point_data['place_success']

        total_count += view_point_count
        total_success += recep_seccess

        recep_sr = recep_seccess/view_point_count
        sr_dict[f"{scene_id}_{episode_id}_{recep_position}"] = recep_sr
        print(f"recep_sr: {recep_sr}")
total_sr = total_success/total_count
print(f"total_sr:{total_sr}")
# save json file
print(f"save file to {os.path.join(parent_dir,'sr_count.json')}")
with open(os.path.join(parent_dir,"sr_count.json"),"w") as f:
    json.dump(sr_dict,f,indent=2)
print("save done ----------------")


    


print("debug")