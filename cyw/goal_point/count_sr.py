'''
统计成功率
'''
import pickle

pkl_dir = "cyw/datasets/place_dataset/train/rl_agent_place_place_waypoint.pkl"

with open(pkl_dir,"rb") as f:
    pkl_data = pickle.load(f)
total_count = 0
total_success = 0 
for skill_waypoint_data in pkl_data[0]["skill_waypoint_data"]:
    view_point_count = 0
    recep_seccess = 0
    for each_view_point_data in skill_waypoint_data["each_view_point_data"]:
        view_point_count += 1
        recep_seccess += each_view_point_data['place_success']

    total_count += view_point_count
    total_success += recep_seccess

    recep_sr = recep_seccess/view_point_count
    print(f"recep_sr: {recep_sr}")
total_sr = total_success/total_count
print(f"total_sr:{total_sr}")

    


print("debug")