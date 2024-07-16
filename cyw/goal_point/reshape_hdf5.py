'''
打补丁：原本数据用concate将所有数据组合起来，不利于读取，将其reshape后放回原始文件
'''
import h5py
import os
import pickle
from tqdm import tqdm
import numpy as np

'''
read data time 0.0023729801177978516
reshape time is 0.2709023952484131

reshape time 是读取时间的上百倍，十分有必要提前reshape

rgb_hfd5文件组织格式
    scene_ep_recep_grp = dataset_file.create_group(f"/scene_{scene_id}/ep_{episode.episode_id}/{recep_position}") 

    scene_ep_recep_grp.create_dataset(name="start_rgb_s",data=start_rgb_s)
    scene_ep_recep_grp.create_dataset(name="start_semantic_s",data=start_semantic_s)
    scene_ep_recep_grp.create_dataset(name="start_depth_s",data=start_depth_s)
    scene_ep_recep_grp.create_dataset(name="start_top_down_map_s",data=start_top_down_map_s)
    scene_ep_recep_grp.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
    scene_ep_recep_grp.create_dataset(name="view_point_position_s",data=view_point_position_s)

waypoint_data 数据组织格式
    total_data.append(scene_ep_data)
        scene_ep_data = {
            "scene_id": scene_id,
            "episode_id":episode.episode_id,
            "recep": recep,
            "object_name": object_name,
            "skill_waypoint_data": []
        }
            scene_ep_data["skill_waypoint_data"].append(skill_waypoint_singile_recep_data)

                skill_waypoint_singile_recep_data = {
                        "recep_position": recep_position,
                        "each_view_point_data":[]
                    }
                skill_waypoint_singile_recep_data["each_view_point_data"].append(
                                {
                                    "view_point_position":view_point_position,
                                    "start_position":start_position,
                                    "start_rotation":start_rotation,
                                    "relative_recep_position": relative_recep_gps,
                                    "end_position": end_position,
                                    "place_success": place_success,
                                    "start_sensor_pose": start_sensor_pose, # 在obstacle map里面的位置
                                    "start_top_down_map_pose": start_top_down_map_pose,
                                    "start_top_down_map_rot": start_top_down_map_rot
                                }
                            )
'''

if __name__ == "__main__":
    data_dir = "cyw/datasets/place_dataset/val/heuristic_agent_place"
    h5py_file = h5py.File(os.path.join(data_dir,"data_out.hdf5"),"r")
    new_file = h5py.File(os.path.join(data_dir,"data_out_reshaped.hdf5"),"w")
    waypoint_file = os.path.join(data_dir,"place_waypoint.pkl")
    with open(waypoint_file,"rb") as f:
        waypoint_data = pickle.load(f)
    # for each episode
    for scene_ep_data in tqdm(waypoint_data):
        scene_id = scene_ep_data["scene_id"]
        episode_id = scene_ep_data["episode_id"]
        recep = scene_ep_data["recep"]
        if not f"scene_{scene_id}" in new_file:
            new_file.create_group(f"scene_{scene_id}")
        new_file.create_group(f"/scene_{scene_id}/ep_{episode_id}/")
        skill_waypoint_data = scene_ep_data["skill_waypoint_data"]
        # for each recep
        for recep_id,skill_waypoint_singile_recep_data in enumerate(skill_waypoint_data):
            recep_position = skill_waypoint_singile_recep_data["recep_position"]
            scene_ep_recep_grp_reshaped = new_file.create_group(f"/scene_{scene_id}/ep_{episode_id}/{recep_position}")
            each_view_point_data = skill_waypoint_singile_recep_data["each_view_point_data"]
            scene_ep_recep_grp = h5py_file[f"/scene_{scene_id}/ep_{episode_id}/{recep_position}"]
            length = len(each_view_point_data)
            start_rgb_s = np.reshape(scene_ep_recep_grp['start_rgb_s'],(length,-1,scene_ep_recep_grp['start_rgb_s'].shape[-2],scene_ep_recep_grp['start_rgb_s'].shape[-1]))
            start_semantic_s = np.reshape(scene_ep_recep_grp['start_semantic_s'],(length,-1,scene_ep_recep_grp['start_semantic_s'].shape[-1]))
            start_depth_s = np.reshape(scene_ep_recep_grp['start_depth_s'],(length,-1,scene_ep_recep_grp['start_depth_s'].shape[-1]))
            view_point_position_s = np.reshape(scene_ep_recep_grp["view_point_position_s"],(length,-1))
            start_top_down_map_s = np.reshape(scene_ep_recep_grp["start_top_down_map_s"],(length,-1,scene_ep_recep_grp["start_top_down_map_s"].shape[-1]))
            start_obstacle_map_s = np.reshape(scene_ep_recep_grp["start_obstacle_map_s"],(length,-1,scene_ep_recep_grp["start_obstacle_map_s"].shape[-1]))

            scene_ep_recep_grp_reshaped.create_dataset(name="start_rgb_s",data=start_rgb_s)
            scene_ep_recep_grp_reshaped.create_dataset(name="start_semantic_s",data=start_semantic_s)
            scene_ep_recep_grp_reshaped.create_dataset(name="start_depth_s",data=start_depth_s)
            scene_ep_recep_grp_reshaped.create_dataset(name="view_point_position_s",data=view_point_position_s)
            scene_ep_recep_grp_reshaped.create_dataset(name="start_top_down_map_s",data=start_top_down_map_s)
            scene_ep_recep_grp_reshaped.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
    new_file.flush()



    