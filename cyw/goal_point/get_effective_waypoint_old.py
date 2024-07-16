'''
计算机器人当前视角下的waypoint，采用GPS坐标系以及单位
用来训练模型： 
Input: rgb(optional),semantic,depth
Output: waypoint relative GPS location 

waypoint过滤条件：
1. 在当前视角范围内
2. 在密度最大的区域（怎么得到密度最大的区域？）

hfov 在环境配置里面有说明
'''
import argparse
import os
import pickle
import math
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("/raid/home-robot")
sys.path.append("/raid/home-robot/projects")
from habitat_ovmm.utils.config_utils import get_omega_config


from cyw.goal_point.utils import transform2relative

debug = True
vis_image = False
img_dir = "cyw/test_data/waypoint_data_debug"

def get_hfov_point(hfov,relative_gps_s):
    '''计算 hfov范围内 gps location
        hfov: 机器人水平视场角 degreee
    '''
    hfov_rad = np.radians(hfov)
    valid_gps = []
    for relative_gps in relative_gps_s:
        angle_rad = math.atan2(relative_gps[1],relative_gps[0])
        if abs(angle_rad) <= hfov_rad/2:
            valid_gps.append(relative_gps)
        # if debug:
        #     print(f"angle_rad is {angle_rad}, hfov is {hfov_rad}")
    return valid_gps

# calculate density
def calculate_density(points, bandwidth=1.0):
    """
    计算一组点的密度

    Args:
        points (numpy.ndarray): 二维数组，表示一组点的坐标，每行为一个点的坐标（x, y）
        bandwidth (float): KDE 的带宽参数，控制估计的平滑程度

    Returns:
        numpy.ndarray: 与输入点对应的密度值数组
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)
    density = np.exp(kde.score_samples(points))
    return density

def find_point_with_max_density(points, bandwidth=1.0):
    """
    从一组点中选择密度最大的点

    Args:
        points (numpy.ndarray): 二维数组，表示一组点的坐标，每行为一个点的坐标（x, y）
        bandwidth (float): KDE 的带宽参数，控制估计的平滑程度

    Returns:
        tuple: 密度最大的点的坐标（x, y）和对应的密度值
    """
    density = calculate_density(points, bandwidth)
    max_density_index = np.argmax(density)
    max_density_point = points[max_density_index]
    max_density_value = density[max_density_index]
    return max_density_point, max_density_value

def find_nearest_point(points):
    """
    寻找一系列点中距离原点最近的点（使用街区距离）

    Args:
        points (numpy.ndarray): 二维数组，表示一系列点的坐标，每行为一个点的坐标（x, y）

    Returns:
        numpy.ndarray: 距离原点最近的点的坐标（x, y）
    """
    distances = np.sum(np.abs(points), axis=1)  # 计算每个点与原点的街区距离
    min_index = np.argmin(distances)  # 找到距离最小的点的索引
    nearest_point = points[min_index]  # 获取距离最小的点的坐标
    return nearest_point

'''get waypoint data'''
def select_waypoint(hfov,data_dir):
    '''选择waypoint，条件见最顶端

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
    pkl_data = os.path.join(data_dir,'place_waypoint.pkl')
    with open(pkl_data,"rb") as f:
        pkl_data = pickle.load(f)
    '''
    NOTE: 需要记录 scene_id episode_id recep_position viewpoint_index才能找到 对应的 rgb semantic depth。为了能用assignment_idx读取数据，使用 和 waypoint_data 同样的组织格式
    [
        scene_ep_data{
            scene_id:
            episode_id:
            recep_data: [
                {
                    recep_loc:
                    waypoint_data: [
                        {
                            waypoint_loc:
                            target_loc:
                            success:
                        }
                    ]
                }
            ]
        }
    ]
    '''
    total_data = []
    for scene_ep_data in tqdm(pkl_data):
        scene_ep_data_process = {}
        scene_id = scene_ep_data["scene_id"]
        episode_id = scene_ep_data["episode_id"]
        scene_ep_data_process["scene_id"] = scene_id
        scene_ep_data_process["episode_id"] = episode_id
        scene_ep_data_process["recep_data"] = []
        for recep_data in scene_ep_data["skill_waypoint_data"]:
            recep_data_process = {}
            recep_position = recep_data["recep_position"]
            recep_data_process["recep_position"] = recep_position
            recep_data_process["waypoint_data"] = []
            each_view_point_data = recep_data["each_view_point_data"]
            # gather all success view_point and transfor to relative recep
            recep_relative_pos, viewpoint_relative_pos = transform2relative(
                data=each_view_point_data,
                keep_success=True,
                recep_position=recep_position,
            )
            for waypoint_index, recep_pos in enumerate(recep_relative_pos):
                success = each_view_point_data[waypoint_index]["place_success"]
                if success:
                    recep_data_process["waypoint_data"].append(
                        {
                            "target_point":np.array([0.,0.]),
                            "view_point_position": each_view_point_data[waypoint_index]["view_point_position"], #验证用
                            "success":success,
                            "another_recep":0
                        }
                    )
                    continue
                if len(viewpoint_relative_pos[waypoint_index])!=0:
                    # get the location with hfov
                    valid_gps = get_hfov_point(hfov=hfov,relative_gps_s=viewpoint_relative_pos[waypoint_index])
                    if len(valid_gps)==0:
                        print("valid gps is None, select min block distace point")
                        target_point = find_nearest_point(np.stack(viewpoint_relative_pos[waypoint_index],axis=0))
                    elif len(valid_gps)==1:
                        target_point = valid_gps[0]
                    else:
                        # select one point
                        target_point, max_density_value = find_point_with_max_density(points=valid_gps,bandwidth=1)
                    recep_data_process["waypoint_data"].append(
                        {
                            "target_point":target_point,
                            "view_point_position": each_view_point_data[waypoint_index]["view_point_position"], #验证用
                            "success":success,
                            "another_recep":0
                        }
                    )
                    # visualize
                    if vis_image:
                        x_recep = recep_pos[0]
                        y_recep = recep_pos[1]
                        waypoints = np.stack(viewpoint_relative_pos[waypoint_index],axis=0)
                        x_waypoints = waypoints[:,0]
                        y_waypoints = waypoints[:,1]
                        x_selected = target_point[0]
                        y_selected = target_point[1]
                        plt.figure()
                        plt.scatter(x_recep,y_recep,color="red",label="recep")
                        plt.scatter(x_waypoints,y_waypoints,color="blue",label="waypoint")
                        if len(valid_gps)!=0:
                            valid_points = np.stack(valid_gps)
                            x_valid = valid_points[:,0]
                            y_valid = valid_points[:,1]
                            plt.scatter(x_valid,y_valid,color="green",label="valid")
                        plt.scatter(x_selected,y_selected,color="black",label="target")
                        plt.legend()
                        # plt.show()
                        os.makedirs(f"{img_dir}/{scene_id}/{episode_id}",exist_ok=True)
                        print(f"save image to {img_dir}***************")
                        plt.savefig(f"{img_dir}/{scene_id}/{episode_id}/{waypoint_index}.jpg")
                else:
                    # 没有任何一个位置可以放置成功
                    recep_data_process["waypoint_data"].append(
                        {
                            "target_point":None,
                            "view_point_position": each_view_point_data[waypoint_index]["view_point_position"], #验证用
                            "success":success,
                            "another_recep":1
                        }
                    )
            scene_ep_data_process["recep_data"].append(recep_data_process)
        total_data.append(scene_ep_data_process)
    # save data
    save_file = os.path.join(data_dir,"target_waypoint.pkl")
    print(f"save data to {os.path.join(data_dir,'target_waypoint.pkl')}********")
    with open(save_file,"wb") as f:
        pickle.dump(total_data,f)
    print("save done**************")
        
if __name__ == "__main__":
    # run parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config",type=str,default="cyw/configs/env/hssd_eval.yaml")
    parser.add_argument("--data_dir",type=str,default="cyw/datasets/place_dataset/train/heuristic_agent_place")
    args = parser.parse_args()
    # read env config
    env_config = get_omega_config(args.env_config)
    hfov = env_config.ENVIRONMENT.hfov # degree
    select_waypoint(hfov,args.data_dir)
    print("over")