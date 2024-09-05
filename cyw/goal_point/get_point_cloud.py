"""
根据 depth, mask 计算点云数据
如果 recep mask is None, use all ones matrix as recep mask, witch is the same with place policy
"""
import numpy as np
import torch
import home_robot.utils.depth as du
from home_robot.utils.image import smooth_mask
import argparse
import os
import h5py
from tqdm import tqdm
import pickle
import  cv2
import matplotlib.pyplot as plt


import sys
sys.path.append('/raid/home-robot/projects/habitat_ovmm/utils')
from config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config

show_image = False
save_image = False
IMG_DIR = 'cyw/test_data/place_point'

def get_target_point_cloud_base_coords(
    config,
    device,
    depth,
    target_mask: np.ndarray,
):
    """Get point cloud coordinates in base frame"""
    goal_rec_depth = torch.tensor(
        depth, device=device, dtype=torch.float32
    ).unsqueeze(0)

    camera_matrix = du.get_camera_matrix(
        config.ENVIRONMENT.frame_width,
        config.ENVIRONMENT.frame_height,
        config.ENVIRONMENT.hfov,
    )
    # Get object point cloud in camera coordinates
    pcd_camera_coords = du.get_point_cloud_from_z_t(
        goal_rec_depth, camera_matrix, device, scale=1
    )

    # Agent height comes from the environment config
    agent_height = torch.tensor(1.3188)

    # Object point cloud in base coordinates
    pcd_base_coords = du.transform_camera_view_t(
        pcd_camera_coords, agent_height, -30, device
    ) # pcd_base_coords 坐标每一个元素是 （x,y,z） x轴为右，y轴为前，z轴为上

    non_zero_mask = torch.stack(
        [torch.from_numpy(target_mask).to(device)] * 3, axis=-1
    )
    pcd_base_coords = pcd_base_coords * non_zero_mask

    return pcd_base_coords[0]

def get_recep_mask(semantic,depth,rgb=None):
    '''
        get recep mask
    '''
    goal_rec_mask = (
        semantic
        == 3 * du.valid_depth_mask(depth)
    ).astype(np.uint8)
    # Get dilated, then eroded mask (for cleanliness)
    erosion_kernel= np.ones((5, 5), np.uint8)
    goal_rec_mask = smooth_mask(
        goal_rec_mask, erosion_kernel, num_iterations=5
    )[1]
    # Convert to booleans
    goal_rec_mask = goal_rec_mask.astype(bool)
    # @cyw
    if np.sum(goal_rec_mask*1.0) < 50:
        goal_rec_mask = np.ones_like(goal_rec_mask)

    if show_image:
        cv2.imshow("rgb",rgb)
        cv2.imshow('semantic',(goal_rec_mask*255).astype(np.uint8))
    if save_image:
        print(f"save img to {IMG_DIR} *****************")
        cv2.imwrite(f'{IMG_DIR}/rgb.jpg',rgb)
        cv2.imwrite(f"{IMG_DIR}/recep.jpg",(goal_rec_mask*255).astype(np.uint8))
    return goal_rec_mask

# 点云归一化
def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center)**2).sum(1))**0.5).max()
    return max_radius, center

def WorldSpaceToBallSpace(pointcloud):
    """
    change the raw pointcloud in world space to united vector ball space
    return: max_radius: the max_distance in raw pointcloud to center
            center: [x,y,z] of the raw center
    # @cyw
    input: N * 3
    """
    max_radius, center = FindMaxDis(pointcloud)
    pointcloud_normalized = (pointcloud - center) / max_radius
    return pointcloud_normalized, max_radius, center

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# 可视化点云函数
def vis_pcd(point_cloud_data):
    # 将点云数据拆分为 x、y、z 坐标
    x = point_cloud_data[:, 0]
    y = point_cloud_data[:, 1]
    z = point_cloud_data[:, 2]

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云数据
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置图形属性
    ax.set_xlabel('X 轴')
    ax.set_ylabel('Y 轴')
    ax.set_zlabel('Z 轴')

    if save_image:
        plt.savefig(f"{IMG_DIR}/pcd.jpg")

    if show_image:
        # 显示图形
        plt.show()

class data_prepare:
    '''数据准备,在map_prepare上面封装一层,抽取local map and target
        存储为文件
    '''
    def __init__(self,data_dir,config) -> None:
        self.data_dir = data_dir
        # NOTE
        self.h5py_dataset = h5py.File(os.path.join(data_dir,"data_out.hdf5"),"r+")
        with open(os.path.join(data_dir,"place_waypoint.pkl"),"rb") as f:
            self.pkl_data = pickle.load(f)
        self.config = config
        self.device = "cuda:3"


    def genrate_data(self):
        '''生成数据，并按照一定地格式存储
        scene_ep_recep_grp里面的数据：
            scene_ep_recep_grp.create_dataset(name="start_rgb_s",data=start_rgb_s)
            scene_ep_recep_grp.create_dataset(name="start_semantic_s",data=start_semantic_s)
            scene_ep_recep_grp.create_dataset(name="start_depth_s",data=start_depth_s)
            scene_ep_recep_grp.create_dataset(name="start_top_down_map_s",data=start_top_down_map_s)
            scene_ep_recep_grp.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
            scene_ep_recep_grp.create_dataset(name="view_point_position_s",data=view_point_position_s)
            scene_ep_recep_grp.create_dataset(name='end_recep_map_s',
            data=end_recep_map_s
            )
        '''
        # for each episode
        for scene_ep_data in tqdm(self.pkl_data):
            scene_id = scene_ep_data["scene_id"]
            episode_id = scene_ep_data["episode_id"]
            skill_waypoint_data = scene_ep_data["skill_waypoint_data"]
            # for each recep
            recep_pos_h5py_s = list(self.h5py_dataset[f"/scene_{scene_id}/ep_{episode_id}/"].keys()) #不知道为什么，重新采集数据，两者有点对不上
            for recep_id,skill_waypoint_singile_recep_data in enumerate(skill_waypoint_data):
                recep_position = skill_waypoint_singile_recep_data["recep_position"]
                each_view_point_data = skill_waypoint_singile_recep_data["each_view_point_data"]
                scene_ep_recep_grp = None
                for recep_position_h5py in recep_pos_h5py_s:
                    recep_position_h5py_array = np.fromstring(recep_position_h5py.replace('[', '').replace(']', '').strip().replace('        ',',').replace('    ', ','), sep=',')
                    if len(recep_position_h5py_array) != 3:
                        print("debug")
                    if np.allclose(recep_position,recep_position_h5py_array,atol=0.1): 
                        scene_ep_recep_grp = self.h5py_dataset[f"/scene_{scene_id}/ep_{episode_id}/{recep_position_h5py}"]
                        break
                if scene_ep_recep_grp is None:
                    print("no this data!")
                    continue
                pcd_base_coord_s = []
                if "pcd_base_coord_s" in scene_ep_recep_grp:
                    print("next data *******************")
                    continue
                for i in tqdm(range(len(scene_ep_recep_grp['start_semantic_s']))):
                    semantic = scene_ep_recep_grp['start_semantic_s'][i]
                    depth = scene_ep_recep_grp['start_depth_s'][i]
                    if not show_image:
                        recep_semantic = get_recep_mask(
                            semantic=semantic,
                            depth=depth,
                            )
                    else:
                        recep_semantic = get_recep_mask(
                        semantic=semantic,
                        depth=depth,
                        rgb = scene_ep_recep_grp['start_rgb_s'][i]
                        )
                    pcd_base_coords = get_target_point_cloud_base_coords(
                        self.config,
                        self.device,
                        depth=depth,
                        target_mask=recep_semantic
                    )
                    pcd_base_coords = pcd_base_coords.cpu().numpy() # (640,480,3)
                    # 下采样
                    pcd_base_coords = pcd_base_coords[::4,::4,:]
                    # 归一化  
                    pcd_base_coords = pcd_base_coords.reshape((-1,3))      
                    if show_image or save_image:
                        vis_pcd(pcd_base_coords) 
                    pcd_base_coords, _, _ = WorldSpaceToBallSpace(pcd_base_coords)
                    # 全部点输入爆内存，参考 https://github.com/yanx27/Pointnet_Pointnet2_pytorch
                    # 先对点云进行采样
                    # 
                    pcd_base_coords = farthest_point_sample(pcd_base_coords, 1024)
                    if show_image or save_image:
                        vis_pcd(pcd_base_coords) 
                    pcd_base_coord_s.append(pcd_base_coords)
                pcd_base_coord_s = np.stack(pcd_base_coord_s,axis=0)
                if "pcd_base_coord_s" in scene_ep_recep_grp:
                    scene_ep_recep_grp.__delitem__("pcd_base_coord_s")
                scene_ep_recep_grp.create_dataset(name="pcd_base_coord_s",data=pcd_base_coord_s)  
        self.h5py_dataset.flush()              

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scene_ep_data",
        help="Path to saving scene info",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    # @cyw
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="cyw/configs/debug/agent/heuristic_agent_place.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type="local"
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)
    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    data_prepare_ins = data_prepare(args.data_dir,agent_config)
    data_prepare_ins.genrate_data()
