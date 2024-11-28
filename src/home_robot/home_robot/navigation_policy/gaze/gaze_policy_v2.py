'''
规则式的gaze:
当机器人结束nav_to_recep技能后：
1. 提取局部地图及recep图
2. 喂给模型，输出goal点
3. goal点转为地图中的点,并转为goal_map
**以上过程为选择goal点，为了对比，或许也可以加上gt数据对比？***
4. 导航到该位置（如果goal点不可到达怎么办？）
5. 朝向容器(how to do) (这一部分有点难以实现,原本代码也没有实现)

实现过程：参考 src/home_robot/home_robot/manipulation/heuristic_place_policy.py
在第一步的时候记录 goal点，并设置为类参数，之后的步数调用nav_planner算法即可

updata: 使用新的模型，有point cloud data 作为输入
但是point cloud 的数据有些问题：用mask来选择应该关注的point，有些情况下没有标注出mask
并且点云数据的归一化存在问题
'''
import torch
from loguru import logger
import numpy as np
import argparse
import cv2
import os

from home_robot.navigation_planner.discrete_planner import DiscretePlanner
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)
# from .unet import UNet # TODO 调试完毕改为简洁模式
from home_robot.navigation_policy.gaze.unet import UNet
from home_robot.navigation_policy.gaze.pointnet_unet.pointunet_model_v1 import PointUNet

import sys
import os
home_root = os.environ.get('HOME_ROBOT_ROOT')
sys.path.append(home_root)
sys.path.append(f"{home_root}/projects")
from home_robot.misc.goal_point.utils import map_prepare,get_recep_mask, WorldSpaceToBallSpace,get_target_point_cloud_base_coords, farthest_point_sample
from home_robot.misc.goal_point.visualize import vis_local_map,visual_init_obstacle_map_norotation
from habitat_ovmm.utils.config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config
import pickle
import matplotlib.pyplot as plt
from typing import Optional

THRESHOLD = 0.5

show_image = False
save_image = False
img_dir = "out_put_image/goal_map_yolo_detic"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)


Gainsboro = [220,220,220]
SteelBlue1 = [99, 184, 255]
LightPink = [255, 182, 193]
Snow1 =	[255, 250, 250]

def vis_local_map(local_map:np.array,recep_map:Optional[np.array]):
    '''
        可视化 local map
        如果输入 recep_map的话，将rece 标记在 local map上
    '''
    local_map_vis = np.zeros((local_map.shape[0],local_map.shape[1],3),dtype=np.uint8)
    local_map_vis[local_map<=0.5] = Snow1
    local_map_vis[local_map>0.5] = Gainsboro
    if recep_map is not None:
        local_map_vis[recep_map == 1] = SteelBlue1
    return local_map_vis

def vis_target(target:np.array,recep_map_vis:Optional[np.array]):
    if recep_map_vis is not None:
        target_map_vis = recep_map_vis.copy()
    else:
        target_map_vis = np.zeros((target.shape[0],target.shape[1],3),dtype=np.uint8)
    target_map_vis[target>0.2] = LightPink
    return target_map_vis

# 可视化点云函数
def vis_pcd(point_cloud_data,ax):
    # 将点云数据拆分为 x、y、z 坐标
    x = point_cloud_data[:, 0]
    y = point_cloud_data[:, 1]
    z = point_cloud_data[:, 2]
    # c = point_cloud_data[:, 3]

    # 绘制点云数据
    ax.scatter(x, y, z, cmap='viridis', marker='o')

def show_results(img_ds, preds, num_rows=1, show_rgb=True,save_dir="batch_show.jpg" ):
    """
        从左到右分别是：grid_map, pcd,  pred, rgb
    """
    if show_rgb:
        num_cols=4
    else:
        num_cols=3
    fig = plt.figure(figsize=(16,5))    
    grid_map = img_ds['map'][0]
    pcd = img_ds['pcd_coords'][0]
    pred = preds
    # pred = torch.sigmoid(pred)
        
    ax = fig.add_subplot(num_rows, num_cols, 1, xticks=[], yticks=[])
    if isinstance(grid_map, torch.Tensor):
        # grid_map = (grid_map[0,:,:].unsqueeze(2)>=0.2)*torch.tensor(SteelBlue1)+(grid_map[1,:,:].unsqueeze(2)>=0.2)*torch.tensor(LightPink)
        grid_map = grid_map.detach().cpu().numpy()
    local_map_vis = vis_local_map(grid_map[0,:,:],grid_map[1,:,:])
    ax.imshow(local_map_vis)
    title = f"grid_map"
    ax.set_title(title) 

    ax = fig.add_subplot(num_rows, num_cols, 2, xticks=[], yticks=[], zticks=[], projection='3d')
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    if isinstance(pcd, np.ndarray):
        vis_pcd(pcd,ax)
    title = f"pcd"
    ax.set_title(title) 

    ax = fig.add_subplot(num_rows, num_cols, 3, xticks=[], yticks=[])
    if isinstance(pred, torch.Tensor):
        # target = target.unsqueeze(2)*torch.tensor(Gainsboro)
        pred = pred.detach().cpu().numpy()
    if isinstance(pred, np.ndarray):
        pred_vis = vis_target(pred,local_map_vis)
        ax.imshow(pred_vis)          
    title = f"pred"
    ax.set_title(title) 
    
    if show_rgb:
        rgb = img_ds['rgb']
        ax = fig.add_subplot(num_rows, num_cols, 4, xticks=[], yticks=[])
        if isinstance(rgb, torch.Tensor):
            # target = target.unsqueeze(2)*torch.tensor(Gainsboro)
            rgb = rgb.detach().cpu().numpy()
        if isinstance(rgb, np.ndarray):
            ax.imshow(rgb)          
        title = f"rgb"
        ax.set_title(title) 
    fig.tight_layout()
    fig.savefig(save_dir)
    # return fig


class gaze_rec_policy:
    '''
        让机器人走到一个合适的抓取或放置位姿 (目前只实现place)
    '''
    def __init__(
        self,
        config,
        device,
        model_type:str="PointUNet", # choose from PointUNet and UNet
        verbose:bool=True,
        pointnet_ckp:str="model/pointnet_v1/PointUnet-81-0.676.ckpt"
    ) -> None:
        self.map_prepare = map_prepare(config)
        # 加载模型 参考 https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        if model_type == "PointUNet":
            self.model = PointUNet(ckp_path=pointnet_ckp,map_encoder_trainable=True)
        elif model_type == "UNet":
            self.model = UNet(n_channels=2,n_classes=1,bilinear=False)
        ckp_path = config.AGENT.SKILLS.GAZE_OBJ.checkpoint_path
        ckpt = torch.load(ckp_path)
        model_weights = ckpt["state_dict"]
        # update keys by dropping `auto_encoder.`
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)
        self.model.load_state_dict(model_weights)
        self.model.eval()
        self.model.to(device)
        self.model_type = model_type
        self.config = config
        self.device = device
        self.verbose = verbose
        self.time_step = 0

    def reset(self):
        # self.time_step = 0
        pass

    def get_pcd(self,obs: Observations):
        '''
            从 obs 中获得点云，归一化
        '''
        semantic = obs.semantic
        depth = obs.depth
        recep_semantic = get_recep_mask(
            semantic=semantic,
            depth=depth,
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
        pcd_base_coords, _, _ = WorldSpaceToBallSpace(pcd_base_coords)
        # 全部点输入爆内存，参考 https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        pcd_base_coords = farthest_point_sample(pcd_base_coords, 1024)
        return pcd_base_coords

    def act(self,
        info,
        obs:Observations,
    ):
        '''将机器人移到适合交互的位置,并朝向recep
            Returns:
                action: what the robot will do - a hybrid action, discrete or continuous
        '''
        obstacle_map=info["obstacle_map"]
        sensor_pose = info["sensor_pose"]
        recep_map = info['end_recep']
        rotated_obstacle_map,map_agent_pos = self.map_prepare.rotate_obstacle_map(
            obstacle_map=obstacle_map,
            sensor_pose=sensor_pose
        )
        loacal_obstacle = self.map_prepare.get_local_map(
            global_map=rotated_obstacle_map,
            map_agent_pos=map_agent_pos
        )
        rotated_end_recep_map,map_agent_pos = self.map_prepare.rotate_obstacle_map(
            obstacle_map=recep_map,
            sensor_pose=sensor_pose
        )
        loacal_recep_map = self.map_prepare.get_local_map(
            global_map=rotated_end_recep_map,
            map_agent_pos=map_agent_pos
        )
        loacal_obstacle = torch.from_numpy(loacal_obstacle)
        loacal_recep_map = torch.from_numpy(loacal_recep_map)
        input_map = torch.stack((loacal_obstacle,loacal_recep_map),axis=0)
        input_map = input_map.squeeze(dim=1)
        input_map = input_map.unsqueeze(dim=0)
        input_map = input_map.to(torch.device(self.device))
        if self.model_type == "PointUNet":
            # 输入数据要求
            # 'map': map_data,
            # 'pcd_coords': data['pcd_base_coord_s']
            pcd_coords = self.get_pcd(obs)
            pcd_coords = torch.from_numpy(pcd_coords)
            pcd_coords = pcd_coords.unsqueeze(dim=0)
            pcd_coords = pcd_coords.to(torch.device(self.device))
            input = {
                'map': input_map,
                'pcd_coords': pcd_coords,
                'rgb':obs.rgb,
            }
            local_goal_map = self.model(input)
        elif self.model_type == "UNet":
            local_goal_map = self.model(input_map)

        local_goal_map = torch.sigmoid(local_goal_map)
        local_goal_map = (local_goal_map > THRESHOLD).float() # 1,1,w, h
        if local_goal_map.sum()==0:
            return None #寻找下一个recep
        # 转换到local map上
        local_goal_map = local_goal_map.cpu().numpy()[0][0]
        goal_map = self.map_prepare.inverse_rotate_map(
            local_goal_map,
            sensor_pose,
            obstacle_map_shape=obstacle_map.shape
        )
        # goal_map = np.expand_dims(goal_map,axis=0) # 1, w, h
         # visualize
        if save_image:
            print("*********calculate image************")
            save_dir = f"{img_dir}/local_map_{self.time_step}.jpg"
            show_results(
                img_ds=input,
                preds=local_goal_map,
                show_rgb=True,
                save_dir=save_dir
            )
            print("*********** save img done **************")
            self.time_step += 1
        return goal_map
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        # default="cyw/configs/debug_config/agent/heuristic_agent_nav_place.yaml",
        default="cyw/configs/agent/heuristic_agent_esc_yolo_nav_gaze_place_cyw.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()
    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)
    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path
    )
    # get env config
    env_config = get_omega_config(args.env_config_path)
    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type='local'
    )
    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    gaze_policy = gaze_rec_policy(
                config=agent_config,
                device="cuda:1"
            )
    # with open("cyw/test_data/place_policy/info.pkl","rb") as f:
    #     info = pickle.load(f)
    with open("cyw/test_data/place_point/info.pkl","rb") as f:
        info = pickle.load(f)
    with open("cyw/test_data/place_point/obs.pkl","rb") as f:
        obs = pickle.load(f)
    
    gaze_policy.act(info,obs)