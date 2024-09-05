'''
/raid/home-robot/src/home_robot/home_robot/navigation_policy/gaze/gaze_policy.py
的备份

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
import sys
import os
home_root = os.environ.get('HOME_ROBOT_ROOT')
sys.path.append(home_root)
sys.path.append(f"{home_root}/projects")
from cyw.goal_point.utils import map_prepare
from cyw.goal_point.visualize import vis_local_map,visual_init_obstacle_map_norotation
from habitat_ovmm.utils.config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config
import pickle

THRESHOLD = 0.5

show_image = False
save_image = True
img_dir = "cyw/test_data/place_policy/goal_map_yolo_detic"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

class gaze_rec_policy:
    '''
        让机器人走到一个合适的抓取或放置位姿 (目前只实现place)
    '''
    def __init__(
        self,
        config,
        debug_vis:bool=False,
        verbose:bool=True,
    ) -> None:
        self.map_prepare = map_prepare(config)
        # 加载模型 参考 https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        self.unet = UNet(n_channels=2,n_classes=1,bilinear=False)
        ckp_path = config.AGENT.SKILLS.GAZE_OBJ.checkpoint_path
        ckpt = torch.load(ckp_path)
        model_weights = ckpt["state_dict"]
        # update keys by dropping `auto_encoder.`
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)
        self.unet.load_state_dict(model_weights)
        self.unet.eval()
        self.debug_vis = debug_vis
        self.verbose = verbose
        self.time_step = 0

    def reset(self):
        # self.time_step = 0
        pass

    def act(self,
        info
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
        local_goal_map = self.unet(input_map)
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
        if self.debug_vis and (show_image or save_image):
            print("*********calculate image************")
            local_map_vis = vis_local_map(
                local_map=loacal_obstacle.cpu().numpy(),
                recep_map = loacal_recep_map.cpu().numpy(),
                goal_map = local_goal_map
            )
            # obstacle_map_vis = visual_init_obstacle_map_norotation(obstacle_map,sensor_pose)
            # obstacle_map_vis[goal_map>0] = 128
            if show_image:
                cv2.imshow("local_obstacle_map",local_map_vis)
                # cv2.imshow("obstacle_map",obstacle_map_vis)
            if save_image:
                cv2.imwrite(f"{img_dir}/local_map_{self.time_step}.jpg",local_map_vis)
                # cv2.imwrite(f"{img_dir}/obstcal_map_{self.time_step}.jpg",obstacle_map_vis)
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
        default="cyw/configs/debug_config/agent/heuristic_agent_nav_place.yaml",
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
            )
    with open("cyw/test_data/place_policy/info.pkl","rb") as f:
        info = pickle.load(f)
    gaze_policy.act(info)