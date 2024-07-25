'''
规则式的gaze:
当机器人结束nav_to_recep技能后：
1. 提取局部地图及recep图
2. 喂给模型，输出goal点
3. goal点转为地图中的点
**以上过程为选择goal点，为了对比，或许也可以加上gt数据对比？***
4. 导航到该位置（如果goal点不可到达怎么办？）
5. 朝向容器(how to do)

实现过程：参考 src/home_robot/home_robot/manipulation/heuristic_place_policy.py
在第一步的时候记录 goal点，并设置为类参数，之后的步数调用nav_planner算法即可

另：或许需要一个单独的可以测试place性能的代码
NOTE: 这个过程没有对地图更新,是否会出现碰撞或者卡住的问题?
'''
import torch
from loguru import logger
import numpy as np

from home_robot.navigation_planner.discrete_planner import DiscretePlanner
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)
from .unet import UNet
from cyw.goal_point.utils import map_prepare

class gaze_rec_policy:
    '''
        让机器人走到一个合适的抓取或放置位姿 (目前只实现place)
    '''
    def __init__(
        self,
        config,
        ckp_path:str,
        min_goal_distance_cm: float = 50.0, #TODO copy from src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent.py, may should be changed
        continuous_angle_tolerance: float = 30.0,
        debug_vis:bool=True,
        verbose:bool=True,
    ) -> None:
        self.map_prepare = map_prepare(config.ENVIRONMENT,config.AGENT)
        # 加载模型
        # 或许还需要一些修改,参考 https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        self.unet = UNet(n_channels=2,n_classes=1,bilinear=False)
        ckpt = torch.load(ckp_path)
        self.unet.load_state_dict(ckpt["state_dict"])
        self.unet.eval()
        # path planner
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=min_goal_distance_cm,
            continuous_angle_tolerance=continuous_angle_tolerance,
        )
        self.debug_vis = debug_vis
        self.verbose = verbose
        
    def reset(self):
        self.time_step = 0

    def act(self,obs: Observations):
        '''将机器人移到适合交互的位置,并朝向recep
            Returns:
                action: what the robot will do - a hybrid action, discrete or continuous
        '''
        '''获得障碍物层和recep层'''
