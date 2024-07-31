# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
专门为了复现place 策略的代码，debug用，之后代码确定后，可删除
'''
import argparse
import os
import pickle
from typing import Optional, Tuple

import h5py
import numpy as np
from evaluator import create_ovmm_env_fn
from habitat.tasks.rearrange.utils import get_robot_spawns
from utils.config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config


from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from habitat.core.simulator import AgentState
import cv2
from home_robot.core.interfaces import DiscreteNavigationAction
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
# from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
# from home_robot.agent.ovmm_agent.ovmm_agent_skill_collect import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.ovmm_agent_skill_collect_refine import OpenVocabManipAgent
# 调试机器人place策略的代码
from home_robot.agent.ovmm_agent.ovmm_agent_pick_place_collect import OpenVocabManipAgent_pick_place
from habitat.utils.visualizations import maps
import json
# from cyw.goal_point.utils import get_relative_position
# cyw/goal_point/data_prepare.py
from cyw.goal_point.data_prepare import visual_obstacle_map,visual_init_obstacle_map
from tqdm import tqdm
from pathlib import Path
import sys

import random

random.seed(1234)
collect_fail_prob = 1 # TODO 当失败时，以collect_fail_prob的概率采集数据 
view_point_num = 10 # 采样 view_point_num 个点来交互

# src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py
show_map_image = False
show_image = False
debug = True

def get_semantic_vis(semantic_map, palette=d3_40_colors_rgb):
    semantic_map_vis = Image.new(
        "P", (semantic_map.shape[1], semantic_map.shape[0])
    )
    semantic_map_vis.putpalette(palette.flatten())
    semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))

    semantic_map_vis = semantic_map_vis.convert("RGB")

    semantic_map_vis = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]

    return semantic_map_vis

# @cyw
def get_agent_state_position(viewpoints_matrix,view_idx) -> AgentState:
    '''
        抽取viewpoint位置
    '''
    view = viewpoints_matrix[view_idx]
    position, rotation, iou = (
        view[:3],
        view[3:7],
        view[7].item(),
    )
    agent_state = AgentState(position, rotation)
    return agent_state

def get_place_success(hab_info):
    '''
        根据hab_info判断是否成功
    '''
    if debug:
        print(f"ovmm_place_success: {hab_info['ovmm_place_success']}")
        print(f"ovmm_placement_stability: {hab_info['ovmm_placement_stability']}")
        print(f"ovmm_place_object_phase_success:{hab_info['ovmm_place_object_phase_success']}")
        print(f"robot_collisions.robot_scene_colls:{hab_info['robot_collisions']['robot_scene_colls']}")
        print(f"ovmm_nav_to_place_succ is {hab_info['ovmm_nav_to_place_succ']}")
        print(f"ovmm_nav_orient_to_place_succ is {hab_info['ovmm_nav_orient_to_place_succ']}")
        # 如果放置成功：ovmm_place_object_phase_success 仍然是false，但 ovmm_place_success 会是 true

    # 参考 projects/habitat_ovmm/utils/metrics_utils.py
    # The task is considered successful if the agent places the object without robot collisions
    # overall_success = (
    #     episode_metrics["END.robot_collisions.robot_scene_colls"] == 0
    # ) * (episode_metrics["END.ovmm_place_success"] == 1)
    place_success = (hab_info['robot_collisions']['robot_scene_colls'] == 0) * (hab_info['ovmm_place_success'])
    nav2place = hab_info['ovmm_nav_to_place_succ'] and hab_info['ovmm_nav_orient_to_place_succ']
    return place_success, nav2place

def convertManualInput(code):
    '''手动控制，将手动输入的代码转为动作
        return: action, info=None
    '''
    ctrl_map = {'a': DiscreteNavigationAction.TURN_LEFT, 'w': DiscreteNavigationAction.MOVE_FORWARD, 'd': DiscreteNavigationAction.TURN_RIGHT,"s":DiscreteNavigationAction.STOP,"n":DiscreteNavigationAction.NAVIGATION_MODE}
    if code in ctrl_map:
        return ctrl_map[code],None
    else:
        return DiscreteNavigationAction.EMPTY_ACTION,None

def extract_scene_id(scene_id: str) -> str:
    """extracts scene id from string containing the scene id"""

    before, _ = scene_id.split(".scene")
    _, after = before.split("uncluttered/")
    return after

# @cyw
# 可视化 top_down_map copy from src/third_party/habitat-lab/examples/shortest_path_follower_example.py
def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

# @cyw
# def get_init_scene_episode_count_dict(
#     env: HabitatOpenVocabManipEnv,
# ) -> Tuple(dict, int):
def get_init_scene_episode_count_dict(
    env: HabitatOpenVocabManipEnv,
) -> Tuple[dict, int]:
    """Returns a dictionary containing entries for all (scene, episode) pairs
    with count value initialized as 0"""
    count_dict = {}
    num_episodes = 0
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        hash_str = f"ep_{episode.episode_id}_scene_{scene_id}"
        count_dict[hash_str] = 0
        num_episodes += 1
    return count_dict, num_episodes

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def receptacle_position_aggregate(data_dir: str, env: HabitatOpenVocabManipEnv):
    """Aggregates receptacles position by scene using all episodes"""

    # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    receptacle_positions = {}
    count_episodes = 0

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        env.reset()
        # episode = env.get_current_episode()
        episode = env._dataset.episodes[count_episodes]
        scene_id = extract_scene_id(episode.scene_id)

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f"ep_{episode.episode_id}_scene_{scene_id}"
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
            count_episodes += 1
        else:
            raise ValueError(
                "count_dict[hash_str] is 0 when hash_str is called for the first time."
            )

        if not scene_id in receptacle_positions:
            receptacle_positions[scene_id] = {}

        if not episode.goal_recep_category in receptacle_positions[scene_id]:
            receptacle_positions[scene_id][episode.goal_recep_category] = []
        for recep in tqdm(episode.candidate_goal_receps):
            recep_position = list(recep.position) # recep数据里面没有朝向
            # 搜索所有waypoint
            view_point_positions = set()
            for view_point in tqdm(recep.view_points):
                view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,view_point).position)
                view_point_positions.add(tuple(view_point_position))
            receptacle_positions[scene_id][episode.goal_recep_category].append(
                {
                    "recep_position": recep_position,
                    "view_point_positions":view_point_positions
                }
            )

        # # @cyw
        print_progress(count_episodes, num_episodes, prefix='count_episodes: %d/%d'%((count_episodes),num_episodes))
        if count_episodes == num_episodes:
            break
        # if count_episodes == 1:
        #     break

    print(f"****************save data to ./{data_dir} ****************")
    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("********save done ***************")

def str2array(str_data):
    # 去除字符串中的方括号和空格，并将剩余的数字转换为浮点数
    data_list = [float(x) for x in str_data.strip('[]').split()]

    # 将列表转换为 NumPy 数组
    data_array = np.array(data_list).astype(np.float32)
    return data_array

def replay_place_policy(
    env: HabitatOpenVocabManipEnv, 
    agent,
    replay_data,
    manual=False,
):
    """Generates images of receptacles by episode for all scenes"""

    # sim = env.habitat_env.env._env._env._sim
    # @cyw
    sim = env.habitat_env.env.habitat_env._sim

    # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    count_episodes = 0
    
    # Ideally, we can make it like an iterator to make it feel more intuitive
    total_data = {}
    data_num = 0
    success_num = 0
    nav_place_success_num = 0
    while True:
        # Get a new episode
        # obs = env.reset()
        observations, done = env.reset(), False #跳转到下一个episode
        episode = env.get_current_episode()
        replay_data_ep = replay_data[episode.episode_id]
        scene_id = extract_scene_id(episode.scene_id)
        agent.reset()
        if debug:
            print(f"*************new episode episode id {episode.episode_id}**************")

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f"ep_{episode.episode_id}_scene_{scene_id}"
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
            count_episodes += 1
        else:
            raise ValueError(
                "count_dict[hash_str] is 0 when hash_str is called for the first time."
            )
        recep = observations.task_observations['place_recep_name']
        recep_vals = replay_data_ep.keys()
        for recep_pos in tqdm(recep_vals):
            print("**************new position ***************")
            view_point_positions = replay_data_ep[recep_pos]
            recep_position = str2array(recep_pos)
            for view_point_position in tqdm(view_point_positions):
                view_point_position = str2array(view_point_position)
                start_position, start_rotation, _ = get_robot_spawns(
                    target_positions=view_point_position[None],
                    rotation_perturbation_noise=0,
                    distance_threshold=0,
                    sim=sim,
                    num_spawn_attempts=100,
                    physics_stability_steps=100,
                    orient_positions=recep_position[None],
                )
                ''' 初始化 '''
                # 拿起任务中要放的东西
                observations = env.pick_up_obj()
                start_observations = env.set_position(start_position,start_rotation)
                
                while not done:
                    if not manual:
                        action, info, _ = agent.act(observations)
                        # sensor_pose: (7,) array denoting global pose (x, y, o) and local map boundaries planning window (gy1, gy2, gx1, gy2)
                    else:
                        manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, s: Stop, u: LookUp, n: LookDown)")
                        action,info = convertManualInput(manual_step)
                    observations, done, hab_info = env.apply_action(action, info)

                
                ''' 执行完毕,获取数据 '''
                # NOTE 需要在env._reset_stats之前，记录位姿信息
                place_success, nav_place = get_place_success(hab_info)
                if debug:
                    print(f"place success is {place_success}")
                
                # # 记录数据
                # total_data[data_key] = place_success
                # # TODO 这样记录对于复现还是很难
                data_num += 1
                success_num += place_success
                nav_place_success_num += nav_place

                agent.reset()
                env._reset_stats() # 重置一些状态，但不跳转到下一个episode
                # 重置状态后，start_position 和star_rotation都会变换，因此，需要重新计算坐标（现在记录绝对坐标，因此不需要重新计算）
                done = False

            # '''运行完一个episode 的一个recep位置，保存数据'''
            # with open(os.path.join(data_dir,"success.json"),"w") as f:
            #     json.dump(total_data,f,indent=2)
            
        print_progress(count_episodes, num_episodes, prefix='count_episodes: %d/%d'%((count_episodes),num_episodes))
        if count_episodes == num_episodes:
            break
        # if count_episodes == 2: # TODO
        #     break
    
    # 统计总的sr
    sr = success_num/data_num
    print(f"success rate is {success_num}/{data_num}  = {sr:0.4f}")
    print(f'place success rate is  {place_success}/{data_num}: {place_success/data_num}')
    # NOTE 一定要放在循环外
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="cyw/temp/heuristic_agent_gaze_place_vs_heuristic_agent_nav_place.pkl",
        help="the replay data file",
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
    parser.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help="if true, use manul control"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "zxy_pick_place"],
        help="Agent to evaluate",
    )
    args = parser.parse_args()

    # 读取replay_data file 并设置episode
    with open(args.datafile,"rb") as f:
        replay_data = pickle.load(f)
    if "habitat.dataset.episode_ids" not in args.overrides:
        episode_ids = list(replay_data.keys())
        args.overrides.append(f"habitat.dataset.episode_ids={episode_ids}")

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)
    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)
    device_id = 1
    # agent = PlaceAgent(agent_config, device_id=device_id)
    if args.agent_type == "baseline":
        agent = OpenVocabManipAgent(agent_config, device_id=device_id)
    elif args.agent_type == "zxy_pick_place":
        agent = OpenVocabManipAgent_pick_place(agent_config, device_id=device_id)


    # Create an env
    env = create_ovmm_env_fn(env_config)

    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # # Generate images of receptacles by episode
    replay_place_policy(env, agent, replay_data)

