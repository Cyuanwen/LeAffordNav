# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
pick up object 数据搜集代码

copy from projects/habitat_ovmm/place_data_collection.py
把原本代码的 环境的top_down_map去掉了
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
from cyw.goal_point.visualize import visual_obstacle_map,visual_init_obstacle_map
from tqdm import tqdm
from pathlib import Path
import sys
import time

import random

random.seed(1234)
collect_fail_prob = 1 # TODO 当失败时，以collect_fail_prob的概率采集数据 

show_map_image = False
show_image = False
debug = False
get_timing = False

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
    find_object = hab_info['ovmm_find_object_phase_success']
    pick_success = hab_info['ovmm_pick_object_phase_success']
    return pick_success,find_object

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

    obj_positions = {}
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

        if not scene_id in obj_positions:
            obj_positions[scene_id] = {}

        if not episode.object_category in obj_positions[scene_id]:
            obj_positions[scene_id][episode.object_category] = []
        for object in tqdm(episode.candidate_objects):
            obj_position = list(object.position) # recep数据里面没有朝向
            # 搜索所有waypoint
            view_point_positions = set()
            for view_point in tqdm(object.view_points):
                view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,view_point).position)
                view_point_positions.add(tuple(view_point_position))
            obj_positions[scene_id][episode.object_category].append(
                {
                    "obj_position": obj_position,
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
    with open(f"./{data_dir}/obj_position.pickle", "wb") as handle:
        pickle.dump(obj_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("********save done ***************")

def gen_pick_data(
    data_dir: str, 
    dataset_file: h5py.File,
    env: HabitatOpenVocabManipEnv, agent,
    thread_num:int,
    manual=False,
    baseline_name:Optional[str]=None,
    append:bool=True, #TODO 是否在原本数据上追加数据
    obj_pos_file:str='recep_position_cluster.pickle',
):
    """Generates images of receptacles by episode for all scenes"""
    if append:
        print("***************append the data *************")

    # sim = env.habitat_env.env._env._env._sim
    # @cyw
    sim = env.habitat_env.env.habitat_env._sim

    # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    # Also, creating folders for storing dataset
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        if f"scene_{scene_id}" not in dataset_file:
            dataset_file.create_group(f"scene_{scene_id}")

    obj_pos_dir = str(Path(data_dir).resolve().parent.parent)
    with open(f"{obj_pos_dir}/{obj_pos_file}", "rb") as handle:
        obj_positions = pickle.load(handle)

    count_episodes = 0
    
    # Ideally, we can make it like an iterator to make it feel more intuitive
    if not append:
        total_data = []
    else:
        with open(os.path.join(data_dir,f"place_waypoint_{thread_num}.pkl"),"rb") as f:
            total_data = pickle.load(f)
    while True:
        # Get a new episode
        # obs = env.reset()
        observations, done = env.reset(), False #跳转到下一个episode
        episode = env.get_current_episode()
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
        if f"/scene_{scene_id}/ep_{episode.episode_id}" not in dataset_file:
            dataset_file.create_group(
                f"/scene_{scene_id}/ep_{episode.episode_id}"
            )

        # for recep in receptacle_positions[scene_id]:
        end_recep = observations.task_observations['place_recep_name']
        start_recep = observations.task_observations['start_recep_name']
        object_name = observations.task_observations["object_name"]
        scene_ep_data = {
            "scene_id": scene_id,
            "episode_id":episode.episode_id,
            "end_recep": end_recep,
            "start_recep":start_recep,
            "object_name": object_name,
            "skill_waypoint_data": []
        }
        obj_vals = obj_positions[scene_id][object_name]
        for pos_pair in tqdm(obj_vals):
            print("**************new position ***************")
            obj_position = np.array(pos_pair["obj_position"])
            if f"/scene_{scene_id}/ep_{episode.episode_id}/{obj_position}" not in dataset_file:
                scene_ep_recep_grp = dataset_file.create_group(f"/scene_{scene_id}/ep_{episode.episode_id}/{obj_position}")
            else:
                scene_ep_recep_grp = dataset_file[f"/scene_{scene_id}/ep_{episode.episode_id}/{obj_position}"]
            # 如果已经采集了相关数据，则不再采集
            if len(scene_ep_recep_grp) != 0:
                print(f"the data for /scene_{scene_id}/ep_{episode.episode_id}/{obj_position} has done, continue***************")
                continue
            view_point_positions = pos_pair["view_point_positions"]
            skill_waypoint_singile_recep_data = {
                "obj_position": obj_position,
                "each_view_point_data":[]
            }

            start_rgb_s = []
            start_semantic_s = []
            start_depth_s = []
            start_obstacle_map_s = []
            view_point_position_s = [] # 为了验证，在h5py文件里面也加上 
            start_recep_map_s = []
            obj_map_s = []

            # for view_point_position in tqdm(view_point_positions):
            # 采样数据
            view_point_positions_list = list(view_point_positions)
            # if len(view_point_positions_list) > view_point_num:
            #     view_point_positions_list = random.sample(view_point_positions_list,view_point_num)
            # for view_point_position in tqdm(view_point_positions_list[:1]): #TODO
            for view_point_position in tqdm(view_point_positions_list):
                view_point_position = np.array(view_point_position).astype(np.float32)
                start_position, start_rotation, _ = get_robot_spawns(
                    target_positions=view_point_position[None],
                    rotation_perturbation_noise=0,
                    distance_threshold=0,
                    sim=sim,
                    num_spawn_attempts=100,
                    physics_stability_steps=100,
                    orient_positions=obj_position[None],
                )
                ''' 初始化 '''
                start_observations = env.set_position(start_position,start_rotation)
                start_rotation = env.get_current_rotation()
                start_position = np.array(env.get_current_position()).astype(np.float32)
                
                '''计算物体gps'''
                relative_obj_gps = env.get_relative_gps(obj_position)
                 

                '''执行pick up 动作 '''
                map_id = 0
                while not done:
                    if not manual:
                        if get_timing:
                            t0 = time.time()
                        action, info, obs_postprocess = agent.act(observations)
                        # sensor_pose: (7,) array denoting global pose (x, y, o) and local map boundaries planning window (gy1, gy2, gx1, gy2)
                        if get_timing:
                            t1 = time.time()
                            print(f"[Agent] act time: {t1 - t0:.2f}")
                    else:
                        manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, s: Stop, u: LookUp, n: LookDown)")
                        action,info = convertManualInput(manual_step)
                    observations, done, hab_info = env.apply_action(action, info)
                    if get_timing:
                        t2 = time.time()
                        print(f"[Env] act time: {t2 - t1:.2f}")
                    if debug:
                        print(f"action is {action}")

                    '''如果look around done, 收集 obstacle map and sensor pos'''
                    if "look_around_done" in info and info["look_around_done"]:
                        start_obstacle_map=info["obstacle_map"]
                        start_sensor_pose = info["sensor_pose"]
                        start_recep_map = info['start_recep']
                        object_map = info['object']

                        '''获取semantic'''
                        start_semantic = obs_postprocess.semantic
                        # 是否识别出物体
                        recegnize_obj = len(np.where(start_semantic==1)[0])>0

                        if show_map_image:
                            init_obstacle_map_vis= visual_init_obstacle_map(
                                obstacle_map=info['obstacle_map'],
                                sensor_pose=info['sensor_pose']
                            )
                            cv2.imshow("init_obstacle_map_vis",init_obstacle_map_vis)
                            obstacle_map_vis = visual_obstacle_map(
                                obstacle_map=np.flipud(info['obstacle_map']),
                                sensor_pose=info['sensor_pose']
                            )
                            cv2.imshow("obstacle_map",obstacle_map_vis)
                            start_recep_vis = np.copy(start_recep_map)*255
                            cv2.imshow("start_recep_vis",start_recep_vis)
                            object_vis = np.copy(object_map)*255
                            cv2.imshow("object_vis",object_vis)
                            cv2.waitKey(1)
                            cv2.imwrite(f"cyw/test_data/init_obstacle_map/init_obstacle_map_{map_id}.jpg",init_obstacle_map_vis)
                            cv2.imwrite(f"cyw/test_data/obstacle_map/obstacle_map_{map_id}.jpg",obstacle_map_vis)
                            cv2.imwrite(f"cyw/test_data/start_recep/start_recep_{map_id}.jpg",start_recep_vis)
                            cv2.imwrite(f"cyw/test_data/object_vis/object_vis_{map_id}.jpg",object_vis)
                            
                    if debug and hab_info['robot_collisions']['robot_scene_colls']!=0:
                        print(f"robot_collisions.robot_scene_colls:{hab_info['robot_collisions']['robot_scene_colls']}")

                    # if show_image:
                    #     print("show rgb ......")
                    #     cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                    #     cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB))
                    #     semantic_img = get_semantic_vis(observations.semantic)
                    #     # 非gt下，semantic为None?
                    #     cv2.imshow("semantic",semantic_img)
                    #     cv2.waitKey(1)
                
                ''' 执行完毕,获取数据 '''
                # NOTE 需要在env._reset_stats之前，记录位姿信息
                end_position = np.array(env.get_current_position()).astype(np.float32)
                pick_success, find_object = get_place_success(hab_info)
                if debug:
                    print(f"pick up success is {pick_success}")
                
                # 记录数据
                if pick_success:
                    record = True
                else:
                    record = random.random()<collect_fail_prob
                if record:
                    print(f"record data ********************")
                    start_rgb_s.append(start_observations.rgb)
                    start_depth_s.append(start_observations.depth)
                    if start_observations.semantic is None:
                        start_semantic_s.append(start_semantic)
                    else:
                        start_semantic_s.append(start_observations.semantic)
                    start_obstacle_map_s.append(start_obstacle_map)
                    view_point_position_s.append(view_point_position)
                    start_recep_map_s.append(start_recep_map)
                    obj_map_s.append(object_map)
                    skill_waypoint_singile_recep_data["each_view_point_data"].append(
                        {
                            "view_point_position":view_point_position,
                            "start_position":start_position,
                            "start_rotation":start_rotation,
                            "relative_obj_gps": relative_obj_gps,
                            "end_position": end_position,
                            "pick_success": pick_success,
                            "find_object": find_object,
                            "recegnize_obj":recegnize_obj,
                            "start_sensor_pose": start_sensor_pose, # 在obstacle map里面的位置
                        }
                    )

                agent.reset()
                env._reset_stats() # 重置一些状态，但不跳转到下一个episode
                # 重置状态后，start_position 和star_rotation都会变换，因此，需要重新计算坐标（现在记录绝对坐标，因此不需要重新计算）
                done = False

            '''运行完一个episode 的一个recep位置，保存数据'''
            scene_ep_data["skill_waypoint_data"].append(skill_waypoint_singile_recep_data)

            start_rgb_s = np.stack(start_rgb_s,axis=0)
            start_semantic_s = np.stack(start_semantic_s,axis=0)
            start_depth_s = np.stack(start_depth_s,axis=0)
            start_obstacle_map_s = np.stack(start_obstacle_map_s,axis=0)
            view_point_position_s = np.stack(view_point_position_s,axis=0)
            start_recep_map_s = np.stack(start_recep_map_s,axis=0)
            obj_map_s = np.stack(obj_map_s,axis=0)

            scene_ep_recep_grp.create_dataset(name="start_rgb_s",data=start_rgb_s)
            scene_ep_recep_grp.create_dataset(name="start_semantic_s",data=start_semantic_s)
            scene_ep_recep_grp.create_dataset(name="start_depth_s",data=start_depth_s)
            scene_ep_recep_grp.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
            scene_ep_recep_grp.create_dataset(name="view_point_position_s",data=view_point_position_s)
            scene_ep_recep_grp.create_dataset(name='start_recep_map_s',
            data=start_recep_map_s
            )
            scene_ep_recep_grp.create_dataset(name='obj_map_s',
            data=obj_map_s
            )
            
        # 运行完一个episode,保存数据
        if not len(scene_ep_data["skill_waypoint_data"])==0:
            total_data.append(scene_ep_data)
        with open(os.path.join(data_dir,f"pick_waypoint_{thread_num}.pkl"),"wb") as f:
            pickle.dump(total_data,f)

        dataset_file.flush()
        print_progress(count_episodes, num_episodes, prefix='count_episodes: %d/%d'%((count_episodes),num_episodes))
        if count_episodes == num_episodes:
            break
        # if count_episodes == 2: # TODO
        #     break
    
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
        "--data_dir",
        type=str,
        default="scene_ep_data",
        help="Path to saving scene info",
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="data_out",
        help="Path to saving data",
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
        "--keep_nonrepeat_episode", # 使用不重复的episode，能大大减小数据收集的时间
        action="store_true",
        default=True,
        help="whether to use non repeat id"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "zxy_pick_place"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="whether to append the data in the exists file"
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        help="which thread to run"
    )
    args = parser.parse_args()

    thread_num = args.thread_num
    if args.keep_nonrepeat_episode:
        with open(os.path.join(args.data_dir,'split_episode',f"episode_ids_{thread_num}.json"),"r") as f:
            episode_ids = json.load(f)
        args.overrides.append(f"habitat.dataset.episode_ids={episode_ids}")
    # # TODO
    # if args.keep_nonrepeat_episode:
    #     with open(os.path.join(args.data_dir,f"episode_ids.json"),"r") as f:
    #         episode_ids = json.load(f)
    #     args.overrides.append(f"habitat.dataset.episode_ids={episode_ids}")

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
    device_id = 0
    # agent = PlaceAgent(agent_config, device_id=device_id)
    if args.agent_type == "baseline":
        agent = OpenVocabManipAgent(agent_config, device_id=device_id)
    elif args.agent_type == "zxy_pick_place":
        agent = OpenVocabManipAgent_pick_place(agent_config, device_id=device_id)

    baseline_name = args.baseline_config_path.split("/")[-1].split(".")[0]
    data_dir = os.path.join(args.data_dir,baseline_name,'multi_thread')
    os.makedirs(f"./{data_dir}", exist_ok=True)
    # Create h5py files
    if args.append and os.path.exists(os.path.join(data_dir,f"place_waypoint_{thread_num}.pkl")):
        dataset_file = h5py.File(f"./{data_dir}/{args.datafile}_{thread_num}.hdf5", "r+") 
        append = True
    else:
        dataset_file = h5py.File(f"./{data_dir}/{args.datafile}_{thread_num}.hdf5", "w")
        append = False

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # # # Generate images of receptacles by episode
    gen_pick_data(data_dir,dataset_file, env, agent, thread_num, args.manual,append=append, obj_pos_file='obj_position.pickle')

    # Close the h5py file
    dataset_file.close()
