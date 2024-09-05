# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
place 位姿数据采集代码：（若需要采集pick skill 需要重新初始化环境）
1. 原本假设gps和pos之间存在坐标旋转以及偏置关系，但测试后，发现随着机器人走动，这个关系似乎不太成立（原因暂不清楚）
2. 阅读源代码 gps获取过程后，按照源代码的方式将pos转为 gps,需要注意的是 env._reset_stats() 后，agent start postiont 和 start rotation都会变化，坐标系按照stat position 和 start rotation建立，因此也发生变化，因此env._reset_stats() 后要重新调用 env.get_gps() 获取 容器 gps


放置物体数据采集，由 projects/habitat_ovmm/receptacles_data_collection.py 复制而来
站在容器的每个waypoint，使用技能，记录是否成功
对每个waypoint,记录如下信息：
1. waypoint位置,机器人朝角
2. 容器位置
3. rgb, depth, semantic
4. 操作是否成功
NOTE 记录世界坐标系下的位置，后续需根据当前位置计算相对坐标

info: rl place会走动，然后放上去；gps和compass是以机器人start position 和 start rotation 建立坐标系的坐标

INFO: compass, gps 坐标系-可点开相应变量，有注释
x 轴正方向为 向前(m)
y 轴正方向为 向左(m)
compass 与x轴正方向的夹角(raid), 向左为正弧度

rot, pos 坐标系-暂未找到说明，不太清楚
rot 范围只有[0，3.7],没有负角度，似乎两个相差不多的角度，是反方向
pos三个维度中，只有第一维度和第三维度有意义，其中 rot 0度的时候对应第一维度，但rot为1.5（即90度）的时候有可能对应正的第三维度，也有可能对应负的第三维度
经实验，似乎 第一维度表示x，第二维度表示y，rot表示与x轴正半轴的夹角的绝对值（源代码注释是三个维度分别表示x,y,z）

TODO
采数据前，要重新采一遍容器位置数据，现在使用采集容器的结果替代(val已经全部采集)
大概估计一下运行完所有episode需要的时间
或许抽一部分 view_point采集数据？
view_point_position_s采集似乎不对?哪里不对？
统计每个容器的sr，实际部署时，选择哪些sr更高的容器作为目标容器

DONE
先测试相对位置变化关系，然后将容器位置转换为GPS
验证一下gps 位置对不对
经过初步测试后，在大批量采集数据的时候，可能要有针对性的采集一下
移动一下，把容器的位置标定出来（此法不通）
记录绝对坐标以及坐标变换关系，之后针对每个开始位置，计算相对坐标
将同一个容器的数据放在同一个list里面
只采集成功的数据，似乎也不行，还要训练模型预测是否成功（按照一定比例采集放置不成功的数据）
每种放置策略的数据单独放置，每个episode数据单独放置文件夹
先环顾四周，建立语义地图
rgb,depth,semantic可以用h5py存储，其它的可以用pickle存储

NOTE 
既然能看到容器，relative recep position 容器一定在agent前面，所以x一定为正
agent 的 fall_wait步数要为 200 ，否则东西没落到recep上，判定也是失败 
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
from home_robot.agent.ovmm_agent.ovmm_agent_skill_collect import OpenVocabManipAgent
from habitat.utils.visualizations import maps
import json
# from cyw.goal_point.utils import get_relative_position
# cyw/goal_point/data_prepare.py
# from cyw.goal_point.data_prepare import visual_obstacle_map,visual_init_obstacle_map
from cyw.goal_point.visualize import visual_obstacle_map,visual_init_obstacle_map
from tqdm import tqdm
from pathlib import Path

import random

random.seed(1234)
collect_fail_prob = 1 # TODO 当失败时，以collect_fail_prob的概率采集数据 

# src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py
show_image = False
debug = False

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
        # 如果放置成功：ovmm_place_object_phase_success 仍然是false，但 ovmm_place_success 会是 true

    # 参考 projects/habitat_ovmm/utils/metrics_utils.py
    # The task is considered successful if the agent places the object without robot collisions
    # overall_success = (
    #     episode_metrics["END.robot_collisions.robot_scene_colls"] == 0
    # ) * (episode_metrics["END.ovmm_place_success"] == 1)
    place_success = (hab_info['robot_collisions']['robot_scene_colls'] == 0) * (hab_info['ovmm_place_success'])
    return place_success

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
        if count_episodes == num_episodes:
            break
        # if count_episodes == 1:
        #     break

    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def gen_place_data(
    data_dir: str, 
    dataset_file: h5py.File,
    env: HabitatOpenVocabManipEnv, agent,
    manual=False,
    baseline_name:Optional[str]=None,
):
    """Generates images of receptacles by episode for all scenes"""

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

    recep_pos_dir = str(Path(data_dir).resolve().parent)
    with open(f"{recep_pos_dir}/recep_position.pickle", "rb") as handle:
        receptacle_positions = pickle.load(handle)

    count_episodes = 0
    
    # Ideally, we can make it like an iterator to make it feel more intuitive
    total_data = []
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
        dataset_file.create_group(
            f"/scene_{scene_id}/ep_{episode.episode_id}"
        )

        # for recep in receptacle_positions[scene_id]:
        recep = observations.task_observations['place_recep_name']
        object_name = observations.task_observations["object_name"]
        scene_ep_data = {
            "scene_id": scene_id,
            "episode_id":episode.episode_id,
            "recep": recep,
            "object_name": object_name,
            "skill_waypoint_data": []
        }
        recep_vals = receptacle_positions[scene_id][recep]
        for pos_pair in tqdm(recep_vals):
        # for pos_pair in tqdm(recep_vals[:2]): #TODO
            print("**************new position ***************")
            recep_position = np.array(pos_pair["recep_position"])
            scene_ep_recep_grp = dataset_file.create_group(f"/scene_{scene_id}/ep_{episode.episode_id}/{recep_position}") 
            view_point_positions = pos_pair["view_point_positions"]
            skill_waypoint_singile_recep_data = {
                "recep_position": recep_position,
                "each_view_point_data":[]
            }

            start_rgb_s = []
            start_semantic_s = []
            start_depth_s = []
            start_top_down_map_s = []
            start_obstacle_map_s = []
            view_point_position_s = [] # 为了验证，在h5py文件里面也加上 view_point_position_s

            for view_point_position in tqdm(view_point_positions):
            # for view_point_position in tqdm(list(view_point_positions)[:4]): #TODO
                view_point_position = np.array(view_point_position).astype(np.float32)
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
                # start_rot, start_pos = env.get_rot_pos()
                # 经实验：rot 和 start_rotation 会相差一些，大多数时候，要么成相反数，要么大小相差不多，似乎没有规律？ start_pos 和 start_position 一般是一致的

                start_rotation = env.get_current_rotation()
                start_position = np.array(env.get_current_position()).astype(np.float32)
                
                '''计算容器gps'''
                relative_recep_gps = env.get_relative_gps(recep_position) # 通过环境转换和通过 自己算的坐标转换的又不一样，但基本是 x 一样，y相差一些，不过似乎 y基本都很小（因为一开始的时候朝向容器了，所以容器基本都在agent正前方
                # 用于后续验证计算是否正确
                # if debug:
                #     cv2.imwrite(f"cyw/test_data/rgb_{relative_recep_gps}.jpg",cv2.cvtColor(start_observations.rgb,cv2.COLOR_BGR2RGB))


                if debug:
                    observations, done, hab_info = env.apply_action(DiscreteNavigationAction.EMPTY_ACTION)
                    start_agent_angle = hab_info['top_down_map']['agent_angle']
                    start_agent_map_coord = hab_info['top_down_map']['agent_map_coord']
                    print(f"agent_angle is {start_agent_angle}")
                    print(f"agent_map_coord is {start_agent_map_coord}")                    

                '''执行放置动作 '''
                map_id = 0
                while not done:
                    if not manual:
                        action, info, _ = agent.act(observations)
                        # sensor_pose: (7,) array denoting global pose (x, y, o) and local map boundaries planning window (gy1, gy2, gx1, gy2)
                    else:
                        manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, s: Stop, u: LookUp, n: LookDown)")
                        action,info = convertManualInput(manual_step)
                    observations, done, hab_info = env.apply_action(action, info)
                    print(f"action is {action}")

                    if debug:
                        print(f"agent_angle is {hab_info['top_down_map']['agent_angle']}")
                        print(f"agent_map_coord is {hab_info['top_down_map']['agent_map_coord']}")
                        top_down_map = draw_top_down_map(hab_info, observations.rgb.shape[0])
                        cv2.imshow("top_down_map",top_down_map)
                        cv2.waitKey(1)
                        if "obstacle_map" in info:
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
                            cv2.waitKey(1)
                            cv2.imwrite(f"cyw/test_data/init_obstacle_map/init_obstacle_map_{map_id}.jpg",init_obstacle_map_vis)
                            cv2.imwrite(f"cyw/test_data/obstacle_map/obstacle_map_{map_id}.jpg",obstacle_map_vis)


                    '''如果look around done, 收集 obstacle map and sensor pos'''
                    if "look_around_done" in info and info["look_around_done"]:
                        start_obstacle_map=info["obstacle_map"]
                        start_sensor_pose = info["sensor_pose"]

                        '''收集top down map 和 姿势'''
                        start_top_down_map=hab_info['top_down_map']['map']
                        start_top_down_map_pose = hab_info['top_down_map']['agent_map_coord']
                        start_top_down_map_rot = hab_info['top_down_map']["agent_angle"]

                        if debug:
                            assert start_top_down_map_pose == start_agent_map_coord,"start_agent_map_coord is wrong"
                            assert np.allclose(start_top_down_map_rot,start_agent_angle,rtol=0.01),"start_agent_angle is wrong"
                            # 不知道为什么角度会有小小的差别

                            # if start_top_down_map_pose != start_agent_map_coord:
                            #     print("start_agent_map_coord is wrong")
                            # if not np.allclose(start_top_down_map_rot,start_agent_angle,rtol=0.01):
                            #     # 不知道为什么角度会有小小的差别
                            #     print("start_agent_angle is wrong")
                        
                        '''可视化 top down map '''
                        if "top_down_map" in hab_info and show_image:
                            # # By default, `get_topdown_map_from_sim` returns image
                            # containing 0 if occupied, 1 if unoccupied, and 2 if border
                            top_down_map = draw_top_down_map(hab_info, observations.rgb.shape[0])
                            cv2.imshow("top_down_map",top_down_map)
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
                            cv2.waitKey(1)
                            cv2.imwrite(f"cyw/test_data/init_obstacle_map/init_obstacle_map_{map_id}.jpg",init_obstacle_map_vis)
                            cv2.imwrite(f"cyw/test_data/top_down_map/top_down_map_{map_id}.jpg",top_down_map)
                            cv2.imwrite(f"cyw/test_data/obstacle_map/obstacle_map_{map_id}.jpg",obstacle_map_vis)
                            # info['sensor_pose']
                            # # 保存 pickle 文件，以便调试
                            # with open(f"cyw/test_data/top_down_map_data/top_down_map_{map_id}.pkl","wb") as f:
                            #     pickle.dump(hab_info["top_down_map"],f)
                            # with open(f"cyw/test_data/info_data/info_{map_id}.pkl","wb") as f:
                            #     pickle.dump(info,f)
                            # map_id += 1
                            # 测试感觉没啥问题


                    # if debug:
                    #     # # 测试相对位姿
                    #     # current_position = env.get_current_position()
                    #     # robot_relative_start_position = get_relative_position(start_position,start_rotation,current_position)
                    #     # print(f"robot_relative_start_position is {robot_relative_start_position}")
                    #     # 经测试，代码正确

                    if show_image:
                        cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                        cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB))
                        semantic_img = get_semantic_vis(observations.semantic)
                        cv2.imshow("semantic",semantic_img)
                        cv2.waitKey(1)
                
                ''' 执行完毕,获取数据 '''
                # NOTE 需要在env._reset_stats之前，记录位姿信息
                end_position = np.array(env.get_current_position()).astype(np.float32)
                place_success = get_place_success(hab_info)
                if debug:
                    print(f"place success is {place_success}")
                
                # 记录数据
                if place_success:
                    record = True
                else:
                    record = random.random()<collect_fail_prob
                if record:
                    print(f"record data ********************")
                    start_rgb_s.append(start_observations.rgb)
                    start_depth_s.append(start_observations.depth)
                    start_semantic_s.append(start_observations.semantic)
                    start_obstacle_map_s.append(start_obstacle_map)
                    start_top_down_map_s.append(start_top_down_map)
                    view_point_position_s.append(view_point_position)
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

                agent.reset()
                env._reset_stats() # 重置一些状态，但不跳转到下一个episode
                # 重置状态后，start_position 和star_rotation都会变换，因此，需要重新计算坐标（现在记录绝对坐标，因此不需要重新计算）
                done = False

            '''运行完一个episode 的一个recep位置，保存数据'''
            scene_ep_data["skill_waypoint_data"].append(skill_waypoint_singile_recep_data)

            start_rgb_s = np.concatenate(start_rgb_s,axis=0)
            start_semantic_s = np.concatenate(start_semantic_s,axis=0)
            start_depth_s = np.concatenate(start_depth_s,axis=0)
            start_top_down_map_s = np.concatenate(start_top_down_map_s,axis=0)
            start_obstacle_map_s = np.concatenate(start_obstacle_map_s,axis=0)
            view_point_position_s = np.concatenate(view_point_position_s,axis=0)

            scene_ep_recep_grp.create_dataset(name="start_rgb_s",data=start_rgb_s)
            scene_ep_recep_grp.create_dataset(name="start_semantic_s",data=start_semantic_s)
            scene_ep_recep_grp.create_dataset(name="start_depth_s",data=start_depth_s)
            scene_ep_recep_grp.create_dataset(name="start_top_down_map_s",data=start_top_down_map_s)
            scene_ep_recep_grp.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
            scene_ep_recep_grp.create_dataset(name="view_point_position_s",data=view_point_position_s)
            
        # 运行完一个episode,保存数据
        total_data.append(scene_ep_data)
        with open(os.path.join(data_dir,"place_waypoint.pkl"),"wb") as f:
            pickle.dump(total_data,f)

        dataset_file.flush()
        # if count_episodes == num_episodes:
        #     break
        if count_episodes == 2:
            break
    
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
    args = parser.parse_args()

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
    agent = OpenVocabManipAgent(agent_config, device_id=device_id)

    baseline_name = args.baseline_config_path.split("/")[-1].split(".")[0]
    data_dir = os.path.join(args.data_dir,baseline_name)
    os.makedirs(f"./{data_dir}", exist_ok=True)
    # Create h5py files
    dataset_file = h5py.File(f"./{data_dir}/{args.datafile}.hdf5", "w") # NOTE 这会覆盖掉原本的文件

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # # Generate images of receptacles by episode
    gen_place_data(data_dir,dataset_file, env, agent,args.manual)


    # Close the h5py file
    dataset_file.close()
