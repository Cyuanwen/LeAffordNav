# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
测试代码：
1. 原本假设gps和pos之间存在坐标旋转以及偏置关系，但测试后，发现随着机器人走动，这个关系似乎不太成立（原因暂不清楚）
2. 阅读源代码 gps获取过程后，按照源代码的方式将pos转为 gps,需要注意的是 env._reset_stats() 后，agent start postiont 和 start rotation都会变化，坐标系按照stat position 和 start rotation建立，因此也发生变化，因此env._reset_stats() 后要重新调用 env.get_gps() 获取 容器 gps


放置物体数据采集，由 projects/habitat_ovmm/receptacles_data_collection.py 复制而来
站在容器的每个waypoint，使用技能，记录是否成功
对每个waypoint,记录如下信息：
1. waypoint位置,机器人朝角
2. 容器位置
3. rgb, depth, semantic
4. 操作是否成功

info: rl place会走动，然后放上去

INFO: compass, gps 坐标系-可点开相应变量，有注释
x 轴正方向为 向前(m)
y 轴正方向为 向左(m)
compass 与x轴正方向的夹角(raid), 向左为正弧度

rot, pos 坐标系-暂未找到说明，不太清楚
rot 范围只有[0，3.7],没有负角度，似乎两个相差不多的角度，是反方向
pos三个维度中，只有第一维度和第三维度有意义，其中 rot 0度的时候对应第一维度，但rot为1.5（即90度）的时候有可能对应正的第三维度，也有可能对应负的第三维度
经实验，似乎 第一维度表示x，第二维度表示y，rot表示与x轴正半轴的夹角的绝对值（源代码注释是三个维度分别表示x,y,z）

TODO
将同一个容器的数据放在同一个list里面
每中放置策略的数据单独放置
移动一下，把容器的位置标定出来

DONE
先测试相对位置变化关系，然后将容器位置转换为GPS
验证一下gps 位置对不对
经过初步测试后，在大批量采集数据的时候，可能要有针对性的采集一下

NOTE 采数据前，要重新采一遍容器位置数据，现在使用采集容器的结果替代
'''
import argparse
import os
import pickle
from turtle import rt
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
from home_robot.agent.ovmm_agent.place_agent import PlaceAgent #单独的放置agent
from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
import json
# cyw/goal_point/utils.py
from cyw.goal_point.utils import coordinate_transform, coord_transfer

# src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py
show_image = False
debug = True

Foward = [
    ["ahead"],
    ["left_90","ahead"],
    ["right_90","ahead"],
    ["left_90","left_90","ahead"]
]
name2action = {
    "ahead":[DiscreteNavigationAction.MOVE_FORWARD],
    "left_90": [DiscreteNavigationAction.TURN_LEFT] * 3,# 每次移动是 30 度
    "right_90": [DiscreteNavigationAction.TURN_RIGHT] * 3 ,
}
Foward_actions = []
for actions in Foward:
    discrete_actions = []
    for name in actions:
        discrete_actions += name2action[name]
    Foward_actions.append(discrete_actions)

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
        for recep in episode.candidate_goal_receps:
            recep_position = list(recep.position) # recep数据里面没有朝向
            # 搜索所有waypoint
            view_point_positions = set()
            for view_point in recep.view_points:
                view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,view_point).position)
                view_point_positions.add(tuple(view_point_position))
            receptacle_positions[scene_id][episode.goal_recep_category].append(
                {
                    "recep_position": recep_position,
                    "view_point_positions":view_point_positions
                }
            )

        # # @cyw
        # if count_episodes == num_episodes:
        #     break
        if count_episodes == 50:
            break

    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def gen_place_data(
    data_dir: str, dataset_file: h5py.File, env: HabitatOpenVocabManipEnv, agent,
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

    with open(f"./{data_dir}/recep_position.pickle", "rb") as handle:
        receptacle_positions = pickle.load(handle)

    count_episodes = 0
    
    coordinate_transformer = coordinate_transform()

    # Ideally, we can make it like an iterator to make it feel more intuitive
    data = []
    while True:
        # Get a new episode
        # obs = env.reset()
        observations, done = env.reset(), False #跳转到下一个episode
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)
        agent.reset()
        if debug:
            print(f"*************new episode episode id {episode.episode_id}**************")

        coordinate_transformer.reset()

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f"ep_{episode.episode_id}_scene_{scene_id}"
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
            count_episodes += 1
        else:
            raise ValueError(
                "count_dict[hash_str] is 0 when hash_str is called for the first time."
            )

        scene_ep_grp = dataset_file.create_group(
            f"/scene_{scene_id}/ep_{episode.episode_id}"
        )

        # for recep in receptacle_positions[scene_id]:
        recep = observations.task_observations['place_recep_name']
        recep_vals = receptacle_positions[scene_id][recep]
        scene_ep_data = []
        success = 0
        for pos_pair in recep_vals:
            print("**************new position ***************")
            recep_position = np.array(pos_pair["recep_position"])
            # 计算recep 在gps坐标系下的位置
            recep_gps = env.get_gps(recep_position)
            robot_gps = env.get_gps()
            if not np.allclose(robot_gps,observations.gps,rtol=0.01):
                print("wrong")
            view_point_positions = pos_pair["view_point_positions"]
            for view_point_position in view_point_positions:
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
                start_observations = env.set_position(start_position,start_rotation)
                start_rot, start_pos = env.get_rot_pos()
                # 经实验：rot 和 start_rotation 会相差一些，大多数时候，要么成相反数，要么大小相差不多，似乎没有规律？ start_pos 和 start_position 一般是一致的
                start_compass = start_observations.compass
                start_gps = start_observations.gps
                
                '''gps 标定 '''
                relative_recep_gps = None # 每一个view point点，相对位置都会变化，都要重新计算
                if relative_recep_gps is None:
                    for index in range(len(Foward_actions)):
                        # NOTE 坐标系变换关系在机器人运动过程中还在变换，这样得到的gps可能是错的
                        actions = Foward_actions[index]
                        for action in actions:
                            observations, done, hab_info = env.apply_action(action)
                            if show_image:
                                cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                                cv2.waitKey(1)
                        new_gps = observations.gps
                        if (new_gps != start_gps).any():
                            _, new_pos = env.get_rot_pos()
                            coordinate_transformer.set_rotation_offset(start_pos,new_pos,start_gps,new_gps)
                            recep_gps_my = coordinate_transformer.get_gps_coord(recep_position)
                            recep_gps = env.get_gps(recep_position)
                            if not np.allclose(recep_gps,recep_gps_my,rtol=0.01):
                                print("gps transfer is wrong")
                            relative_recep_gps = coord_transfer(
                                robot_gps=start_gps,
                                compass=start_compass,
                                gps=recep_gps
                                )
                            if debug:
                                cv2.imwrite(f"cyw/test_data/rgb_{relative_recep_gps}.jpg",cv2.cvtColor(start_observations.rgb,cv2.COLOR_BGR2RGB))
                                transferd_new_gps = coordinate_transformer.get_gps_coord(new_pos)
                                if not np.allclose(transferd_new_gps,observations.gps,rtol=0.01):
                                    print("error the position to gps transfer is wrong")
                            break
                        else:
                            # 可能前方有障碍物，没走成功，将机器人放回原始位置
                            observations = env.set_position(start_position,start_rotation)
                            if debug:
                                print(f"place robot to initial location {observations.gps == start_gps}, {observations.compass == start_compass}")
                
                    observations = env.set_position(start_position,start_rotation)
                    if debug:
                        print(f"place robot to initial location {observations.gps == start_gps}, {observations.compass == start_compass}")
                               
                if relative_recep_gps is None: #如果没有获得 recep gps，则放弃该 view_point_position
                    continue

                # 拿起任务中要放的东西
                observations = env.pick_up_obj()

                '''执行放置动作 '''
                while not done:
                    if not manual:
                        action, info, _ = agent.act(observations)
                    else:
                        manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, s: Stop, u: LookUp, n: LookDown)")
                        action,info = convertManualInput(manual_step)
                    observations, done, hab_info = env.apply_action(action, info)
                    print(f"action is {action}")

                    if debug:
                        # 测试 坐标变化是否正确
                        _, new_pos = env.get_rot_pos()
                        new_gps = coordinate_transformer.get_gps_coord(new_pos)
                        print(f"new_pos is {new_pos}, new_rotation is {_}, compass {observations.compass}")
                        if not np.allclose(new_gps,observations.gps,rtol=0.01):
                            print(f"error the position to gps transfer is wrong, transfered gps is {new_gps}, gt gps is {observations.gps}")
                            # 可能 pos 坐标系 一直 在随着运动变化？
                        robot_gps = env.get_gps()
                        if not np.allclose(robot_gps,observations.gps,rtol=0.01):
                            print("wrong")

                    if show_image:
                        cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                        cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB))
                        semantic_img = get_semantic_vis(observations.semantic)
                        cv2.imshow("semantic",semantic_img)
                        cv2.waitKey(1)
                
                ''' 执行完毕,获取数据 '''
                # NOTE 需要在env._reset_stats之前，记录位姿信息
                end_gps = observations.gps
                end_compass = observations.compass
                if debug:
                    # 测试 坐标变化是否正确
                    _, new_pos = env.get_rot_pos()
                    new_gps = coordinate_transformer.get_gps_coord(new_pos)
                    if not np.allclose(new_gps,end_gps,rtol=0.01):
                        print("error the position to gps transfer is wrong")
                    # TODO 这两个gps对不上

                place_success = get_place_success(hab_info)
                if debug:
                    print(f"place success is {place_success}")
                


                agent.reset()
                coordinate_transformer.reset()
                env._reset_stats() # 重置一些状态，但不跳转到下一个episode
                # 重置状态后，start_position 和star_rotation都会变换，因此，需要重新计算坐标
            
        # 运行完一个episode
        place_sr = success / len(recep_vals)
        print(f"*************the place sr is {place_sr} **********")
        data.append(
            {
                "scene_id":scene_id,
                "episode": episode.episode_id,
                "scene_ep_data":scene_ep_data
            }
        )
        with open(os.path.join(data_dir,f"{baseline_name}_test.pkl"),"wb") as f:
            pickle.dump(data,f)
        dataset_file.flush()

        # if count_episodes == num_episodes:
        #     break
        if count_episodes == 10:
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
    device_id = 0
    # agent = PlaceAgent(agent_config, device_id=device_id)
    agent = OpenVocabManipAgent(agent_config, device_id=device_id)

    # Create h5py files
    os.makedirs(f"./{args.data_dir}", exist_ok=True)
    dataset_file = h5py.File(f"./{args.data_dir}/{args.datafile}.hdf5", "w")

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # Generate images of receptacles by episode
    baseline_name = args.baseline_config_path.split("/")[-1].split(".")[0]
    gen_place_data(args.data_dir, dataset_file, env, agent,args.manual,baseline_name)


    # Close the h5py file
    dataset_file.close()
