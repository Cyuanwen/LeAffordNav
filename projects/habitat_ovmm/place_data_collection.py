# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
放置物体数据采集，由 projects/habitat_ovmm/receptacles_data_collection.py 复制而来
站在容器的每个waypoint，尝试放置或抓取，记录是否成功
对每个waypoint,记录如下信息：
1. waypoint位置,机器人朝角
2. 容器位置
3. rgb, depth, semantic
4. 操作是否成功
5. 机器人摄像头朝角？
info: rl place会走动，然后放上去
'''
import argparse
import os
import pickle
from typing import Tuple

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

# src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py
show_image = False
debug = True
manipulation = "place" # 操作技能 place or pick

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

def get_receptacle_position(data_dir: str, env: HabitatOpenVocabManipEnv):
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
            receptacle_positions[scene_id][episode.goal_recep_category] = set()
        for recep in episode.candidate_goal_receps:
            recep_position = list(recep.position)
            # view_point_position = list(recep.view_points[0].agent_state.position)
            # @cyw
            view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,recep.view_points[0]).position)
            receptacle_positions[scene_id][episode.goal_recep_category].add(
                tuple(recep_position + view_point_position)
            )
        # @cyw
        # if count_episodes == num_episodes:
        #     break
        if count_episodes == 10:
            break

    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
            receptacle_positions[scene_id][episode.goal_recep_category] = set()
        for recep in episode.candidate_goal_receps:
            recep_position = list(recep.position)
            # view_point_position = list(recep.view_points[0].agent_state.position)
            # @cyw 只搜集每个episode第一个waypoint
            # view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,recep.view_points[0]).position)
            # receptacle_positions[scene_id][episode.goal_recep_category].add(
            #     tuple(recep_position + view_point_position)
            # )
            # 搜索所有waypoint
            for view_point in recep.view_points:
                view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,view_point).position)
                receptacle_positions[scene_id][episode.goal_recep_category].add(
                    tuple(recep_position + view_point_position)
                )
        # @cyw
        # if count_episodes == num_episodes:
        #     break
        if count_episodes == 10:
            break

    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gen_place_data(
    data_dir: str, dataset_file: h5py.File, env: HabitatOpenVocabManipEnv, agent
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

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        # obs = env.reset()
        observations, done = env.reset(), False #跳转到下一个episode
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)
        agent.reset()
        if debug:
            print(f"episode id {episode.episode_id}")

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
        recep_vals = list(receptacle_positions[scene_id][recep])

        for pos_pair in recep_vals:
            pos_pair_lst = list(pos_pair)
            recep_position = np.array(pos_pair_lst[:3])
            view_point_position = np.array(pos_pair_lst[3:]).astype(np.float32)
            start_position, start_rotation, _ = get_robot_spawns(
                target_positions=view_point_position[None],
                rotation_perturbation_noise=0,
                distance_threshold=0,
                sim=sim,
                num_spawn_attempts=100,
                physics_stability_steps=100,
                orient_positions=recep_position[None],
            )
            observations = env.set_position(start_position,start_rotation)
            rot, pos = env.get_rot_pos()
            if debug:
                if rot != start_rotation:
                    print(f"new rot {rot} is not equall with start_rotation {start_rotation}")
                    # rot 和 start_rotation 会相差一些，大多数时候，要么成相反数，要么大小相差不多，似乎没有规律？
                if pos != start_position:
                    print(f"new pos {pos} is not equall with start_position {start_position}")
            observations = env.pick_up_obj() #NOTE 因为在前面就加上1，因此这里要减1
            done = False
            if show_image:
                cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB))
                semantic_img = get_semantic_vis(observations.semantic)
                cv2.imshow("semantic",semantic_img)
                cv2.waitKey() 
            
            # 执行放置动作
            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = env.apply_action(action, info)
                # if debug:
                #     new_rot, new_pos = env.get_rot_pos()
                #     if new_rot != rot:
                #         print(f"new_rot {new_rot} is not equall with rot {rot}")
                #     if new_pos != pos:
                #         print(f"new_pos {new_pos} is not equall with pos {pos}")
                if show_image:
                    cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                    cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB))
                    semantic_img = get_semantic_vis(observations.semantic)
                    cv2.imshow("semantic",semantic_img)
                    cv2.waitKey()
                if debug:
                    if done:
                        print("debug")
            if debug:
                new_rot, new_pos = env.get_rot_pos()
                if new_rot != rot:
                    print(f"new_rot {new_rot} is not equall with rot {rot}")
                if new_pos != pos:
                    print(f"new_pos {new_pos} is not equall with pos {pos}")
                # NOTE 需要在env._reset_stats之前，记录位姿信息
                # 执行完动作，机器人的位置似乎就不对了？不是因为执行动作使得机器人位置不对，而是因为env._reset_stats()使得机器人位置不对 
            place_success = get_place_success(hab_info)
            if place_success:
                print("debug")
            # TODO 做好结果记录
            agent.reset()
            env._reset_stats() # 重置一些状态，但不跳转到下一个episode
            

                
            # recep_images = np.concatenate(recep_images, axis=0)  # Shape is (N, H, W, 3)
            # scene_ep_grp.create_dataset(recep, data=recep_images)
        
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
    gen_place_data(args.data_dir, dataset_file, env, agent)


    # Close the h5py file
    dataset_file.close()
