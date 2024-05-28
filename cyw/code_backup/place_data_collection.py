# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
放置物体数据采集，由 projects/habitat_ovmm/receptacles_data_collection.py 复制而来
'''
import argparse
import os
import pickle
from typing import Tuple

import h5py
import numpy as np
from evaluator import create_ovmm_env_fn
from habitat.tasks.rearrange.utils import get_robot_spawns
from utils.config_utils import create_env_config, get_habitat_config, get_omega_config

from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

all_receptacles = [
    "cabinet",
    "stool",
    "trunk",
    "shoe_rack",
    "chest_of_drawers",
    "table",
    "toilet",
    "serving_cart",
    "bed",
    "washer_dryer",
    "hamper",
    "stand",
    "bathtub",
    "couch",
    "counter",
    "shelves",
    "chair",
    "bench",
]
# @cyw
# {'sink', 'wardrobe', 'filing_cabinet'} 缺少 21 类中的这三类
from habitat.core.simulator import AgentState
import cv2
from home_robot.core.interfaces import DiscreteNavigationAction
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
# src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py
show_image = True
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


def gen_receptacle_images(
    data_dir: str, dataset_file: h5py.File, env: HabitatOpenVocabManipEnv
):
    """Generates images of receptacles by episode for all scenes"""

    # sim = env.habitat_env.env._env._env._sim
    sim = env.habitat_env.env.habitat_env._sim
    task = env.habitat_env.env.env.habitat_env.task

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
        obs = env.reset()
        episode = env.get_current_episode()
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

        scene_ep_grp = dataset_file.create_group(
            f"/scene_{scene_id}/ep_{episode.episode_id}"
        )

        for recep in receptacle_positions[scene_id]:
            recep_vals = list(receptacle_positions[scene_id][recep])

            if (
                len(recep_vals) > 4
            ):  # Too many views around same receptacle can be unneccassary
                np.random.shuffle(recep_vals)
                recep_len = np.random.randint(1, 5)
                recep_vals = recep_vals[:recep_len]

            recep_images = []
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
                # @ cyw可能接口有变化，进行了一些修改
                # sim.robot.base_pos = start_position               
                # sim.robot.base_rot = start_rotation               
                # sim.maybe_update_robot()
                # recep_images.append(obs["robot_head_rgb"][:, :, :3][None])
                sim.articulated_agent.base_pos = start_position
                sim.articulated_agent.base_rot = start_rotation
                sim.maybe_update_articulated_agent()
                obs = sim.get_sensor_observations()
                # recep_images.append(obs["head_rgb"][:, :, :3][None])
                if show_image:
                    cv2.imshow("rgb",cv2.cvtColor(obs["head_rgb"],cv2.COLOR_BGR2RGB))
                    cv2.imshow("third_rgb",cv2.cvtColor(obs["third_rgb"],cv2.COLOR_BGR2RGB))
                    cv2.waitKey()
                
                # if manipulation == "place":
                #     # 强制抓起一个东西
                #     abs_obj_idx = sim.scene_obj_ids[task.abs_targ_idx]
                #     sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)
                # else:
                #     # Remove whatever the agent is currently holding.
                #     abs_obj_idx = sim.scene_obj_ids[task.abs_targ_idx]
                #     sim.grasp_mgr.desnap(force=True)
                #     # sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)
                # sim.maybe_update_articulated_agent()
                # obs = sim.get_sensor_observations()
                
                # 调用env 直接抓起一个东西
                observations = env.pick_up_obj(count_episodes-1) #NOTE 因为在前面就加上1，因此这里要减1
                action = DiscreteNavigationAction.EMPTY_ACTION
                observations, done, hab_info = env.apply_action(action)
                
                if show_image:
                    cv2.imshow("rgb",cv2.cvtColor(obs["head_rgb"],cv2.COLOR_BGR2RGB))
                    cv2.imshow("third_rgb",cv2.cvtColor(obs["third_rgb"],cv2.COLOR_BGR2RGB))
                    semantic_img = get_semantic_vis(observations.semantic)
                    # semantic = np.asarray(observations.semantic,np.uint8)
                    # semantic_img = Image.new("P",(semantic.shape[1],semantic.shape[0]))
                    # semantic_img.putpalette(d3_40_colors_rgb.flatten())
                    # semantic_img.putdata(semantic.flatten())
                    # semantic_img = semantic_img.convert("RGB")
                    # semantic_img = np.asarray(semantic_img)
                    # 不知道为什么这样写做出来的图是混乱的
                    # 因为少了一句 semantic_map_vis = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]？ 不应该啊
                    cv2.imshow("semantic",semantic_img)
                    cv2.waitKey()

            # recep_images = np.concatenate(recep_images, axis=0)  # Shape is (N, H, W, 3)
            # scene_ep_grp.create_dataset(recep, data=recep_images)

        dataset_file.flush()

        if count_episodes == num_episodes:
            break


def gen_dataset_question(
    data_dir: str, dataset_file: h5py.File, env: HabitatOpenVocabManipEnv
):
    """Generates templated Q/A per episode for all scenes"""

    # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    with open(f"./{data_dir}/recep_position.pickle", "rb") as handle:
        receptacle_positions = pickle.load(handle)

    count_episodes = 0

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        env.reset()
        episode = env.get_current_episode()
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
        scene_ep_grp = dataset_file[f"/scene_{scene_id}/ep_{episode.episode_id}"]

        scene_receptacles = list(receptacle_positions[scene_id].keys())
        scene_receptacles_id = {}

        recep_idx = 0
        for recep in scene_receptacles:
            n_images = scene_ep_grp[recep].shape[0]
            scene_receptacles_id[recep] = list(range(recep_idx, recep_idx + n_images))
            recep_idx += n_images

        questions = []
        for recep in all_receptacles:
            question_str = f"We show images from different locations inside a home. Which location(s) contain {recep.replace('_', ' ')}? You can choose multiple options\n"
            options_str = "Options: "
            for idx in range(recep_idx):
                options_str += f"image {idx}: <tok></tok>, "
            options_str = options_str[:-2] + "\n"
            answer_str = "Answer with image index: "

            if recep in scene_receptacles:
                for idx in scene_receptacles_id[recep]:
                    answer_str += f"<{idx}> "
                answer_str = answer_str[:-1]
            else:
                answer_str += "Not found"

            prompt_str = question_str + options_str + answer_str
            questions.append(prompt_str)

        questions = [question.encode("ascii", "ignore") for question in questions]
        scene_ep_grp.create_dataset("questions", (len(questions), 1), "S10", questions)
        dataset_file.flush()

        if count_episodes == num_episodes:
            break


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

    # Create h5py files
    os.makedirs(f"./{args.data_dir}", exist_ok=True)
    dataset_file = h5py.File(f"./{args.data_dir}/{args.datafile}.hdf5", "w")

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # @cyw
    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # Generate images of receptacles by episode
    gen_receptacle_images(args.data_dir, dataset_file, env)

    # @cyw
    # # Generate templated Q/A per episode
    # gen_dataset_question(args.data_dir, dataset_file, env)

    # Close the h5py file
    dataset_file.close()
