# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from place_data_collection import get_semantic_vis
from cyw.detection_utils.yolo_train.extra_label import extract_labels
# show_image = True
show_image = False

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
            ) #NOTE 严格来说，这里面的东西应该确保不重复，否则，会收集到很多重复图像，也应该尽量不遗漏，否则，可能会漏掉一些数据
        # @cyw
        if count_episodes == num_episodes:
            break
        # if count_episodes == 10:
        #     break

    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/recep_position.pickle", "wb") as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def gen_receptacle_images(
#     data_dir: str, dataset_file: h5py.File, env: HabitatOpenVocabManipEnv
# ):
def gen_receptacle_images(
    data_dir: str, env: HabitatOpenVocabManipEnv
):
    """Generates images of receptacles by episode for all scenes"""
    '''
        将原本代码修改为以rgb 和 label格式存储
    '''

    # sim = env.habitat_env.env._env._env._sim
    sim = env.habitat_env.env.habitat_env._sim

    # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    # Also, creating folders for storing dataset
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        # if f"scene_{scene_id}" not in dataset_file:
        #     dataset_file.create_group(f"scene_{scene_id}")
        if not os.path.exists(os.path.join(data_dir,scene_id)):
            os.makedirs(os.path.join(data_dir,scene_id),exist_ok=True)
            os.mkdir(os.path.join(data_dir,scene_id,"images"))
            os.mkdir(os.path.join(data_dir,scene_id,"labels"))

    with open(f"./{data_dir}/recep_position.pickle", "rb") as handle:
        receptacle_positions = pickle.load(handle)
    # with open("cyw/datasets/datasets_v1/recep_data/val/recep_position.pickle", "rb") as handle:
    #     receptacle_positions = pickle.load(handle)

    count_episodes = 0
    count_image = 0

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        observations = env.reset()
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)
        # if scene_id != "104348361_171513414":
        #     continue

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f"ep_{episode.episode_id}_scene_{scene_id}"
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
            count_episodes += 1
        else:
            raise ValueError(
                "count_dict[hash_str] is 0 when hash_str is called for the first time."
            )

        # scene_ep_grp = dataset_file.create_group(
        #     f"/scene_{scene_id}/ep_{episode.episode_id}"
        # )

        for recep in receptacle_positions[scene_id]:
            recep_vals = list(receptacle_positions[scene_id][recep])

            if (
                len(recep_vals) > 4
            ):  # Too many views around same receptacle can be unneccassary
                np.random.shuffle(recep_vals)
                recep_len = np.random.randint(1, 5)
                recep_vals = recep_vals[:recep_len]

            # recep_images = []
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

                # 以下代码无法获得segmentation
                # sim.articulated_agent.base_pos = start_position
                # sim.articulated_agent.base_rot = start_rotation
                # sim.maybe_update_articulated_agent()
                # obs = sim.get_sensor_observations()
                # recep_images.append(obs["head_rgb"][:, :, :3][None])
                # if show_image:
                #     cv2.imshow("rgb",cv2.cvtColor(obs["head_rgb"],cv2.COLOR_BGR2RGB))
                #     # semantic = obs['head_panoptic'] # 这是所有实例分割的数据
                #     cv2.waitKey()
                observations = env.set_position(start_position,start_rotation)
                if show_image:
                    cv2.imshow("rgb",cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                    # cv2.imshow("third_rgb",cv2.cvtColor(observations.third_person_image,cv2.COLOR_BGR2RGB)) # 需要env 配置中指定 visualize才会有 third_rgb
                    semantic_img = get_semantic_vis(observations.semantic)
                    cv2.imshow("semantic",semantic_img)
                    cv2.waitKey()
                # 保存图像和semantic
                cv2.imwrite(os.path.join(data_dir,scene_id,"images",f"{count_image}.png"),cv2.cvtColor(observations.rgb,cv2.COLOR_BGR2RGB))
                # np.save(os.path.join(data_dir,scene_id,"semantic",f"{count_image}.npy"),observations.semantic)
                extract_labels(observations.semantic,os.path.join(data_dir,scene_id,"labels",f"{count_image}.txt"))
                count_image += 1

            # recep_images = np.concatenate(recep_images, axis=0)  # Shape is (N, H, W, 3)
            # scene_ep_grp.create_dataset(recep, data=recep_images)

        # dataset_file.flush()

        if count_episodes == num_episodes:
            break
        # if count_episodes == 10:
        #     break


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
    # dataset_file = h5py.File(f"./{args.data_dir}/{args.datafile}.hdf5", "w")

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # # @cyw
    # # Aggregate receptacles position by scene using all episodes
    # receptacle_position_aggregate(args.data_dir, env)

    # Generate images of receptacles by episode
    # gen_receptacle_images(args.data_dir, dataset_file, env)
    gen_receptacle_images(args.data_dir, env)

    # @cyw
    # # Generate templated Q/A per episode
    # gen_dataset_question(args.data_dir, dataset_file, env)

    # # Close the h5py file
    # dataset_file.close()
