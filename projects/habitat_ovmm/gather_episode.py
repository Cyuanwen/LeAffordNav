import argparse
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from evaluator import create_ovmm_env_fn
from utils.config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config


from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys


def extract_scene_id(scene_id: str) -> str:
    """extracts scene id from string containing the scene id"""

    before, _ = scene_id.split(".scene")
    _, after = before.split("uncluttered/")
    return after

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

def gather_episode(data_dir:str,env:HabitatOpenVocabManipEnv):
    '''获取episode，使得同一个场景，同一个容器，只用一个episode
    '''
        # This is for iterating through all episodes once using only one env
    count_dict, num_episodes = get_init_scene_episode_count_dict(env)

    receptacle_positions = {}
    count_episodes = 0
    episode_ids = []

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
        # if not (scene_id in receptacle_positions and episode.goal_recep_category in receptacle_positions[scene_id]):
        # NOTE
        if not (scene_id in receptacle_positions and episode.start_recep_category in receptacle_positions[scene_id]):
            # 该场景的该容器之前没有记录过
            episode_ids.append(episode.episode_id)

            if not scene_id in receptacle_positions:
                receptacle_positions[scene_id] = {}

            if not episode.goal_recep_category in receptacle_positions[scene_id]:
                receptacle_positions[scene_id][episode.goal_recep_category] = []

        print_progress(count_episodes, num_episodes, prefix='count_episodes: %d/%d'%((count_episodes),num_episodes))
        if count_episodes == num_episodes:
            break
        # if count_episodes == 1:
        #     break

    print(f"****************save data to ./{data_dir} ****************")
    os.makedirs(f"./{data_dir}", exist_ok=True)
    with open(f"./{data_dir}/episode_ids.json", "w") as f:
        json.dump(episode_ids,f)
    print("********* save done **************")


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
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scene_ep_data",
        help="Path to saving scene info",
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

    data_dir = args.data_dir
    os.makedirs(f"./{data_dir}", exist_ok=True)

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # gather episode
    gather_episode(data_dir=data_dir,env=env)