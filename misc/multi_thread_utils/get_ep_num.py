"""
获得数据量大小
"""
from typing import Optional, Tuple
import argparse

from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)
import os
home_root = os.environ.get("HOME_ROBOT_ROOT")
import sys
sys.path.append(f"{home_root}/projects/habitat_ovmm/")
from evaluator import create_ovmm_env_fn
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

def get_episode_count(
    env: HabitatOpenVocabManipEnv,
) -> Tuple[dict, int]:
    num_episodes = len(env._dataset.episodes)
    return num_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        # default="/raid/home-robot/src/third_party/habitat-lab/habitat-baselines/habitat_baselines/config/ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/heuristic_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "random", "explore", "zxy_pick_place","place","gaze_place"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--force_step",
        type=int,
        default=20,
        help="force to switch to new episode after a number of steps",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="whether to save obseration history for data collection",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    # @cyw
    parser.add_argument(
        "--id_file", #运行特定的id 列表
        type=str,
        default=None,
        help="whether to use id file to run"
    )
    parser.add_argument(
        "--EXP_NAME_suffix", 
        type=str,
        default=None,
        help="whether to add suffix to the env config experiment name"
    )
    args = parser.parse_args()

    if args.id_file is not None:
        import json
        with open(args.id_file,"r") as f:
            episode_ids = json.load(f)
        args.overrides.append(f"habitat.dataset.episode_ids={episode_ids}")
        # NOTE EXP_NAME_suffix与habitat.dataset.episode_ids={episode_ids}不能同时使用

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # @cyw
    if args.EXP_NAME_suffix is not None:
        from omegaconf import DictConfig, OmegaConf
        OmegaConf.set_readonly(env_config, False)
        env_config.EXP_NAME = env_config.EXP_NAME +"_" + args.EXP_NAME_suffix
        OmegaConf.set_readonly(env_config, True)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # Create an env
    env = create_ovmm_env_fn(env_config)

    ep_num = get_episode_count(env)

    print(ep_num)

    # ep_num = 1199
    # print(1199)
    # os.environ["EP_NUM"] = f"{ep_num}"
    # print(f"export EP_NUM={ep_num}")
    # 写入结果到文件  
    os.makedirs("temp/",exist_ok=True)
    with open(f"temp/ep_num.txt", "w") as f:  
        f.write(f"{ep_num}")

    print("over")