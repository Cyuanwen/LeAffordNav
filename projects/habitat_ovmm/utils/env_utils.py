# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING

from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.gym.gym_definitions import _get_env_name

from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def create_ovmm_env_fn(config: "DictConfig") -> HabitatOpenVocabManipEnv:
    """
    Creates an environment for the OVMM task.

    Creates habitat environment from config and wraps it into HabitatOpenVocabManipEnv.

    :param config: configuration for the environment.
    :return: environment instance.
    """
    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env

# @cyw habitat-lab/habitat/gym/gym_definitions.py rl 训练里面的 env 初始化
# def make_gym_from_config(config: "DictConfig", dataset=None) -> gym.Env:
#     """
#     From a habitat-lab or habitat-baseline config, create the associated gym environment.
#     """
#     if "habitat" in config:
#         config = config.habitat
#     env_class_name = _get_env_name(config)
#     env_class = get_env_class(env_class_name)
#     assert (
#         env_class is not None
#     ), f"No environment class with name `{env_class_name}` was found, you need to specify a valid one with env_task"
#     return make_env_fn(env_class=env_class, config=config, dataset=dataset)