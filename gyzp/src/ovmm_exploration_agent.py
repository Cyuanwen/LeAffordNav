#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-
# quick fix for import

import sys
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import torch

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations

sys.path.append(r"/raid/home-robot/gyzp/utils")
import json

from dataset.extract import extract_goal_object, extract_labels
from PIL import Image

labels_counter = 0


class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()
    EXPLORE = auto()
    NAV_TO_INSTANCE = auto()
    FALL_WAIT = auto()


class OVMMExplorationAgent(OpenVocabManipAgent):
    def __init__(
        self, config, device_id: int = 0, obs_spaces=None, action_spaces=None, args=None
    ):
        super().__init__(config, device_id=device_id)
        print("Exploration Agent created")
        self.args = args

        labels_file_path = "/raid/home-robot/projects/real_world_ovmm/configs/example_cat_map.json"
        with open(labels_file_path, "r") as f:
            self.labels_dict = json.load(f).get("obj_category_to_obj_category_id")

    def _explore(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        if self.skip_skills.nav_to_obj:
            terminate = True
        elif nav_to_obj_type == "heuristic":
            if self.verbose:
                print("[OVMM AGENT] step heuristic nav policy")
            action, info, terminate = self._heuristic_nav(obs, info)
        elif nav_to_obj_type == "rl":
            action, info, terminate = self.nav_to_obj_agent.act(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
            )
        new_state = None
        if terminate:
            action = None
            new_state = True
        return action, info, new_state

    def reset_vectorized(self, episodes=None):
        """Initialize agent state."""
        super().reset_vectorized()
        self.states = torch.tensor([Skill.EXPLORE] * self.num_environments)
        self.remaining_actions = None
        self.high_level_plan = None
        self.world_representation = None

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        # /raid/home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml

        """State machine"""

        if self.timesteps[0] == 0:
            self._init_episode(obs)

        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            obs = self.semantic_sensor(obs)
        else:
            obs.task_observations["semantic_frame"] = None
        info = self._get_info(obs)

        self.timesteps[0] += 1

        # is_finished = False
        action = None

        while action is None:
            if self.states[0] == Skill.EXPLORE:
                obs.task_observations["instance_id"] = 100000000000
                action, info, new_state = self._explore(obs, info)
            else:
                raise ValueError

        goal_object_name = info["goal_name"].split(" ")[1]

        global labels_counter
        # extract_labels(
        #     obs.semantic,
        #     obs.rgb,
        #     "/raid/home-robot/gyzp/data/receptacle/val/labels/" + str(labels_counter) + ".txt",
        #     "/raid/home-robot/gyzp/data/receptacle/val/images/" + str(labels_counter) + ".png",
        #     "/raid/home-robot/gyzp/data/receptacle/val/marked/"
        # )
        extract_goal_object(
            obs.semantic,
            obs.rgb,
            "/raid/home-robot/gyzp/data/goal/val/labels/" + str(labels_counter) + "-" + goal_object_name + ".txt",
            "/raid/home-robot/gyzp/data/goal/val/images/" + str(labels_counter) + "-" + goal_object_name + ".png",
            "/raid/home-robot/gyzp/data/goal/val/marked/",
            goal_object_name,
            self.labels_dict.get(goal_object_name),
            "/raid/home-robot/gyzp/data/goal/val/pixel/" + str(labels_counter) + ".txt",
        )
        labels_counter += 1

        # update the curr skill to the new skill whose action will be executed
        info["curr_skill"] = Skill(self.states[0].item()).name
        # if self.verbose:
        print(f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}')
        if self.args.force_step == self.timesteps[0]:
            print(
                "Max agent step reached, forcing the agent to quit and wrapping up..."
            )
            action = DiscreteNavigationAction.STOP

        return action, info, obs
