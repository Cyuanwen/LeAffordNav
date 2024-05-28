#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-
# quick fix for import

'''
为了调试单独的放置任务的代码，稍微修改能跑通，但基本弃用
'''

from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple
from home_robot.core.abstract_agent import Agent

import torch

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPlacePolicy_OnlyPlace


# @cyw
import cv2
show_image = True

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


class PlaceAgent(Agent):
    def __init__(
        self, config, device_id: int = 0, obs_spaces=None, action_spaces=None, args=None
    ):
        super().__init__()
        print("Exploration Agent created")
        self.config = config
        self.device_id = device_id
        self.args = args
        self.skip_skills = config.AGENT.skip_skills
        if config.AGENT.SKILLS.PLACE.type == "heuristic" and not self.skip_skills.place:
            self.place_policy = HeuristicPlacePolicy_OnlyPlace(
                config, self.device, verbose=self.verbose
            )
        elif config.AGENT.SKILLS.PLACE.type == "rl" and not self.skip_skills.place:
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.place_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.PLACE,
                device_id=device_id,
            )
    
    def reset(self):
        self.timesteps = 0
        self.place_done = 0
    
    def _hardcoded_place(self):
        """Hardcoded place skill execution
        Orients the agent's arm and camera towards the recetacle, extends arm and releases the object
        """
        place_step = self.timesteps
        forward_steps = 0
        if place_step < forward_steps:
            # for experimentation (TODO: Remove. ideally nav should drop us close)
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step == forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif place_step == forward_steps + 1:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif place_step == forward_steps + 2:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= forward_steps + 3:
            # allow the object to come to rest
            action = DiscreteNavigationAction.STOP
        else:
            raise ValueError(
                f"Something is wrong. Episode should have ended. Place step: {place_step}, Timestep: {self.timesteps}"
            )
        return action
    
    def _rl_place(self, obs: Observations, info: Dict[str, Any]):
        place_step = self.timesteps
        if place_step == 0:
            action = DiscreteNavigationAction.POST_NAV_MODE
        elif self.place_done == 1:
            action = DiscreteNavigationAction.STOP
            self.place_done = 0
        else:
            action, info, terminate = self.place_agent.act(obs, info)
            if terminate:
                action = DiscreteNavigationAction.DESNAP_OBJECT
                self.place_done = 1
        return action, info
    
    def _place(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        place_type = self.config.AGENT.SKILLS.PLACE.type
        if self.skip_skills.place:
            action = DiscreteNavigationAction.STOP
        elif place_type == "hardcoded":
            action = self._hardcoded_place()
        elif place_type == "heuristic":
            action, info = self.place_policy(obs, info)
        elif place_type == "rl":
            action, info = self._rl_place(obs, info)
        else:
            raise ValueError(f"Got unexpected value for PLACE.type: {place_type}")
        new_state = None
        # if action == DiscreteNavigationAction.STOP:
        #     action = None
        #     new_state = Skill.FALL_WAIT
        return action, info, new_state

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """State machine"""

        self.timesteps += 1
        info = self._get_info(obs)

        # is_finished = False
        action = None
        if show_image:
            cv2.imshow("rgb",cv2.cvtColor(obs.rgb,cv2.COLOR_BGR2RGB))
            cv2.imshow("third_rgb",cv2.cvtColor(obs.third_person_image,cv2.COLOR_BGR2RGB))
            # semantic_img = get_semantic_vis(obs.semantic)
            # cv2.imshow("semantic",semantic_img)
            cv2.waitKey()
            
        while action is None:
            print(f"step: {self.timesteps} -- place")
            action, info, new_state = self._place(obs, info)
            print(f"action: {action}")

        return action, info, obs
