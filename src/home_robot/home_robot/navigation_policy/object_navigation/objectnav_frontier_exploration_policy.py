# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from array import array
import enum
from optparse import Option
from typing import Dict, Optional
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation, binary_erosion

# @cyw
import cv2
from home_robot.navigation_planner.fmm_planner import FMMPlanner
import sys
import os
sys.path.append(os.getcwd())
from cyw.test.psl_agent import psl_agent
from cyw.test.psl_agent import recep_category_21, rooms


class ObjectNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_sem_categories: int,
        psl_config: str,
        explored_area_dilation_radius=10,
        obs_dilation_selem_radius: int=3,
        esc_frontier: bool=False
    ):
        # @cyw add obs_dilation_selem_radius, psl_config
        '''
        psl_config: psl agent config
        obs_dilation_selem_radius: radius (in cells) of obstacle dilation
                structuring element
        esc_frontier: 是否使用esc的方式选择frontier
        '''
        super().__init__()
        assert exploration_strategy in ["seen_frontier", "been_close_to_frontier"]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(explored_area_dilation_radius))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.num_sem_categories = num_sem_categories

        # @cyw
        self.esc_frontier = esc_frontier
        if self.esc_frontier:
            self.obs_dilation_selem_radius = obs_dilation_selem_radius
            self.obs_dilation_selem = skimage.morphology.disk(
                self.obs_dilation_selem_radius
            )
            self.psl_config = psl_config
            self.psl_frontier_agent = psl_agent(self.psl_config)
        self.time_step = 0 # 可视化保存图像用

    @property
    def goal_update_steps(self):
        return 1
    
    # @cyw
    def reset(self, vocab:Optional[Dict], gt_seg:bool=False):
        if self.esc_frontier:
            self.psl_frontier_agent.set_vocab(vocab, gt_seg)
            

    def reach_single_category(self, map_features, category, states):
        # if the goal is found, reach it
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, category)
        # otherwise, do frontier exploration
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal, states, category)
        return goal_map, found_goal

    def reach_object_recep_combination(
        self, map_features, object_category, recep_category, states, 
    ):
        # First check if object (small goal) and recep category are in the same cell of the map. if found, set it as a goal
        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            small_goal_category=object_category,
        )
        # Then check if the recep category exists in the map. if found, set it as a goal
        goal_map, found_rec_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            reject_visited_regions=True,
            goal_map=goal_map,
            found_goal=found_goal,
        )
        # Otherwise, set closest frontier as the goal
        goal_map = self.explore_otherwise(map_features, goal_map, found_rec_goal, states, recep_category)
        return goal_map, found_goal

    # @cyw modified
    def forward(
        self,
        map_features,
        object_category=None,
        start_recep_category=None,
        end_recep_category=None,
        instance_id=None,
        nav_to_recep=None,
        states = None,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
            start_recep_category: start receptacle category
            end_recep_category: end receptacle category
            nav_to_recep: If both object_category and recep_category are specified, whether to navigate to receptacle
            # @cyw
            states: the local position of agent, local map 坐标系下
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert (
            object_category is not None
            or end_recep_category is not None
            or instance_id is not None
        )
        if instance_id is not None:
            instance_map = map_features[0][
                2 * MC.NON_SEM_CHANNELS
                + self.num_sem_categories : 2 * MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
                :,
                :,
            ]
            if len(instance_map) != 0:
                inst_map_idx = instance_map == instance_id
                inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                goal_map = (
                    (instance_map[inst_map_idx] == instance_id)
                    .to(torch.float)
                    .unsqueeze(0)
                )
                if torch.sum(goal_map) == 0:
                    found_goal = torch.tensor([0])
                else:
                    found_goal = torch.tensor([1])
            else:
                # try to navigate to instance without an instance map -- explore
                # create an empty goal map
                batch_size, _, height, width = map_features.shape
                device = map_features.device
                goal_map = torch.zeros((batch_size, height, width), device=device)
                found_goal = torch.tensor([0])

            # NOTE: record instance时，没有用esc 
            goal_map = self.explore_otherwise(map_features, goal_map, found_goal, states, None)
            return goal_map, found_goal

        elif object_category is not None and start_recep_category is not None:
            if nav_to_recep is None or end_recep_category is None:
                nav_to_recep = torch.tensor([0] * map_features.shape[0])

            # there is at least one instance in the batch where the goal is object
            if nav_to_recep.sum() < map_features.shape[0]:
                goal_map_o, found_goal_o = self.reach_object_recep_combination(
                    map_features, object_category, start_recep_category, states
                )
            # there is at least one instance in the batch where the goal is receptacle
            elif nav_to_recep.sum() > 0:
                goal_map_r, found_goal_r = self.reach_single_category(
                    map_features, end_recep_category, states
                )
            # some instances in batch may be navigating to objects (before pick skill) and some may be navigating to recep (before place skill)
            if nav_to_recep.sum() == 0:
                return goal_map_o, found_goal_o
            elif nav_to_recep.sum() == map_features.shape[0]:
                return goal_map_r, found_goal_r
            else:
                goal_map = (
                    goal_map_o * nav_to_recep.view(-1, 1, 1)
                    + (1 - nav_to_recep).view(-1, 1, 1) * goal_map_o
                )
                found_goal = (
                    found_goal_r * nav_to_recep + (1 - nav_to_recep) * found_goal_r
                )
                return goal_map, found_goal
        else:
            # Here, the goal is specified by a single object or receptacle to navigate to with no additional constraints (eg. the given object can be on any receptacle)
            goal_category = (
                object_category if object_category is not None else end_recep_category
            )
            return self.reach_single_category(map_features, goal_category, states)

    def cluster_filtering(self, m):
        # m is a 480x480 goal map
        if not m.any():
            return m
        device = m.device

        # cluster goal points
        k = DBSCAN(eps=4, min_samples=1)
        m = m.cpu().numpy()
        data = np.array(m.nonzero()).T
        k.fit(data)

        # mask all points not in the largest cluster
        mode = scipy.stats.mode(k.labels_, keepdims=True).mode.item()
        mode_mask = (k.labels_ != mode).nonzero()
        x = data[mode_mask]

        m_filtered = np.copy(m)
        m_filtered[x] = 0.0
        m_filtered = torch.tensor(m_filtered, device=device)

        return m_filtered

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        small_goal_category=None,
        reject_visited_regions=False,
        goal_map=None,
        found_goal=None,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        if goal_map is None and found_goal is None:
            goal_map = torch.zeros((batch_size, height, width), device=device)
            found_goal_current = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else:
            # crate a fresh map
            found_goal_current = torch.clone(found_goal)
        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]
                if small_goal_category is not None:
                    # additionally check if the category has the required small object on it
                    category_map = (
                        category_map
                        * map_features[
                            e, small_goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                        ]
                    )
                if reject_visited_regions:
                    # remove the receptacles that the already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BEEN_CLOSE_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()
        else:
            raise Exception("not implemented")

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )

        return frontier_map

    # @cyw modified
    def explore_otherwise(self, map_features, goal_map, found_goal, states = None, recep_categories = None):
        '''
        states: 当前位置
        recep_category: 要找的容器类别 idx
        '''
        """Explore closest unexplored region otherwise."""
        frontier_map = self.get_frontier_map(map_features)
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                if not self.esc_frontier:
                    goal_map[e] = frontier_map[e]
                else:
                    esc_frontier_map = self.get_esc_frontier_map(
                        map_feature = map_features[e],
                        frontier_map = frontier_map[e],
                        state = states[e],
                        recep_category = recep_categories[e],
                        near_room_range = self.psl_config.near_room_range,
                        near_obj_range = self.psl_config.near_obj_range,
                    )
                    if esc_frontier_map is not None:
                        goal_map[e] = esc_frontier_map
                    else:
                        goal_map[e] = frontier_map[e]
        return goal_map
    
    
    def get_esc_frontier_map(self, map_feature, frontier_map, state, recep_category, debug = False, save_image = False, near_room_range = 12, near_obj_range = 16):
        '''
        计算概率软逻辑的前端点地图，图中只有一个点的值为1，该点即为目标点，作用与原本代码的 get_frontier_map 类似
        : map_feature: 单个样本的语义图
        : frontier_map: 单个样本的frontier_map [seq,wide,height]
        : recep_category: 单个样本的目标类别，限定于recep类物体, id
        : state: 当前位置（转换为语义图坐标及度量标度下）
        : near_room_range: 多大范围内认为是 near room
        : near_obj_range: 多大范围内认为是 near object
        return: psl_frontier_map or None 
        不出意外 map_feature 的各层代表的意思：local_non_sem, global_non_sem, ., goal_object, recep, room, other
        '''
        self.psl_frontier_agent.set_target(int(recep_category))
        frontier_locations = torch.stack([torch.where(frontier_map[0])[0], torch.where(frontier_map[0])[1]]).T
        frontier_locations = frontier_locations.cpu().numpy()
        num_frontiers = len(torch.where(frontier_map[0])[0])
        if num_frontiers == 0:
            return None
        else:
            # for each frontier, calculate the distance
            traversible = self.get_traversible(map_feature)
            planner = FMMPlanner(traversible)
            planner.set_goal(state)
            fmm_dist = planner.fmm_dist
            # NOTE esc源代码计算了逆序，感觉不应该逆序？
            distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
            ## use the threshold of 1.6 to filter close frontiers to encourage exploration
            idx_16 = np.where(distances>=1.6)
            distances_16 = distances[idx_16]
            distances_16 = distances_16.T
            if len(distances_16) == 0:
                return None
            frontier_locations_16 = frontier_locations[idx_16]
            if self.psl_config.reasoning in ["both","room"]:
                # calculate the room frontier nearby matrix
                num_room = len(rooms)
                sem_channel = len(map_feature)
                room_map = map_feature[sem_channel-1-num_room:sem_channel-1,:,:]
                if debug:
                    pass
                near_room_frontier_mtx = self.get_near_frontier_mtx(
                    map = room_map.cpu().numpy(),
                    frontier_locations = frontier_locations_16,
                    near_range = near_room_range,
                )
            else:
                near_room_frontier_mtx = None
            
            if self.psl_config.reasoning in ["both", "object"]:
                # calculate the object frontier nearby matrix
                num_obj = len(recep_category_21)
                obj_map = map_feature[2 * MC.NON_SEM_CHANNELS + 2: 2 * MC.NON_SEM_CHANNELS + 2 + num_obj,:,:]
                if debug:
                    pass
                near_obj_frontier_mtx = self.get_near_frontier_mtx(
                    map = obj_map.cpu().numpy(),
                    frontier_locations = frontier_locations_16,
                    near_range = near_obj_range,
                )
            else:
                near_obj_frontier_mtx = None
            # PSL infer
            scores = self.psl_frontier_agent.infer(
                dist_frontier = distances_16,
                near_room_frontier = near_room_frontier_mtx,
                near_object_frontier = near_obj_frontier_mtx
            )
            idx_frontier = np.argmax(scores)
            loc_frontier = frontier_locations_16[idx_frontier]
            psl_frontier_map = torch.zeros_like(frontier_map)
            psl_frontier_map[0, loc_frontier[0], loc_frontier[1]] = 1

            if debug or save_image:
                # 可视化frontier_map, frontier_dist, frontier_score
                # 有必要的话，可视化一下obj_map, room_map, near_matrix，可参考/raid/cyw/nav_alg/multion-challenge/cyw/visualize/visualize.py
                from PIL import Image
                import matplotlib.pyplot as plt
                # depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
                frontier_image = Image.fromarray((frontier_map[0].cpu().numpy()*255).astype(np.uint8),mode="L")
                # frontier_image.show("frontier_image")
                traversible_image = Image.fromarray((traversible*255).astype(np.uint8),mode="L")
                # traversible_image.show("traversible_image")
                # fmm_dist_img = Image.fromarray(fmm_dist.astype(np.uint8),mode="L")
                # fmm_dist_img.show()
                frontier_dist = np.zeros_like(frontier_map[0].cpu().numpy())
                frontier_dist[tuple(zip(*frontier_locations_16))] = distances_16
                frontier_dist = grey_stretch(frontier_dist)
                frontier_dist_image = Image.fromarray(frontier_dist.astype(np.uint8),mode="L")
                # frontier_dist_image.show()
                frontier_score = np.zeros_like(frontier_map[0].cpu().numpy())
                frontier_score[tuple(zip(*frontier_locations_16))] = scores
                frontier_score = grey_stretch(frontier_score)
                # 当前位置
                frontier_score[state[0]-7:state[0]+8,state[1]-7:state[1]+8] = 255 
                # 最优分数位置
                frontier_score[loc_frontier[0]-3:loc_frontier[0]+3,loc_frontier[1]-3:loc_frontier[1]+3] = 255
                # 最小距离位置
                min_dist_idx = np.argmin(distances_16)
                min_dist_location = frontier_locations_16[min_dist_idx]
                frontier_score[min_dist_location[0]-3:min_dist_location[0]+3,min_dist_location[1]-3:min_dist_location[1]+3] = 127
                frontier_score_image = Image.fromarray(frontier_score.astype(np.uint8),mode="L")
                # frontier_score_image.show()
                arr = [frontier_image, traversible_image, frontier_dist_image, frontier_score_image]
                titles = ["frontier_image", "traversible_image", "frontier_dist_image", "frontier_score_image"]
                plt.figure(num="esc_frontier", figsize=(12,8))
                plt.clf()
                for i, data in enumerate(arr):
                    # 为和源代码中图像对应，将所有图像都翻转
                    data = np.flipud(data)
                    ax = plt.subplot(2, 2, i+1)
                    ax.axis('off')
                    ax.set_title(titles[i])
                    plt.imshow(data)
                if save_image:
                    plt.savefig(f"cyw/img/esc_frontier/{self.psl_config.PSL_infer}/{self.time_step}.jpg")
                    self.time_step += 1
                # plt.pause(1)
                plt.show() #NOTE 可视化窗口不会随着程序运行变化，只有当程序阻塞（暂停）时，才会更新图像
            return psl_frontier_map

    
    def get_near_frontier_mtx(self, map:array, frontier_locations, near_range):
        '''
        计算room或obj 与 frontier 的 nearby matrix
        : map : (num_channel(num_room or num_obj), w, h) room 或者 obj 的语义图层
        : frontier_locations: (num_frontier, 2) frontier的位置
        : near_range: 多大距离认为是附近，默认为12格，即 12*5 = 60 cm
        return: near_frontier_mtx (num_room or num_obj, num_frontier) room 或者 obj 与 frontier的共现矩阵
        # TODO 是否考虑使用卷积来计算？
        '''
        map_size = map.shape[1:]
        near_frontier_mtx = np.zeros((len(frontier_locations), len(map)))
        for i, loc in enumerate(frontier_locations):
            x_l, x_r, y_l, y_r = self.get_near_boundary(loc, map_size, near_range)
            sub_map = map[:,x_l:x_r, y_l:y_r] # select the room map around the frontier
            near_frontier_mtx[i] = np.max(np.max(sub_map, 1),1) # 1*9 wether the frontier is close to each room
        near_frontier_mtx = near_frontier_mtx.T
        return near_frontier_mtx


    def get_near_boundary(self, loc, map_size, near_range):
        '''
        : loc: (x,y) location， sem_map坐标系下
        : map_size: map_feature[0].shape 
        : near_range: 多大范围内认为是附近 默认12格
        '''
        x, y = loc[0], loc[1]
        x_l = np.clip(x - near_range, 0, map_size[0])
        x_r = np.clip(x + near_range, 0, map_size[0])
        y_l = np.clip(y - near_range, 0, map_size[1])
        y_r = np.clip(y + near_range, 0, map_size[1])
        
        return x_l, x_r, y_l, y_r

    def get_traversible(self, map_feature):
        '''
        计算 可行驶区域，参考：src/home_robot/home_robot/navigation_planner/discrete_planner.py _get_short_term_goal
        但是，这里的计算较为简单：
        1. 没有考虑 visited_map
        2. 没有考虑当前位置
        3. 没有对traversible进行 add_boundary

        :map_feature: sem_map建立的语义图，每次只计算一个样本
        '''
        obstacles = map_feature[MC.OBSTACLE_MAP, :, :].cpu().float().numpy()
        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        # Create inverse map of obstacles - this is territory we assume is traversible
        traversible = 1 - dilated_obstacles

        return traversible

# @cyw
def grey_stretch(gray_map:array):
    '''
    灰度图拉伸，将值映射到0-255之间，可视化用
    Input:
        gray_map: 灰度图 （wide,height）
    return:
        stretch_gray_map: 拉伸后的灰度图，值位于[0,255]之间
    '''
    min_value = gray_map.min()
    max_value = gray_map.max()
    gray_map_new = np.copy(gray_map)
    gray_map_new = (gray_map_new - min_value)/(max_value-min_value)*255
    return gray_map_new







