# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch.nn as nn

from home_robot.mapping.instance import InstanceMemory
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.navigation_policy.object_navigation.objectnav_frontier_exploration_policy import (
    ObjectNavFrontierExplorationPolicy,
)
# @cyw
import torch
import home_robot.utils.pose as pu

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False
# debug_frontier_map = True

# @cyw
debug = False


class ObjectNavAgentModule(nn.Module):
    def __init__(self, config, instance_memory: Optional[InstanceMemory] = None):
        super().__init__()
        self.instance_memory = instance_memory
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            instance_association=getattr(
                config.AGENT.SEMANTIC_MAP, "instance_association", "map_overlap"
            ),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            exploration_type=getattr(
                config.AGENT.SEMANTIC_MAP, "exploration_type", "default"
            ),
            gaze_width=getattr(config.AGENT.SEMANTIC_MAP, "gaze_width", 40),
            gaze_distance=getattr(config.AGENT.SEMANTIC_MAP, "gaze_distance", 1.5),
        )
        # @cyw
        self.esc_frontier = (config.AGENT.SKILLS.NAV_TO_REC.type == "heuristic_esc")
        self.policy = ObjectNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            explored_area_dilation_radius=getattr(
                config.AGENT.PLANNER, "explored_area_dilation_radius", 10
            ),
            psl_config=getattr(config.AGENT, "PSL_AGENT", None),
            obs_dilation_selem_radius=getattr(
                config.AGENT.PLANNER, "obs_dilation_selem_radius",3
            ),
            esc_frontier = self.esc_frontier,
        )
        self.map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps
    
    # @cyw
    def reset(self, vocab:Optional[dict], gt_seg:bool=False):
        self.policy.reset(vocab,gt_seg)

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        seq_start_recep_goal_category=None,
        seq_end_recep_goal_category=None,
        seq_instance_id=None,
        seq_nav_to_recep=None,
        semantic_max_val=None,
        seq_obstacle_locations=None,
        seq_free_locations=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, semantic_segmentation, instance_segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories + 1),
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, sequence_length, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)
            seq_object_goal_category: sequence of object goal categories of shape
             (batch_size, sequence_length, 1)
            seq_start_recep_goal_category: sequence of start recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_end_recep_goal_category: sequence of end recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_nav_to_recep: sequence of binary digits indicating if navigation is to object or end receptacle of shape
             (batch_size, 1)
        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
            semantic_max_val=semantic_max_val,
            seq_obstacle_locations=seq_obstacle_locations,
            seq_free_locations=seq_free_locations,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")
        if debug:
            # seq_obs: sequence of frames containing (RGB, depth, segmentation)
            #  of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
            #  frame_height, frame_width)(RGB, depth, semantic_segmentation, instance_segmentation)
            # seq_map_features: (batch_size, sequence_length, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            # import torch
            seg_cat = torch.unique(torch.nonzero(seq_obs[0,0,4:])[:,0])
            semMap_cat = torch.unique(torch.nonzero(seq_map_features[0,0,2*6:,:,:])[:,0])
            # torch.nonzero(seq_obs[0,0,4:]):获得每一个非0元素的索引,shape:[num_nonzero_point,dim]即：点的个数，点的坐标维数
            print(f"catogaries in semantic_segmentation:{seg_cat}")
            print(f"catoreries in seq_map_feature:{semMap_cat}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        map_features = seq_map_features.flatten(0, 1)
        if seq_object_goal_category is not None:
            seq_object_goal_category = seq_object_goal_category.flatten(0, 1)
        if seq_start_recep_goal_category is not None:
            seq_start_recep_goal_category = seq_start_recep_goal_category.flatten(0, 1)
        if seq_end_recep_goal_category is not None:
            seq_end_recep_goal_category = seq_end_recep_goal_category.flatten(0, 1)
        if seq_instance_id is not None:
            seq_instance_id = seq_instance_id.flatten(0, 1)
        # @cyw
        # 如果使用esc 导航，计算当前位置
        if self.esc_frontier:
            states = seq_local_pose[:, -1, 0:2] # seq_local_pose shape (batch, seq_length, 3) [:, -1]选取第二维的最后一个
            states = states * 100.0 / self.map_resolution
            states = states.to(torch.int)
            for i in range(len(states)):
                states[i] = pu.threshold_poses(states[i], seq_map_features.shape[-2:])
            # 参考 src/home_robot/home_robot/navigation_planner/discrete_planner.py 172行，将 x, y坐标翻转
            states = states[:,[1,0]]
        else:
            states = None
        
        # start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        # gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        # planning_window = [gx1, gx2, gy1, gy2]

        # start = [
        #     int(start_y * 100.0 / self.map_resolution - gx1),
        #     int(start_x * 100.0 / self.map_resolution - gy1),
        # ]
        # start = pu.threshold_poses(start, obstacle_map.shape)
        # start = np.array(start)
        
        # Compute the goal map
        goal_map, found_goal = self.policy(
            map_features,
            seq_object_goal_category,
            seq_start_recep_goal_category,
            seq_end_recep_goal_category,
            seq_instance_id,
            seq_nav_to_recep,
            states,
        )
        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])
        seq_found_goal = found_goal.view(batch_size, sequence_length)

        # Compute the frontier map here
        frontier_map = self.policy.get_frontier_map(map_features)
        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )
        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(seq_frontier_map[0, 0].numpy())
            plt.subplot(122)
            plt.imshow(goal_map[0].numpy())
            plt.show()
            breakpoint()
        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
