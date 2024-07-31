# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh.transformations as tra

import home_robot.utils.depth as du
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.motion.stretch import STRETCH_STANDOFF_DISTANCE
from home_robot.utils.image import smooth_mask
from home_robot.utils.rotation import get_angle_to_pos

RETRACTED_ARM_APPROX_LENGTH = 0.15
HARDCODED_ARM_EXTENSION_OFFSET = 0.15
HARDCODED_YAW_OFFSET = 0.25


class HeuristicPlacePolicy(nn.Module):
    """
    Policy to place object on end receptacle using depth and point-cloud-based heuristics. Objects will be placed nearby, on top of the surface, based on point cloud data. Requires segmentation to work properly.
    """

    # TODO: read these values from the robot kinematic model
    # look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    look_at_ee = np.array([-np.pi / 2, -np.pi / 8])
    max_arm_height = 1.2

    def __init__(
        self,
        config,
        device,
        placement_drop_distance: float = 0.4,
        debug_visualize_xyz: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters:
            config
            device
            placement_drop_distance: distance from placement point that we add as a margin
            debug_visualize_xyz: whether to display point clouds for debugging
            verbose: whether to print debug statements
        """
        super().__init__()
        self.timestep = 0
        self.config = config
        self.device = device
        self.debug_visualize_xyz = debug_visualize_xyz
        self.erosion_kernel = np.ones((5, 5), np.uint8)
        self.placement_drop_distance = placement_drop_distance
        self.verbose = verbose

    def reset(self):
        self.timestep = 0 #重置时间步长


    #将目标点云转换为基础坐标系。
    #通过摄像机深度图和相机矩阵获取点云数据，并转换为机器人基础坐标。
    def get_target_point_cloud_base_coords(
        self,
        obs: Observations,
        target_mask: np.ndarray,
        arm_reachability_check: bool = False,
    ):
        """Get point cloud coordinates in base frame"""
         # 将深度图转换为张量并添加一个维度
        goal_rec_depth = torch.tensor( 
            obs.depth, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        # 获取相机矩阵
        camera_matrix = du.get_camera_matrix(
            self.config.ENVIRONMENT.frame_width,
            self.config.ENVIRONMENT.frame_height,
            self.config.ENVIRONMENT.hfov,
        )
        # Get object point cloud in camera coordinates
        pcd_camera_coords = du.get_point_cloud_from_z_t(
            goal_rec_depth, camera_matrix, self.device, scale=self.du_scale
        )
         # 获取基础坐标系中的点云
        # get point cloud in base coordinates
        camera_pose = np.expand_dims(obs.camera_pose, 0)
        angles = [tra.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
        tilt = angles[0][1]  # [0][1]

        # Agent height comes from the environment config
        agent_height = torch.tensor(camera_pose[0, 2, 3], device=self.device)

        # Object point cloud in base coordinates，将点云从相机坐标系转换到基础坐标系
        pcd_base_coords = du.transform_camera_view_t(
            pcd_camera_coords, agent_height, np.rad2deg(tilt), self.device
        )
        # 如果需要调试可视化，显示点云
        if self.debug_visualize_xyz:
            # Remove invalid points from the mask，# 从maks中移除无效点
            xyz = (
                pcd_base_coords[0]
                .cpu()
                .numpy()
                .reshape(-1, 3)[target_mask.reshape(-1), :]
            )
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = (obs.rgb).reshape(-1, 3) / 255.0
            show_point_cloud(xyz, rgb, orig=np.zeros(3))
            # show_point_cloud(xyz, rgb, orig=np.zeros(3),save="cyw/test_data/point_cloud")

        # 检查手臂是否能够达到这些点
        # Whether or not I can extend the robot's arm in order to reach each point
        if arm_reachability_check:
            # 根据体素的Y和Z坐标过滤掉不可达的点（Z是向上）
            # filtering out unreachable points based on Y and Z coordinates of voxels (Z is up)
            height_reachable_mask = (pcd_base_coords[0, :, :, 2] < agent_height).to(int)
            height_reachable_mask = torch.stack([height_reachable_mask] * 3, axis=-1)
            pcd_base_coords = pcd_base_coords * height_reachable_mask

            length_reachable_mask = (pcd_base_coords[0, :, :, 1] < agent_height).to(int)
            length_reachable_mask = torch.stack([length_reachable_mask] * 3, axis=-1)
            pcd_base_coords = pcd_base_coords * length_reachable_mask
        # 根据目标掩码过滤点云
        non_zero_mask = torch.stack(
            [torch.from_numpy(target_mask).to(self.device)] * 3, axis=-1
        )
        pcd_base_coords = pcd_base_coords * non_zero_mask

        return pcd_base_coords[0] # 返回基础坐标系中的点云

    #计算物体放置的3D空间位置。
    #使用随机采样和邻域搜索找到最佳放置点
    def get_receptacle_placement_point(
        self,
        obs: Observations,
        vis_inputs: Optional[Dict] = None,
        arm_reachability_check: bool = False,
        visualize: bool = False,
    ):
        """
        Compute placement point in 3d space.

        Parameters:
            obs: Observation object; describes what we've seen.
            vis_inputs: optional dict; data used for visualizing outputs
        """
        NUM_POINTS_TO_SAMPLE = 50  # number of points to sample from receptacle point cloud to find best placement point
        SLAB_PADDING = 0.2  # x/y padding around randomly selected points
        SLAB_HEIGHT_THRESHOLD = 0.015  # 1cm above and below, i.e. 2cm overall
        ALPHA_VIS = 0.5 # 可视化时的透明度

        #获取目标接收器的掩码，使用语义标签和有效深度掩码
        goal_rec_mask = (
            obs.semantic
            == obs.task_observations["end_recep_goal"] * du.valid_depth_mask(obs.depth)
        ).astype(np.uint8)
        # Get dilated, then eroded mask (for cleanliness)，使用腐蚀和膨胀操作清理掩码
        goal_rec_mask = smooth_mask(
            goal_rec_mask, self.erosion_kernel, num_iterations=5
        )[1]
        # Convert to booleans
        goal_rec_mask = goal_rec_mask.astype(bool)

        if visualize:
            cv2.imwrite(f"{self.end_receptacle}_semantic.png", goal_rec_mask * 255)

        # 如果目标接收器掩码中没有任何有效像素，则打印调试信息并返回None
        if not goal_rec_mask.any():
            if self.verbose:
                print("End receptacle not visible.")
            return None
        else:
            rgb_vis = obs.rgb # 获取RGB图像用于后续可视化
            pcd_base_coords = self.get_target_point_cloud_base_coords(
                obs, goal_rec_mask, arm_reachability_check=arm_reachability_check
            ) # 获取基础坐标系中的点云坐标

            ## 随机采样接收器点云中的点以找到最佳放置点
            ## randomly sampling NUM_POINTS_TO_SAMPLE of receptacle point cloud – to choose for placement
            reachable_point_cloud = pcd_base_coords.cpu().numpy()
            flat_array = reachable_point_cloud.reshape(-1, 3)

            # 找到矩阵前两个维度中非零元素的索引
            # find the indices of the non-zero elements in the first two dimensions of the matrix
            nonzero_indices = np.nonzero(flat_array[:, :2].any(axis=1))[0] #怎么会这么多元素为0？ 
            # 创建包含非零索引的元组列表
            # create a list of tuples containing the non-zero indices in the first two dimensions
            nonzero_tuples = [
                (                                                         
                    #.shape返回点云的形状。.shape[-2]点云在第二维度的大小，即点云的高度。              
                    index // reachable_point_cloud.shape[-2], #取整，得行索引
                    index % reachable_point_cloud.shape[-2],  #取余，得列索引
                )
                for index in nonzero_indices
            ]
            # 随机选择非零索引的子集
            # select a random subset of the non-zero indices
            random_indices = random.sample(
                nonzero_tuples, min(NUM_POINTS_TO_SAMPLE, len(nonzero_tuples))
            )# 随机选择采样点

            x_values = pcd_base_coords[:, :, 0]  # 获取点云的X坐标，：：获取所有的高度和宽度
            y_values = pcd_base_coords[:, :, 1]  # 获取点云的Y坐标
            z_values = pcd_base_coords[:, :, 2]  # 获取点云的Z坐标

            max_surface_points = 0 # 初始化最大表面点数
            # max_height = 0

            # 初始化最佳体素相关变量
            max_surface_mask, best_voxel_ind, best_voxel = None, None, None

            ## iterating through all randomly selected voxels and choosing one with most XY neighboring surface area within some height threshold
            # 遍历所有随机选择的体素并选择 高度阈值内具有最大XY相邻表面积最大的体素作为最佳放置点
            for ind in random_indices:
                sampled_voxel = pcd_base_coords[ind[0], ind[1]] # 获取随机选择的体素,ind[0]点云高度,ind[1]点云宽度
                sampled_voxel_x, sampled_voxel_y, sampled_voxel_z = (
                    sampled_voxel[0], # 体素的X坐标
                    sampled_voxel[1], # 体素的Y坐标
                    sampled_voxel[2], # 体素的Z坐标
                )

                # 采样随机选择的体素周围的点云平面（高度容差范围内）
                # sampling plane of pcd voxels around randomly selected voxel (with height tolerance)
                slab_points_mask_x = torch.bitwise_and(
                    (x_values >= sampled_voxel_x - SLAB_PADDING),
                    (x_values <= sampled_voxel_x + SLAB_PADDING),
                )
                slab_points_mask_y = torch.bitwise_and(
                    (y_values >= sampled_voxel_y - SLAB_PADDING),
                    (y_values <= sampled_voxel_y + SLAB_PADDING),
                )
                slab_points_mask_z = torch.bitwise_and(
                    (z_values >= sampled_voxel_z - SLAB_HEIGHT_THRESHOLD),
                    (z_values <= sampled_voxel_z + SLAB_HEIGHT_THRESHOLD),
                )


                #生成平面掩码
                slab_points_mask = torch.bitwise_and(
                    slab_points_mask_x, slab_points_mask_y
                ).to(torch.uint8)  # 结合X和Y坐标的掩码
                slab_points_mask = torch.bitwise_and(
                    slab_points_mask, slab_points_mask_z
                ).to(torch.uint8) # 结合Z坐标的掩码

                # ALTERNATIVE: choose slab with maximum (area x height) product
                # TODO: remove dead code
                # slab_points_mask_stacked = torch.stack(
                #     [
                #         slab_points_mask * 255,
                #         slab_points_mask,
                #         slab_points_mask,
                #     ],
                #     axis=-1,
                # )
                # height = (slab_points_mask_stacked * pcd_base_coords)[..., 2].max()
                # if slab_points_mask.sum() * height >= max_surface_points * max_height:
                
                # 选择具有最大XY邻域表面积的平面
                if slab_points_mask.sum() >= max_surface_points:
                    max_surface_points = slab_points_mask.sum()
                    max_surface_mask = slab_points_mask
                    # max_height = height
                    best_voxel_ind = ind
                    best_voxel = sampled_voxel #确实是随机选的点，可是为什么每次都会选到边缘上？

            # 用于可视化的平面点掩码
            slab_points_mask_vis = torch.stack(
                [
                    max_surface_mask * 255, # 扩展掩码用于可视化
                    max_surface_mask,
                    max_surface_mask,
                ],
                axis=-1,
            )  # for visualization
            rgb_vis_tmp = cv2.addWeighted(
                rgb_vis, ALPHA_VIS, slab_points_mask_vis.cpu().numpy(), 1 - ALPHA_VIS, 0
            )# 合并RGB图像和掩码进行可视化

            rgb_vis_tmp = cv2.circle(
                rgb_vis_tmp,
                (best_voxel_ind[1], best_voxel_ind[0]),# 在最佳体素位置绘制圆圈
                4,
                (0, 255, 0),#绿色
                thickness=2,
            )

            # 如果有可视化输入，更新其语义帧
            if vis_inputs is not None and vis_inputs["semantic_frame"] is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis_tmp
            
            # 在最佳体素上添加放置边距
            # Add placement margin to the best voxel that we chose
            best_voxel[2] += self.placement_drop_distance # 加了一个高度？！

            # 如果需要调试可视化，显示点云
            if self.debug_visualize_xyz:
                from home_robot.utils.point_cloud import show_point_cloud

                show_point_cloud(
                    pcd_base_coords.cpu().numpy(),
                    rgb=obs.rgb / 255.0,
                    orig=best_voxel.cpu().numpy(),
                )
                # show_point_cloud(
                #     pcd_base_coords.cpu().numpy(),
                #     rgb=obs.rgb / 255.0,
                #     orig=best_voxel.cpu().numpy(),
                #     save="cyw/test_data/point_cloud"
                # )

            return best_voxel.cpu().numpy(), vis_inputs

    #实现了放置物体的动作序列，包括转向、前进、抬起手臂、放置物体等步骤。
    def forward(self, obs: Observations, vis_inputs: Optional[Dict] = None):
        """
        1. Get estimate of point on receptacle to place object on.
        2. Orient towards it.
        3. Move forward to get close to it.
        4. Rotate 90º to have arm face the object. Then rotate camera to face arm.
        5. (again) Get estimate of point on receptacle to place object on.
        6. With camera, arm, and object (hopefully) aligned, set arm lift and
        extension based on point estimate from 4.

        Returns:
            action: what the robot will do - a hybrid action, discrete or continuous
            vis_inputs: dictionary containing extra info for visualizations
        """
        """
        实现物体放置的主要逻辑流程。
        1. 获取接收器上的放置点估计。
        2. 朝向放置点方向调整。
        3. 向前移动靠近放置点。
        4. 转动90度使手臂对准物体，再转动摄像头对准手臂。
        5. 再次获取接收器上的放置点估计。
        6. 将摄像头、手臂和物体对准后，基于第4步的估计设置手臂的提升和扩展。

        返回：
            action: 机器人将执行的动作，可以是离散或连续动作
            vis_inputs: 包含用于可视化的额外信息的字典
        """

        # 获取环境配置中的转角度和前进步长
        turn_angle = self.config.ENVIRONMENT.turn_angle
        fwd_step_size = self.config.ENVIRONMENT.forward

        if self.timestep == 0:
            self.du_scale = 1  # TODO: working with full resolution for now
            self.end_receptacle = obs.task_observations["goal_name"].split(" ")[-1]
            found = self.get_receptacle_placement_point(obs, vis_inputs)
            # found = self.get_receptacle_placement_point(obs, vis_inputs,visualize=True)

            if found is None:
                if self.verbose:
                    print("Receptacle not visible. Execute hardcoded place.")
                self.total_turn_and_forward_steps = 0
                self.initial_orient_num_turns = -1
                self.fall_wait_steps = 0
                self.t_go_to_top = 1
                self.t_extend_arm = 2
                self.t_release_object = 3
                self.t_lift_arm = 4
                self.t_retract_arm = 5
                self.t_go_to_place = -1
                self.t_done_waiting = 5 + self.fall_wait_steps
            else:
                self.placement_voxel, vis_inputs = found

                center_voxel_trans = np.array(
                    [
                        self.placement_voxel[1],
                        self.placement_voxel[2],
                        self.placement_voxel[0],
                    ]
                )

                delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans)) # 头部需要转的角度,根据当前(x,y)和pos(x,y)计算

                self.initial_orient_num_turns = abs(delta_heading) // turn_angle
                self.orient_turn_direction = np.sign(delta_heading)
                # This gets the Y-coordiante of the center voxel
                # Base link to retracted arm - this is about 15 cm
                fwd_dist = (
                    self.placement_voxel[1]
                    - STRETCH_STANDOFF_DISTANCE
                    - RETRACTED_ARM_APPROX_LENGTH
                )

                fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
                self.forward_steps = fwd_dist // fwd_step_size
                self.total_turn_and_forward_steps = (
                    self.forward_steps + self.initial_orient_num_turns
                )
                self.fall_wait_steps = 0
                self.t_go_to_top = self.total_turn_and_forward_steps + 1
                self.t_go_to_place = self.total_turn_and_forward_steps + 2
                self.t_release_object = self.total_turn_and_forward_steps + 3
                self.t_lift_arm = self.total_turn_and_forward_steps + 4
                self.t_retract_arm = self.total_turn_and_forward_steps + 5
                self.t_extend_arm = -1
                self.t_done_waiting = (
                    self.total_turn_and_forward_steps + 5 + self.fall_wait_steps
                )
                if self.verbose:
                    print("-" * 20)
                    print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
                    print(f"Move forward for {self.forward_steps} steps.")

        if self.verbose:
            print("-" * 20)
            print("Timestep", self.timestep)
        if self.timestep < self.initial_orient_num_turns:
            if self.orient_turn_direction == -1:
                action = DiscreteNavigationAction.TURN_RIGHT
            if self.orient_turn_direction == +1:
                action = DiscreteNavigationAction.TURN_LEFT
            if self.verbose:
                print("[Placement] Turning to orient towards object")
        elif self.timestep < self.total_turn_and_forward_steps:
            if self.verbose:
                print("[Placement] Moving forward")
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif self.timestep == self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif self.timestep == self.t_go_to_top:
            # We should move the arm back and retract it to make sure it does not hit anything as it moves towards the target position
            action = self._retract(obs)
        elif self.timestep == self.t_go_to_place:
            if self.verbose:
                print("[Placement] Move arm into position")
            placement_height, placement_extension = (
                self.placement_voxel[2],
                self.placement_voxel[1],
            )

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift
            # # @cyw
            # delta_arm_lift = delta_arm_lift-0.2

            current_arm_ext = obs.joint[:4].sum()
            delta_arm_ext = (
                placement_extension
                - STRETCH_STANDOFF_DISTANCE
                - RETRACTED_ARM_APPROX_LENGTH
                - current_arm_ext
                + HARDCODED_ARM_EXTENSION_OFFSET
            )
            # # # # @cyw
            # delta_arm_ext = (
            #     placement_extension
            #     # - STRETCH_STANDOFF_DISTANCE
            #     - RETRACTED_ARM_APPROX_LENGTH
            #     - current_arm_ext
            #     + HARDCODED_ARM_EXTENSION_OFFSET
            # )
            # # 修改 extension 和 lift后,能够放上去,但是会发生碰撞,所以根本上还是这个点计算得不对?
            center_voxel_trans = np.array(
                [
                    self.placement_voxel[1],
                    self.placement_voxel[2],
                    self.placement_voxel[0],
                ]
            )
            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            delta_gripper_yaw = delta_heading / 90 - HARDCODED_YAW_OFFSET

            if self.verbose:
                print("[Placement] Delta arm extension:", delta_arm_ext)
                print("[Placement] Delta arm lift:", delta_arm_lift)
            joints = np.array(
                [delta_arm_ext]
                + [0] * 3
                + [delta_arm_lift]
                + [delta_gripper_yaw]
                + [0] * 4
            )
            # @cyw
            joints = self._look_at_ee(joints)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_release_object:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif self.timestep == self.t_lift_arm:
            action = self._lift(obs)
        elif self.timestep == self.t_retract_arm:
            action = self._retract(obs)
        elif self.timestep == self.t_extend_arm:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif self.timestep <= self.t_done_waiting:
            if self.verbose:
                print("[Placement] Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            if self.verbose:
                print("[Placement] Stopping")
            action = DiscreteNavigationAction.STOP

        debug_texts = {
            self.total_turn_and_forward_steps: "[Placement] Aligning camera to arm",
            self.t_go_to_top: "[Placement] Raising the arm before placement.",
            self.t_go_to_place: "[Placement] Move arm into position",
            self.t_release_object: "[Placement] Desnapping object",
            self.t_lift_arm: "[Placement] Lifting the arm after placement.",
            self.t_retract_arm: "[Placement] Retracting the arm after placement.",
            self.t_extend_arm: "[Placement] Extending the arm out for placing.",
            self.t_done_waiting: "[Placement] Empty action",
        }
        if self.verbose and self.timestep in debug_texts:
            print(debug_texts[self.timestep])

        self.timestep += 1
        return action, vis_inputs

    #_lift、_look_at_ee、_retract等方法用于控制机器人的手臂位置和运动。
    def _lift(self, obs: Observations) -> ContinuousFullBodyAction:
        """Compute a high-up lift position to avoid collisions when releasing"""
        # Hab sim dimensionality for arm == 10
        joints = np.zeros(10)
        # We take the lift position = 1
        current_arm_lift = obs.joint[4]
        # Target lift is 0.99
        lift_delta = self.max_arm_height - current_arm_lift
        joints[4] = lift_delta
        # @cyw
        joints = self._look_at_ee(joints)
        action = ContinuousFullBodyAction(joints)
        return action

    def _look_at_ee(self, joints: np.ndarray) -> np.ndarray:
        """Make sure it's actually looking at the end effector."""
        joints[8] = self.look_at_ee[0] # WRIST_YAW 偏航
        joints[9] = self.look_at_ee[1] # HEAD_PAN 按理说修改的是平移,但是仿真上看到的却是上下旋转
        # NOTE 这个似乎不太管用,place策略把它取消了
        return joints

    def _retract(self, obs: Observations) -> ContinuousFullBodyAction:
        """Compute a high-up retracted position to avoid collisions"""
        # Hab sim dimensionality for arm == 10
        joints = np.zeros(10)
        # We take the lift position = 1
        current_arm_lift = obs.joint[4]
        # Target lift is 0.99
        lift_delta = self.max_arm_height - current_arm_lift
        # @cyw
        # lift_delta = self.max_arm_height - current_arm_lift - 0.2 # 最大高度挡住机器人视线,剪掉一部分还是挡住的,这东西不好弄
        # Arm should be fully retracted
        arm_delta = -1 * np.sum(obs.joint[:4])
        joints[0] = arm_delta
        joints[4] = lift_delta
        # @cyw
        joints = self._look_at_ee(joints)
        action = ContinuousFullBodyAction(joints)
        return action
