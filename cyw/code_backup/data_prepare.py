import cv2
import h5py
import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np

from habitat.utils.visualizations import maps
import home_robot.utils.visualization as vu

import sys
sys.path.append("/raid/home-robot")
sys.path.append("/raid/home-robot/projects")
from cyw.goal_point.utils import map_prepare, transform2relative,to_grid
from habitat_ovmm.utils.config_utils import create_env_config, get_habitat_config, get_omega_config,create_agent_config

debug = False
save_img = False
show_img = False
img_dir = "cyw/test_data/data_prepare_debug/heuristic_agent"

def visual_rotated_top_down_map(top_down_map,map_agent_coord,map_agent_rot):
    info = {
        "map":top_down_map,
        "fog_of_war_mask":None,
        "agent_map_coord":map_agent_coord,
        "agent_angle":map_agent_rot
    }
    return maps.colorize_draw_agent_and_fit_to_height_test(
        info
    )

def visual_top_down_map(top_down_map,map_agent_coord,map_agent_rot):
    info = {
        "map":top_down_map,
        "fog_of_war_mask":None,
        "agent_map_coord":map_agent_coord,
        "agent_angle":map_agent_rot
    }
    return maps.colorize_draw_agent_and_fit_to_height(
        info,222
    )

def visual_obstacle_map(obstacle_map,sensor_pose):
    ''' 可视化障碍物地图
        NOTE: obstacle_map 需要在sem_map返回的obstacle_map基础上，上下翻转，否则方位是错的
    '''
    obstacle_map_new = np.copy(obstacle_map)
    obstacle_map_new = obstacle_map_new * 255
    curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
    # Agent arrow
    pos = (
        (curr_x * 100.0 / 5 - gx1)
        * 480
        / obstacle_map.shape[0],
        (obstacle_map.shape[1] - curr_y * 100.0 / 5 + gy1)
        * 480
        / obstacle_map.shape[1],
        np.deg2rad(-curr_o),
    )
    # 为什么这里用 -curr_o ？
    # curr_o 是相对于 cv2 坐标系中 x轴 ➡ 顺时针旋转的角度
    # 因为cv坐标原点位于左上角，这里是为了在cv坐标系中把朝角表示出来
    agent_arrow = vu.get_contour_points(pos,origin=[0,0])
    cv2.drawContours(obstacle_map_new, [agent_arrow], 0, 128,-1)
    # NOTE 原本可视化的时候是把所有 image 都上下翻转了一下 

    # 把图像顺时针旋转 curr_o 角度
    rotation_origin = [int(pos[0]), int(pos[1])]
    obstacle_map_new = maps.rotate_matrix_numbers(
        matrix=obstacle_map_new,
        angle=-curr_o,
        center=rotation_origin
    )
    return obstacle_map_new

def visual_init_obstacle_map(obstacle_map,sensor_pose):
    ''' 可视化障碍物地图
        NOTE: 这里直接在sem_map的基础上可视化obstacle_map，因此与top_down_map存在上下翻转的问题
        # 经调试，这一版正确
    '''
    obstacle_map_new = np.copy(obstacle_map)
    obstacle_map_new = obstacle_map_new * 255
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
    start = [
        int(start_y * 100.0 / 5 - gx1),
        int(start_x * 100.0 / 5 - gy1),
    ]
    # NOTE 这里得到的坐标是 矩阵坐标系 ⬇ x ➡ y
    # Agent arrow
    pos = (
        start[1]* 480 / obstacle_map.shape[0],
        start[0]* 480 / obstacle_map.shape[1],
        np.deg2rad(start_o),
    )
    # 转为 cv2坐标系
    # gps坐标系下 start_o，因为现在在翻转的图片上作图，要做得夹角为 -start_o，传入的参数是 -(-start_o)
    agent_arrow = vu.get_contour_points(pos,origin=[0,0])
    cv2.drawContours(obstacle_map_new, [agent_arrow], 0, 128,-1)

    # 把图像顺时针旋转 curr_o 角度
    # rotate_matrix_numbers 要求输入坐标为 row_index col_index
    rotation_origin = [int(start[0]), int(start[1])]
    obstacle_map_new = maps.rotate_matrix_numbers(
        matrix=obstacle_map_new,
        angle=start_o,
        center=rotation_origin
    )
    return obstacle_map_new


def visual_waypoint(robo_view_map,agent_coord,relative_position,grid_resolution=[5,5],color=[255,0,0]):
    '''在地图上可视化waypoint
        robo_view_map: 机器人视角下的地图，机器人朝向为向右 ➡
        agent_position: row_index, col_index
        relative_position: (x,y) x表示向前 ➡，y表示向左 ⬆ 
    '''
    robo_view_map_new = robo_view_map.copy()
    coords = to_grid(relative_position[0],relative_position[1],grid_resolution,agent_coord)
    robo_view_map_new[coords[0]-3:coords[0]+4,coords[1]-3:coords[1]+4,:] = color
    # if debug:
    #     print(f"agent position is {agent_coord}, relative_position is {relative_position}, coords is {coords}")
    return robo_view_map_new

class data_prepare:
    '''数据准备,在map_prepare上面封装一层,抽取local map and target
        存储为文件
    '''
    def __init__(self,data_dir,env_config,agent_config) -> None:
        self.data_dir = data_dir
        self.map_prepare = map_prepare(env_config=env_config,agent_config=agent_config)
        self.h5py_dataset = h5py.File(os.path.join(data_dir,"data_out_reshaped.hdf5"),"r+")
        with open(os.path.join(data_dir,"place_waypoint.pkl"),"rb") as f:
            self.pkl_data = pickle.load(f)
        # self.processed_dataset = h5py.File(os.path.join(self.data_dir,"prepared_dataset.hdf5"),"w")
    
    def genrate_data(self):
        '''生成数据，并按照一定地格式存储
        '''
        # for each episode
        for scene_ep_data in tqdm(self.pkl_data):
            scene_id = scene_ep_data["scene_id"]
            episode_id = scene_ep_data["episode_id"]
            recep = scene_ep_data["recep"]
            skill_waypoint_data = scene_ep_data["skill_waypoint_data"]
            # if not f"scene_{scene_id}" in self.processed_dataset:
            #     self.processed_dataset.create_group(f"scene_{scene_id}")
            # self.processed_dataset.create_group(f"/scene_{scene_id}/ep_{episode_id}/")
            # for each recep
            for recep_id,skill_waypoint_singile_recep_data in enumerate(skill_waypoint_data):
                recep_position = skill_waypoint_singile_recep_data["recep_position"]
                each_view_point_data = skill_waypoint_singile_recep_data["each_view_point_data"]
                scene_ep_recep_grp = self.h5py_dataset[f"/scene_{scene_id}/ep_{episode_id}/{recep_position}"]
                # scene_ep_recep_grp_processed = self.processed_dataset.create_group(f"/scene_{scene_id}/ep_{episode_id}/{recep_position}")
                # gather all success view_point and transfor to relative recep
                recep_relative_pos, viewpoint_relative_pos = transform2relative(
                    data=each_view_point_data,
                    keep_success=True,
                    recep_position=recep_position,
                )
                # for each view_point
                local_top_down_map_s = []
                local_obstacle_map_s = []
                target_s = []
                recep_coord_s = []
                waypoint_s = []
                # # concatanate的效率更高，所以直接使用concate，然后reshape
                # view_point_position_s = np.reshape(scene_ep_recep_grp["view_point_position_s"],(len(recep_relative_pos),-1))
                # start_top_down_map_s = np.reshape(scene_ep_recep_grp["start_top_down_map_s"],(len(recep_relative_pos),-1,scene_ep_recep_grp["start_top_down_map_s"].shape[-1]))
                # start_obstacle_map_s = np.reshape(scene_ep_recep_grp["start_obstacle_map_s"],(len(recep_relative_pos),-1,scene_ep_recep_grp["start_obstacle_map_s"].shape[-1]))

                view_point_position_s = scene_ep_recep_grp["view_point_position_s"]
                start_top_down_map_s = scene_ep_recep_grp["start_top_down_map_s"]
                start_obstacle_map_s = scene_ep_recep_grp["start_obstacle_map_s"]
                if show_img or save_img:
                    start_rgb_s = scene_ep_recep_grp["start_rgb_s"]

                for i in range(len(recep_relative_pos)):
                    # 验证数据正确性用
                    # view_point_position_h5py = scene_ep_recep_grp["view_point_position_s"][i] 
                    view_point_position_h5py = view_point_position_s[i]
                    view_point_position_pkl = each_view_point_data[i]['view_point_position']
                    assert np.allclose(view_point_position_h5py, view_point_position_pkl,rtol=0.01),"the view point position is not equall"
                    waypoint_s.append(view_point_position_pkl) #验证数据正确性使用
                    # get local top down map
                    # start_top_down_map = scene_ep_recep_grp["start_top_down_map_s"][i]
                    start_top_down_map = start_top_down_map_s[i]
                    start_top_down_map_pose = each_view_point_data[i]["start_top_down_map_pose"]
                    start_top_down_map_rot = each_view_point_data[i]["start_top_down_map_rot"]
                    rotated_top_down_map,_ = self.map_prepare.rotate_top_down_map(
                        top_down_map=start_top_down_map,
                        map_agent_pos=start_top_down_map_pose[0],
                        map_agent_angle=start_top_down_map_rot[0]
                    )
                    local_rtdm = self.map_prepare.get_local_map(
                        global_map=rotated_top_down_map,
                        map_agent_pos=start_top_down_map_pose[0]
                    )
                    local_rtdm = np.flipud(local_rtdm)
                    # NOTE 为了和obstacle对应，将 local_rtdm 上下翻转
                    # get_obstacle_map
                    # start_obstacle_map = scene_ep_recep_grp["start_obstacle_map_s"][i]
                    start_obstacle_map = start_obstacle_map_s[i]
                    start_sensor_pose = each_view_point_data[i]["start_sensor_pose"]
                    rotated_obstacle_map,map_agent_pos = self.map_prepare.rotate_obstacle_map(
                        obstacle_map=start_obstacle_map,
                        sensor_pose=start_sensor_pose
                    )
                    loacal_rom = self.map_prepare.get_local_map(
                        global_map=rotated_obstacle_map,
                        map_agent_pos=map_agent_pos
                    )
                    # 以上其实是travisible map，只是分别从 gt 和 建立的图得到
                    # get_recep_coord
                    recep_coord = self.map_prepare.raletive_pos2localmap_coord(
                        relative_position=np.expand_dims(recep_relative_pos[i],axis=0),
                        flipud=True,
                        keep_local=False
                    )
                    # get target
                    if len(viewpoint_relative_pos[i]) != 0:
                        target_coord = self.map_prepare.raletive_pos2localmap_coord(
                            relative_position=np.stack(viewpoint_relative_pos[i]),
                            flipud=True,
                            keep_local=True
                        )
                    else:
                        target_coord = None
                    target_map = self.map_prepare.get_target_map(
                        localmap_coord=target_coord, 
                        gau_filter=True
                    )
                    '''save data'''
                    local_top_down_map_s.append(local_rtdm)
                    local_obstacle_map_s.append(loacal_rom)
                    target_s.append(target_map)
                    recep_coord_s.append(recep_coord)
                    # get local map position encoding
                    '''visulize*******************************'''
                    if show_img or save_img:
                        # rgb
                        # rgb_vis = scene_ep_recep_grp["start_rgb_s"]
                        rgb_vis = start_rgb_s[i]
                        # visualize initial top_down_map
                        init_top_down_map_vis = visual_top_down_map(
                            top_down_map=start_top_down_map,
                            map_agent_coord=start_top_down_map_pose,
                            map_agent_rot=start_top_down_map_rot,
                        )
                        top_down_map_vis = visual_rotated_top_down_map(
                            top_down_map=start_top_down_map,
                            map_agent_coord=start_top_down_map_pose,
                            map_agent_rot=start_top_down_map_rot,
                        )
                        # rotated local top_down_map
                        local_rtdm_vis = local_rtdm*255
                        # initial obstacle map
                        init_obstacle_map_vis = visual_init_obstacle_map(
                            obstacle_map=start_obstacle_map,
                            sensor_pose=start_sensor_pose
                        )
                        # flipup obstacle map
                        obstacle_map_vis = visual_obstacle_map(
                            obstacle_map=np.flipud(start_obstacle_map),
                            sensor_pose=start_sensor_pose
                        )
                        # rotated local obstacle map
                        loacal_rom_vis = loacal_rom *255
                        # success view point
                        target_no_gau = self.map_prepare.get_target_map(
                            localmap_coord=target_coord, 
                            gau_filter=False
                        )
                        target_no_gau_vis = target_no_gau*255
                        target_map_vis = target_map * 255
                        # recep_position
                        target_map_recep_vis = np.copy(target_map_vis)
                        target_map_recep_vis[recep_coord[0][0]-3:recep_coord[0][0]+4,recep_coord[1][0]-3:recep_coord[1][0]+4] = 255
                        # 把机器人当前位置，recep position success viewpoint 都标记在obstacle map上 
                        local_rom_recep_vis = loacal_rom.copy()
                        local_rom_recep_vis = local_rom_recep_vis * 255
                        local_rom_recep_vis = np.stack([local_rom_recep_vis,local_rom_recep_vis,local_rom_recep_vis],axis=-1)
                        local_rom_recep_vis[recep_coord[0][0]-3:recep_coord[0][0]+4,recep_coord[1][0]-3:recep_coord[1][0]+4,:] = [255,0,0]
                        local_rom_recep_vis[self.map_prepare.localmap_agent_pose[0]-3:self.map_prepare.localmap_agent_pose[0]+4,self.map_prepare.localmap_agent_pose[1]-3:self.map_prepare.localmap_agent_pose[1]+4,:] = [0,255,0]
                        local_rom_recep_vis[target_map!=0] = [0,0,255]
                        # 在top_down_map上可视化
                        top_down_map_waypoint = visual_waypoint(top_down_map_vis,start_top_down_map_pose[0],recep_relative_pos[i],[5,5],[255,0,0])
                        for single_viewpoint_pos in viewpoint_relative_pos[i]:
                            top_down_map_waypoint = visual_waypoint(top_down_map_waypoint,start_top_down_map_pose[0],single_viewpoint_pos,[5,5],[0,0,255])
                        ''' 
                        NOTE 需要验证的几个点：
                        1. whether rgb, top down map, obstacle map match
                        2. whether local match to global (match)
                        3. whether recep_position right
                        4. whether success view point reasonable
                        看起来没啥问题
                        '''
                        imgs = [
                            rgb_vis,
                            init_top_down_map_vis,
                            top_down_map_vis,
                            local_rtdm_vis,
                            init_obstacle_map_vis,
                            obstacle_map_vis,
                            loacal_rom_vis,
                            target_map_vis,
                            target_no_gau_vis,
                            target_map_recep_vis,
                            local_rom_recep_vis,
                            top_down_map_waypoint,
                        ]
                        img_names = [
                            "rgb_vis",
                            "init_top_down_map_vis",
                            "top_down_map_vis",
                            "local_rtdm_vis",
                            "init_obstacle_map_vis",
                            "obstacle_map_vis",
                            "loacal_rom_vis",
                            "target_map_vis",
                            "target_no_gau_vis",
                            "target_map_recep_vis",
                            "local_rom_recep_vis",
                            "top_down_map_waypoint",
                        ]
                        # 保存图像
                        if save_img:
                            save_img_dir = os.path.join(img_dir,f"scene_{scene_id}/ep_{episode_id}/{recep_id}/{i}")
                            os.makedirs(save_img_dir,exist_ok=True)
                            for img,img_name in zip(imgs,img_names):
                                cv2.imwrite(os.path.join(save_img_dir,f"{img_name}.jpg"),img)
                        if show_img:
                            for img,img_name in zip(imgs,img_names):
                                cv2.imshow(img_name,img)
                            cv2.waitKey(1)
                '''保存数据'''
                local_top_down_map_s = np.stack(local_top_down_map_s,axis=0)
                local_obstacle_map_s = np.stack(local_obstacle_map_s,axis=0)
                target_s = np.stack(target_s,axis=0)
                recep_coord_s = np.stack(recep_coord_s,axis=0)
                waypoint_s = np.stack(waypoint_s,axis=0)
                # scene_ep_recep_grp_processed.create_dataset(name="local_top_down_map_s",data=local_top_down_map_s)
                # scene_ep_recep_grp_processed.create_dataset(name="local_obstacle_map_s",data=local_obstacle_map_s)
                # scene_ep_recep_grp_processed.create_dataset(name="target_s",data=target_s)
                # scene_ep_recep_grp_processed.create_dataset(name="recep_coord_s",data=recep_coord_s)
                # scene_ep_recep_grp_processed.create_dataset(name="waypoint_s",data=waypoint_s)

                scene_ep_recep_grp.create_dataset(name="local_top_down_map_s",data=local_top_down_map_s)
                scene_ep_recep_grp.create_dataset(name="local_obstacle_map_s",data=local_obstacle_map_s)
                scene_ep_recep_grp.create_dataset(name="target_s",data=target_s)
                scene_ep_recep_grp.create_dataset(name="recep_coord_s",data=recep_coord_s)
                # scene_ep_recep_grp.create_dataset(name="waypoint_s",data=waypoint_s)

                
            # # 运行完一个episode数据
            # self.processed_dataset.flush()
            self.h5py_dataset.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    # @cyw
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="cyw/configs/debug/agent/heuristic_agent_place.yaml",
        help="Path to config yaml",
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
        habitat_config, env_config, evaluation_type="local"
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)
    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    data_prepare_ins = data_prepare(args.data_dir,env_config,agent_config)
    data_prepare_ins.genrate_data()
