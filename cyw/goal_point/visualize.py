'''
可视化函数
'''
import cv2
import numpy as np

from habitat.utils.visualizations import maps
import home_robot.utils.visualization as vu
from typing import Optional

# import sys
# sys.path.append("/raid/home-robot")
# from cyw.goal_point.utils import to_grid
from array import array
from typing import Tuple

def to_grid(
    realworld_x: array,
    realworld_y: array,
    grid_resolution: Tuple[int, int],
    origin: Tuple[int, int]
) -> Tuple[array, array]:
    r"""
        Argument:
            (x, y) where positive x is forward, positive y is translation to left in meters
            ➡ x
            ⬆ y
            realworld_x, realworld_y 为相对坐标
            grid_resolution: 每个格 横向和纵向分别代表多少 厘米
            origin: (行标, 列标)
        Return:
            row_index: 行标
            col_index: 列标
    """
    row_index = origin[0] - (realworld_y*100/grid_resolution[0]).astype(int)
    col_index = origin[1] + (realworld_x*100/grid_resolution[1]).astype(int)
    return row_index,col_index

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

def visual_init_obstacle_map_norotation(obstacle_map,sensor_pose):
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

    # # 把图像顺时针旋转 curr_o 角度
    # # rotate_matrix_numbers 要求输入坐标为 row_index col_index
    # rotation_origin = [int(start[0]), int(start[1])]
    # obstacle_map_new = maps.rotate_matrix_numbers(
    #     matrix=obstacle_map_new,
    #     angle=start_o,
    #     center=rotation_origin
    # )
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


def vis_local_map(local_map:np.array,recep_map:Optional[np.array],goal_map:Optional[np.array]):
    '''
        可视化 local map
        如果输入 recep_map的话，将rece 标记在 local map上
        goal_map
    '''
    local_map_vis = local_map.copy()
    local_map_vis = local_map_vis * 255
    local_map_vis =\
        np.stack([local_map_vis,local_map_vis,local_map_vis],axis=-1)
    if recep_map is not None:
        local_map_vis[recep_map == 1] = [255,0,0]
    if goal_map is not None:
        local_map_vis[goal_map==1] = [0,255,0]
    return local_map_vis

