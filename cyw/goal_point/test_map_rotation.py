'''
测试代码，具有一定借鉴意义，勿删
'''

from array import array
from habitat.utils.visualizations import maps
import cv2
import pickle
import os
import numpy as np
import math
import home_robot.utils.visualization as vu
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import sys
sys.path.append("/raid/home-robot")
from cyw.goal_point.utils import get_relative_position

debug = True

'''
关于坐标系的总结：
1. center:list 返回的是各个智能体的位置（实际只有一个智能体），分别为 图像点在 图像矩阵的 第几行，第几列 （有待验证）
2. angle 是机器人朝角与 top_down_map(未经任何变换) 竖直轴的夹角
3. curr_o 是相对于 cv2 坐标系中 x轴的逆时针旋转角度，逆时针为正
4. cv2 坐标系中 x 轴朝右， y轴朝下 （x,y）表示点的坐标，与矩阵索引恰好相反
'''


def draw_top_down_map(info):
    return maps.colorize_draw_agent_and_fit_to_height_test(
        info
    )

def rotate_matrix_numbers(matrix, angle, center):
    '''将矩阵围绕 center(行标，列标) 旋转 angle角度
        Argument:
            matrix: the matrix to rotation
            angle: 要旋转的角度，正 表示 逆时针旋转
            center: 旋转中心点,(cow_index, row_index)
    '''
    # 转换矩阵为图像
    image = np.array(matrix, dtype=np.uint8)

    image_show = image.copy()
    image_show = np.uint8(image_show*255/13)
    image_show[center[0][0]-2:center[0][0]+3,center[0][1]-2:center[0][1]+3] = 255
    cv2.imwrite("cyw/test_data/rotated_image/image_initial.jpg",image_show)
    
    # 获取图像的形状
    rows, cols = image.shape
    
    # 创建旋转矩阵
    rotation_point = [center[0][1],center[0][0]]
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
    
    # 进行图像旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # dst(x,y)=src(M11x+M12y+M13,M21x+M22y+M23)
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
    # dst(x,y) 取 （x,y）经 M 变换后的位置 的点，相当于 dst 是 原始图像 经过 M 逆变换的图像
    # 然而真正代码似乎和这有些出入，真正代码实现的确实是把 src 按照 M 旋转

    rotated_image_2 = warpAffine(image,rotation_matrix,(cols, rows))
    # 这部分和 “dst(x,y) 取 （x,y）经 M 变换后的位置 的点，相当于 dst 是 原始图像 经过 M 逆变换的图像” 一致

    image_show = rotated_image.copy()
    image_show = np.uint8(rotated_image*255/13)
    image_show[center[0][0]-2:center[0][0]+3,center[0][1]-2:center[0][1]+3] = 255
    cv2.imwrite("cyw/test_data/rotated_image/image_rotation.jpg",image_show)

    image_show = rotated_image_2.copy()
    image_show = np.uint8(rotated_image_2*255/13)
    image_show[center[0][0]-2:center[0][0]+3,center[0][1]-2:center[0][1]+3] = 255
    cv2.imwrite("cyw/test_data/rotated_image/image_rotation_2.jpg",image_show)
    
    # 将旋转后的图像转换回矩阵
    rotated_matrix = np.array(rotated_image, dtype=matrix.dtype)
    
    return rotated_matrix

def warpAffine(src, M, dsize):
    src_height, src_width = src.shape[:2]
    dst_width, dst_height = dsize

    dst = np.zeros((dst_height, dst_width), dtype=src.dtype)

    for y in range(dst_height):
        for x in range(dst_width):
            src_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            src_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]

            if src_x >= 0 and src_x < src_width-1 and src_y >= 0 and src_y < src_height-1:
                x1 = int(src_x)
                y1 = int(src_y)
                x2 = x1 + 1
                y2 = y1 + 1

                dx = src_x - x1
                dy = src_y - y1

                pixel_value = (1 - dx) * (1 - dy) * src[y1, x1] + \
                              dx * (1 - dy) * src[y1, x2] + \
                              (1 - dx) * dy * src[y2, x1] + \
                              dx * dy * src[y2, x2]

                dst[y, x] = pixel_value

    return dst


def visualize_obs(info):
    '''
        可视化 obstacle map
    '''
    obstacle_map = (1-info['obstacle_map'])*255
    # obstacle_map = info['obstacle_map']
    sensor_pose = info["sensor_pose"]
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
    # curr_o 是相对于 cv2 坐标系中 x轴 ➡ 的逆时针旋转角度，逆时针为正
    agent_arrow = vu.get_contour_points(pos,origin=[0,0])
    cv2.drawContours(obstacle_map, [agent_arrow], 0, 128,-1)
    # cv2.drawContours(image_vis, [agent_arrow], 0, color, -1)
    # NOTE 原本可视化的时候是把所有 image 都上下翻转了一下 

    # 把图像顺时针旋转 curr_o 角度
    rotation_origin = [int(pos[0]), int(pos[1])]
    obstacle_map = maps.rotate_matrix_numbers(
        matrix=obstacle_map,
        angle=-curr_o,
        center=rotation_origin
    )
    obstacle_map = np.uint8(obstacle_map)
    obstacle_map_new = np.dstack((obstacle_map, obstacle_map, obstacle_map))

    return obstacle_map_new,pos

def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    origin: Tuple[int, int]
) -> Tuple[int, int]:
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
    row_index = origin[0] - int(realworld_y*100/grid_resolution[0])
    col_index = origin[1] + int(realworld_x*100/grid_resolution[1])
    return int(row_index),int(col_index)


def gather_data(data:dict,key_name:str)->list:
    '''聚集字典中的 key_name 指定的数据
    Return: 该数据按顺序排放的 list
    '''
    key_data = []
    for item in data:
        key_data.append(item[key_name])
    return key_data

def transform2relative(data:dict,recep_position:array)->Tuple[list,list]:
    '''对data中的数据，对每个点，将其它点转为相对坐标
        recep_position: 验证用
        Return: 按顺序排列的
        recep_relative_pos, waypoint_relative_pos
    '''
    start_position = gather_data(data,"start_position")
    end_position = gather_data(data,"end_position")
    start_rotation = gather_data(data,"start_rotation")
    relative_recep_position = gather_data(data,"relative_recep_position")
    waypoints = []
    for i, start_position_singel in enumerate(start_position):
        start_rotation_singel = start_rotation[i]
        relative_end_pos = []
        for end_position_single in end_position:
            relative_end_pos_single = get_relative_position(
                current_position = start_position_singel,
                rotation_world_current = start_rotation_singel,
                position=end_position_single
            )
            relative_end_pos.append(relative_end_pos_single)
        if debug:
            relative_recep_position_new = get_relative_position(
                current_position = start_position_singel,
                rotation_world_current = start_rotation_singel,
                position=recep_position
                )
            assert np.allclose(relative_recep_position_new,relative_recep_position[i],rtol=0.01),"relative position transfer wrong"
        waypoints.append(relative_end_pos)
    return relative_recep_position, waypoints
  
def visual_waypoint(robo_view_map,agent_coord,relative_position,grid_resolution=[5,5],color=[255,0,0]):
    '''在地图上可视化waypoint
        robo_view_map: 机器人视角下的地图，机器人朝向为向右 ➡
        agent_position: row_index, col_index
        relative_position: (x,y) x表示向前 ➡，y表示向左 ⬆ 
    '''
    robo_view_map_new = robo_view_map.copy()
    coords = to_grid(relative_position[0],relative_position[1],grid_resolution,agent_coord)
    robo_view_map_new[coords[0]-3:coords[0]+4,coords[1]-3:coords[1]+4,:] = color
    if debug:
        print(f"agent position is {agent_coord}, relative_position is {relative_position}, coords is {coords}")
    return robo_view_map_new

def visual_waypoint_obs(robo_view_map,agent_coord,relative_position,grid_resolution=[5,5],color=128):
    '''在地图上可视化waypoint
        robo_view_map: 机器人视角下的地图，机器人朝向为向右 ➡
        agent_position: row_index, col_index
        relative_position: (x,y) x表示向前 ➡，y表示向左 ⬆ 
    '''
    robo_view_map_new = robo_view_map.copy()
    coords = to_grid(relative_position[0],relative_position[1],grid_resolution,agent_coord)
    coords = [int(coords[0]),int(coords[1])]
    robo_view_map_new[coords[0]-3:coords[0]+4,coords[1]-3:coords[1]+4] = color
    # robo_view_map_new[coords[1]-3:coords[1]+4,coords[0]-3:coords[0]+4,:] = color
    return robo_view_map_new

if __name__ == "__main__":
    data_dir = "cyw/test_data/top_down_map_data"
    pkl_datas = os.listdir(data_dir)[:1]

    info_dir = "cyw/test_data/info_data"
    info_datas = os.listdir(info_dir)[:1]

    place_waypoint_dir = "cyw/datasets/place_dataset/train/rl_agent_place_place_waypoint.pkl"
    with open(place_waypoint_dir,"rb") as f:
        place_waypoint_datas = pickle.load(f)
    
    recep_relative_pos, viewpoint_relative_pos = transform2relative(place_waypoint_datas[0]["skill_waypoint_data"][0]["each_view_point_data"],place_waypoint_datas[0]["skill_waypoint_data"][0]["recep_position"])

    for i in range(len(pkl_datas)):
        with open(os.path.join(data_dir,f"top_down_map_{i+1}.pkl"),"rb") as f:
            pkl_data = pickle.load(f)
        top_down_map_1 = draw_top_down_map(pkl_data, 640)
        # MAP_INVALID_POINT = 0
        # MAP_VALID_POINT = 1
        # top_down_map 中 0 表示障碍物
        rotated_matrix = rotate_matrix_numbers(
            matrix = pkl_data['map'],
            angle = -math.degrees(pkl_data['agent_angle'][0]), # 地图要按照机器人旋转方向的反方向旋转
            center= pkl_data['agent_map_coord'],
        )
        cv2.imwrite(f"cyw/test_data/top_down_map_1/top_down_map_{i}.jpg",top_down_map_1)

        with open(os.path.join(info_dir,f"info_{i+1}.pkl"),"rb") as f:
            info_data = pickle.load(f)
        obstacle_map, pos = visualize_obs(info=info_data)
        # obstacle_map 中 1 表示障碍物
        # cv2.imwrite(f"cyw/test_data/obstacle_map_agent_initial/obstacle_map_{i}.jpg",obstacle_map)
        cv2.imwrite(f"cyw/test_data/obstacle_map_agent/obstacle_map_{i}.jpg",obstacle_map)

        # agent_coord = [pkl_data['agent_map_coord'][0][0],pkl_data['agent_map_coord'][0][1]]
        # top_down_map_waypoint = visual_waypoint(top_down_map_1,agent_coord,recep_relative_pos[0],[5,5],[255,255,0])
        # for single_viewpoint_pos in viewpoint_relative_pos[0]:
        #     top_down_map_waypoint = visual_waypoint(top_down_map_waypoint,agent_coord,single_viewpoint_pos,[5,5],[0,255,0])
        # cv2.imwrite(f"cyw/test_data/top_down_map_viewpoint/top_down_map_viewpoint_{i}.jpg",top_down_map_waypoint)

        # 可视化障碍物
        agent_coord = pos
        obstacle_map_waypoint = visual_waypoint(obstacle_map,agent_coord,recep_relative_pos[0],[5,5],[255,255,0])
        for single_viewpoint_pos in viewpoint_relative_pos[0]:
            obstacle_map_waypoint = visual_waypoint(obstacle_map_waypoint,agent_coord,single_viewpoint_pos,[5,5],[0,255,0])
        cv2.imwrite(f"cyw/test_data/obstacle_map_viewpoint/obstacle_map_viewpoint_{i}.jpg",obstacle_map_waypoint)

    


    print("over")


