'''
收集操作点数据的相关函数
当x,y坐标轴设置，以及夹角设置满足:
     x = l * cos(alpha)
     y = l * sin(alpha)
     其中 l表示向量长度，alpha表示向量与x轴的夹角，（x,y）表示向量坐标
当坐标系绕原点逆时针旋转 theta 角时，新坐标与原坐标满足以下关系：
    x' = cos(theta) x + sin(theta) y
    y' = -sin(theta) x + cos(theta) y
(可计算原本基在新坐标下的表示或几何关系变化获得)
(然而 habitat 放置环境范围的position是三维空间的位置，不能这么简单的变换)

关于坐标系的总结：
1. center:list 返回的是各个智能体的位置（实际只有一个智能体），分别为 图像点在 图像矩阵的 第几行，第几列
2. angle 是机器人朝角与 top_down_map(未经任何变换) 竖直轴的夹角
3. curr_o 是相对于 cv2 坐标系中 x轴的逆时针旋转角度，逆时针为正
4. cv2 坐标系中 x 轴朝右， y轴朝下 （x,y）表示点的坐标，与矩阵索引恰好相反
'''
from array import array
import numpy as np
import math
import quaternion
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)
import cv2
from habitat.utils.visualizations import maps
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter

debug = True
visualize = True

def get_relative_position(
        current_position:np.array,
        rotation_world_current:quaternion.quaternion,
        position:np.array
    ):
    '''以current_position， current_rotation 为朝向，建立坐标系，计算 position 在此坐标系下的位置
    '''
    origin = np.array(current_position, dtype=np.float32)
    relative_position = quaternion_rotate_vector(
            rotation_world_current.inverse(), position - origin
            )
    gps = np.array(
        [relative_position[0],-relative_position[2]],
        dtype=np.float32,
    )
    return gps

def rotate_matrix_numbers(matrix, angle, center):
    '''将矩阵围绕 center(行标，列标) 旋转 angle角度
        Argument:
            matrix: the matrix to rotation
            angle: 要旋转的角度，正 表示 逆时针旋转, 角度制
            center: 旋转中心点,(cow_index, row_index)
    '''
    # 转换矩阵为图像
    image = np.array(matrix, dtype=np.uint8)

    # 获取图像的形状
    if len(image.shape) == 2:
        rows, cols= image.shape
    else:
        rows, cols, _ = image.shape
    # TODO 之后统一后可以去掉
    
    # 创建旋转矩阵
    rotation_point = [center[1],center[0]] # cv2 坐标系 ➡ x， ⬇ y
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
    
    # 进行图像旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # dst(x,y)=src(M11x+M12y+M13,M21x+M22y+M23)
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
    # 按照文档说明 dst(x,y) 取 （x,y）经 M 变换后的位置 的点，相当于 dst 是 原始图像 经过 M 逆变换的图像
    # 然而真正代码似乎和这有些出入，真正代码实现的确实是把 src 按照 M 旋转
    
    # 将旋转后的图像转换回矩阵
    rotated_matrix = np.array(rotated_image, dtype=matrix.dtype)
    
    return rotated_matrix

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

def flipup_grid(row_index,col_index,image_shape):
    '''将 行坐标和列坐标进行上下翻转（为了与obstacle对齐）
    '''
    new_row_index = row_index.copy()
    new_row_index = image_shape[0]-new_row_index
    return new_row_index,col_index

def gather_data(data:dict,key_name:str)->list:
    '''聚集字典中的 key_name 指定的数据
    Return: 该数据按顺序排放的 list
    '''
    key_data = []
    for item in data:
        key_data.append(item[key_name])
    return key_data

def gather_success_end_position(data):
    '''只记录成功的end_position
    '''
    end_position_s = []
    for item in data:
        if item["place_success"]:
            end_position_s.append(item['end_position'])
    return end_position_s
    

def transform2relative(data:dict,keep_success:bool=True,recep_position:Optional[array]=None)->Tuple[list,list]:
    '''对data中的数据，对每个点，将其它点转为相对坐标
        Argument:
            data: 数据采集的 skill_waypoint_singile_recep_data["each_view_point_data"]
            recep_position: recep的绝对位姿 验证用
            keep_success: 是否只保留成功的end_position
        Return: 
            按顺序排列的recep_relative_pos, waypoint_relative_pos
    '''
    start_position = gather_data(data,"start_position")
    if keep_success:
        end_position = gather_success_end_position(data)
    else:
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
        if debug and recep_position is not None:
            relative_recep_position_new = get_relative_position(
                current_position = start_position_singel,
                rotation_world_current = start_rotation_singel,
                position=recep_position
                )
            assert np.allclose(relative_recep_position_new,relative_recep_position[i],rtol=0.01),"relative position transfer wrong"
        waypoints.append(relative_end_pos)
    return relative_recep_position, waypoints

class map_prepare:
    '''准备 map, 以及 target
        1. 将障碍物地图根据机器人朝向进行旋转，使得机器人朝向始终为右 ➡
        2. 选取旋转后的地图的一个局部
        3. 计算该局部地图各个点的交互概率
    '''
    def __init__(self,env_config,agent_config) -> None:
        self.top_down_resolution = getattr(env_config.habitat.task.measurements.top_down_map,"meters_per_pixel",None) # top_down_map 每个网格多少米
        self.top_down_resolution = self.top_down_resolution * 100
        self.semmap_resolution = getattr(agent_config.AGENT.SEMANTIC_MAP,"map_resolution",None)
        assert self.top_down_resolution == self.semmap_resolution, "the map resolution is not the same"
        self.map_bound_meter = 10 # 取机器人多少米范围的地图作为局部地图，单位：m
        self.gau_sigma = 2 # 高斯平滑的sigma参数
        self.grid_bound = int(self.map_bound_meter*100/self.semmap_resolution)
        self.localmap_agent_pose = [self.grid_bound//2,self.grid_bound//2]
        self.map_size = [(self.grid_bound//2)*2,(self.grid_bound//2)*2]

    def rotate_top_down_map(self,top_down_map:array,map_agent_pos:array,map_agent_angle:array):
        '''以机器人当前位置为旋转点，旋转地图，使得机器人朝向为 ➡
            Argument:
                topdown_map_info: sensor matric 返回的 top_down_map，包含地图占用信息以及agent location
                agent_location:在地图中的 row_index col_index
                agent_angle # 返回的为 agent 与 ⬇ 的夹角，逆时针为正，顺时针为负
            Return:
                可访问图：1表示可访问，0表示不可访问（即 0 表示障碍物）
                map_agent_pos: 机器人在旋转后的地图的位置
        '''
        top_down_map_travisible = np.zeros((top_down_map.shape[0],top_down_map.shape[1]))
        top_down_map_travisible[top_down_map==maps.MAP_INVALID_POINT] = 0
        top_down_map_travisible[top_down_map!=maps.MAP_INVALID_POINT] = 1

        top_down_map_travisible = rotate_matrix_numbers(
            matrix= top_down_map_travisible,
            angle= -math.degrees(map_agent_angle)+90,
            center= map_agent_pos
        )
        return top_down_map_travisible, map_agent_pos
    
    def rotate_obstacle_map(self,obstacle_map,sensor_pose):
        '''以机器人当前位置为旋转点，旋转地图，使得机器人朝向为 ➡
            Argument:
                obstacle_map: agent返回的建立的语义地图 并非完全的 0-1 变量，似乎越靠近1 越代表有障碍物
                sensor_pose： agent返回的（x,y,o,local_boundary） 7x
            Return:
                可访问图：1表示可访问，0表示不可访问（即 0 表示障碍物）
                map_agent_pos: 机器人在旋转后的地图的位置
        '''
        # print("debug") # obstacle似乎不是 0，1 是的
        travisible_map = 1-obstacle_map
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        start = [
            int(start_y * 100.0 / 5 - gx1),
            int(start_x * 100.0 / 5 - gy1),
        ]
        # 把图像顺时针旋转 curr_o 角度
        # rotate_matrix_numbers 要求输入坐标为 row_index col_index
        rotation_origin = [int(start[0]), int(start[1])]
        travisible_map = maps.rotate_matrix_numbers(
            matrix=travisible_map,
            angle=start_o,
            center=rotation_origin
        )
        return travisible_map, rotation_origin

    def get_local_map(self,global_map,map_agent_pos):
        '''map_agent_pos 为中心，从global_map中，向右和向右分别取self.grid_bound个像素，向上和向下分别取 self.grid_bound/2 个像素，如果超出边界，用 0 填充
            Argument:
                global_map
                map_agent_pos: agent 在 global_map 上的位置 （row_index,col_index）
            Return:
                local_map: shape (self.grid_bound,self.grid_bound)
                local_map_agent_pose: (row_index,col_index)
        '''
        global_shape = global_map.shape
        left_bound = map_agent_pos[1] - self.grid_bound//2
        right_bound = map_agent_pos[1] + self.grid_bound//2
        up_bound = map_agent_pos[0] - self.grid_bound//2
        low_bound = map_agent_pos[0] + self.grid_bound//2

        if debug:
            if right_bound>global_shape[1] or up_bound < 0 or low_bound > global_shape[1]:
                print("out of boundary")
        
        left_bound_clip = np.clip(left_bound,0,global_shape[1])
        right_bound_clip = np.clip(right_bound,0,global_shape[1])
        up_bound_clip = np.clip(up_bound,0,global_shape[0])
        low_bound_clip = np.clip(low_bound,0,global_shape[0])
        left_pad = left_bound_clip -left_bound
        right_pad = right_bound - right_bound_clip
        up_pad = up_bound_clip - up_bound
        low_pad = low_bound - low_bound_clip

        local_map = global_map[up_bound_clip:low_bound_clip,left_bound_clip:right_bound_clip]
        local_map = np.pad(local_map,((up_pad,low_pad),(left_pad,right_pad)),mode='constant')
        assert list(local_map.shape) == self.map_size,"the map shape is wrong"
        # if debug:
        #     print("debug") #NOTE 测试填充是否正确,经测试，正确
        
        return local_map
    
    def raletive_pos2localmap_coord(self,relative_position:array,flipud:bool=True,keep_local:bool=True):
        '''将相对位姿转换为局部地图的坐标
            Argument:
                relative_position: shape: (n,2) 第一列表示 x，第二列表示 y，其中 (x,y) x表示向前 ➡，y表示向左 ⬆
                keep_local: 只保留在local map里面的点
                flipud: 是否将坐标进行上下翻转
            Return: row_index,col_index
        '''
        row_index,col_index = to_grid(
            realworld_x = relative_position[:,0],
            realworld_y = relative_position[:,1],
            grid_resolution = [self.semmap_resolution,self.semmap_resolution],
            origin = self.localmap_agent_pose
        )
        if flipud:
            row_index,col_index = flipup_grid(row_index,col_index,self.map_size)
        if keep_local:
            # 只保留在局部地图范围内的坐标
            keep_index = (row_index>=0) & (row_index<self.grid_bound) & (col_index>=0) & (col_index<self.grid_bound)
            row_index = row_index[keep_index]
            col_index = col_index[keep_index]
        return row_index,col_index

    def get_target_map(self,localmap_coord:array,gau_filter:bool=True)->array:
        '''将 localmap_coord 在地图上标为1，其它为0
            Argument:
                localmap_coord: 坐标点
                gau_filter: 是否使用高斯滤波
            Return:
                target_map:array
        '''
        target_map = np.zeros(self.map_size)
        target_map[localmap_coord[0],localmap_coord[1]] = 1
        # 高斯平滑
        if gau_filter:
            target_map = gaussian_filter(target_map,sigma=self.gau_sigma)
            target_map = target_map / target_map.max()
        return target_map
        





        




