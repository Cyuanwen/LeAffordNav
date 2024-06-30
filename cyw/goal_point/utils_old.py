'''
代码基本可弃用

收集操作点数据的相关函数
当x,y坐标轴设置，以及夹角设置满足:
     x = l * cos(alpha)
     y = l * sin(alpha)
     其中 l表示向量长度，alpha表示向量与x轴的夹角，（x,y）表示向量坐标
当坐标系绕原点逆时针旋转 theta 角时，新坐标与原坐标满足以下关系：
    x' = cos(theta) x + sin(theta) y
    y' = -sin(theta) x + cos(theta) y
(可计算原本基在新坐标下的表示或几何关系变化获得)
'''
from array import array
import numpy as np
import math
import quaternion
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

debug = True

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

# 以下代码基本不使用
def coord_transfer(robot_gps:array,compass:array,gps:array)->array:
    '''
        坐标变换：将世界坐标系下的坐标转换为机器人坐标系下的坐标
        当x,y坐标轴设置，以及夹角设置满足:
            x = l * cos(alpha)
            y = l * sin(alpha)
            其中 l表示向量长度，alpha表示向量与x轴的夹角，（x,y）表示向量坐标
        当坐标系绕原点逆时针旋转 theta 角时，新坐标与原坐标满足以下关系：
            x' = cos(theta) x + sin(theta) y
            y' = -sin(theta) x + cos(theta) y
        若机器人坐标为（a,b）,compass theta
        则机器人视角下的坐标
            x' = cos(theta) (x-a) + sin(theta) (y-b)
            y' = -sin(theta) (x-a) + cos(theta) (y -b)
            NOTE 先做平移，再做旋转
        Argument: 
            robot_gps:机器人的GPS,(x,y) /m np.array
            compass: 机器人朝向, rad np.array
            gps: 要转换的位置的GPS, (x,y) /m
        Return:
            relative_gps: gps 在机器人坐标系下的坐标
    '''
    rotation_matrix = [[np.cos(compass),np.sin(compass)],[-np.sin(compass),np.cos(compass)]]
    rotation_matrix = np.array(rotation_matrix)
    rotation_matrix = np.squeeze(rotation_matrix)
    relative_gps = rotation_matrix @ (gps - robot_gps)
    return relative_gps

def get_rotation_matrix(relative_position,relative_gps):
    '''计算 机器人 base 坐标 与 gps 坐标的变换
        Argument: 
            relative_position: base坐标系下，两点的相对位置
            relative_gps： gps 坐标系下，同样两点的相对位置
        Return:
            rotation_matrix = [[a,b],
                            [-b,a]]
            relative_gps = rotation_matrix @ relative_position
    '''
    assert not (relative_position[0]== 0 and relative_position[1] == 0), "the position chage is 0"
    A = np.array(
        [[relative_position[0],relative_position[1]],
        [relative_position[1],-relative_position[0]]]
    )
    par = np.linalg.inv(A) @ relative_gps
    rotation_matrix = np.array([
        [par[0],par[1]],
        [-par[1],par[0]]
    ])
    if debug:
        rotation_angle = math.acos(par[0])
        print(f"par is {par} and the norm is {np.linalg.norm(par,ord=2)},rotation_angle is {rotation_angle}")
        if not np.allclose(relative_gps,(rotation_matrix @ relative_position),rtol=0.01):
            print("wrong!")
    return rotation_matrix

def get_offset_matrix(rotation_matrix:array,init_pos,new_pos):
    '''
        rotation_matrix: new_pos 对应坐标系 相对于 init_pos对应坐标系的旋转矩阵
        new_pos 与 init_pos 应存在如下关系： new_pos = rotation_matrix @ init_pos - offset
        因此 offset = rotation_matrix @ init_pos - new_pos
        这里 offset 的物理含义为 坐标系offset 与 rotation_matrix的乘积
    '''
    offset = rotation_matrix @ init_pos - new_pos
    return offset

def get_rotation_offset(old_position,new_position,old_gps,new_gps):
    '''计算旋转矩阵以及偏置
        Argument:
            old_position,new_position, base坐标系下的两个不同的位置
            old_gps,new_gps, gps坐标系下的两个位置
        Return
            rotation_matrix, offset
    '''
    relative_position = list(new_position - old_position)
    relative_position = np.array(relative_position)[[0,2]]
    relative_gps = new_gps - old_gps
    # assert np.linalg.norm(relative_position) == np.linalg.norm(relative_gps),"the norm of two relative location is not equall" 
    print(f"relative_position norm is {np.linalg.norm(relative_position)},relative_gps norm is {np.linalg.norm(relative_gps)}")
    rotation_matrix = get_rotation_matrix(
        relative_position=relative_position,
        relative_gps=relative_gps)
    offset_1 = get_offset_matrix(rotation_matrix=rotation_matrix,init_pos=np.array(old_position)[[0,2]],new_pos=old_gps)
    # assert offset_1 == offset_2, "the offset of initial and new is not equall"
    if debug:
        offset_2 = get_offset_matrix(rotation_matrix=rotation_matrix,init_pos=np.array(new_position)[[0,2]],new_pos=new_gps)
        print(f"offset_1 {offset_1},offset_2 {offset_2}") #两者基本相等
    return rotation_matrix, offset_1

def compare_rot_compass(rotation,compass):
    '''对比rotation 和 compass
        测试使用
    '''
    relative_rad = float(rotation) - compass[0]
    ralative_rad_2 = -float(rotation) - compass[0]
    if debug:
        print(f"relative_rad is {relative_rad}, ralative_rad_2 is {ralative_rad_2}")    

def transform_coord(position,rotaion_matrix,offset):
    '''将position坐标转换到gps坐标
        new_pos 与 init_pos 应存在如下关系： new_pos = rotation_matrix @ init_pos - offset
        Argument:
            position: base坐标系下的坐标位置
            rotation_matrix: 旋转矩阵
            offset:偏执向量
        Return:
            position 在 gps 坐标系下的坐标
    '''
    position_coor = np.array(position)[[0,2]]
    gps_coor = rotaion_matrix @ position_coor - offset # rotation_matrix @ init_pos - offset
    if debug:
        print(f"gps_coor is {gps_coor}")
    return gps_coor

# 将上面函数封装成类，以便调用
class coordinate_transform:
    def __init__(self) -> None:
        self.rotation_matrix = None
        self.offset = None

    def reset(self):
        self.rotation_matrix = None
        self.offset = None
    
    def set_rotation_offset(self,old_position,new_position,old_gps,new_gps):
        '''
            NOTE old_position 和 new_position 不能相同
        '''
        self.rotation_matrix,self.offset = get_rotation_offset(old_position,new_position,old_gps,new_gps)
    
    def get_gps_coord(self,position):
        gps_coord = transform_coord(position,self.rotation_matrix,self.offset)
        return gps_coord