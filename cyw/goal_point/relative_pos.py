'''
验证起点、终点、容器之间的相对位置
'''
import pickle
import numpy as np
import math

# 假设position 和  gps 之间坐标系存在旋转，平移
# 经测试，旋转角可能为 30 度，求解 偏移量
def get_offset(rotation_rad, init_pos,new_pos,rotation_matrix=None):
    '''
        rotation_rad: new_pos 对应坐标系 相对于 init_pos对应坐标系的旋转角度,弧度制
    '''
    if rotation_matrix is None:
        rotation_matrix = [[np.cos(rotation_rad),np.sin(rotation_rad)],[-np.sin(rotation_rad),np.cos(rotation_rad)]]
        rotation_matrix = np.array(rotation_matrix)
    offset = new_pos - np.matmul(rotation_matrix,init_pos)
    print(f"rotation_rad is {rotation_rad}")
    print(f"offset is {offset}")

def get_angle(vector1,vector2):
    # 计算向量的长度
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # 计算两个向量的内积
    dot_product = np.dot(vector1, vector2)

    # 计算夹角（弧度）
    cosine_angle = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(cosine_angle)

    # 将弧度转换为角度
    angle_deg = np.degrees(angle_rad)

    print(angle_deg)
    

# 求解旋转矩阵
def get_rotation(init_org_pos,new_org_pos):
    '''求解旋转矩阵
        rotation_matrix = [[a,b],
                            [-b,a]]
        new_pos = rotation_matrix @ init_pos
        NOTE init_pos 和 new_pos 要有相同的原点
        Return:
            rotation_matrix, rotation_rad
    '''
    A = np.array(
        [[init_org_pos[0],init_org_pos[1]],
        [-init_org_pos[1],init_org_pos[0]]]
    )
    par = np.linalg.inv(A) @ new_org_pos
    rotation_matrix = np.array([
        [par[0],par[1]],
        [-par[1],par[0]]
    ])
    rotation_angle = math.acos(par[0])
    print(f"par is {par} and the norm is {np.linalg.norm(par,ord=2)},rotation_angle is {rotation_angle}")
    return rotation_matrix, rotation_angle

if __name__ == "__main__":
    data_file = "cyw/datasets/place_dataset/test.pkl"
    with open(data_file,"rb") as f:
        data = pickle.load(f)
    for item in data:
        start_pos = item["start_info"]["position"]
        start_gps = item['start_info']['gps']
        end_pos = item["end_info"]["position"]
        end_gps = item['end_info']['gps']
        start_rot = item['start_info']["rotation"] 
        start_compass = item['start_info']["compass"]
        relative_position = np.array(end_pos) - np.array(start_pos)
        relative_position = relative_position[[0,2]]
        relative_gps = end_gps-start_gps
        relative_gps = np.array([-relative_gps[1],relative_gps[0]])
        angel = get_angle(relative_position,relative_gps)
        print(f"ralative_position {relative_position}, relative_gps {relative_gps}")
        # 计算向量的长度
        norm_pos = np.linalg.norm(relative_position)
        norm_gps = np.linalg.norm(relative_gps)
        print(f"norm_pos {norm_pos}, norm_gps {norm_gps}")
        # rotation_rad = np.radians(30)
        rotation_matrix, rotation_rad = get_rotation(relative_position,relative_gps)
        relative_rad = start_rot - start_compass
        ralative_rad_2 = -start_rot - start_compass
        print(f"relative_rad is {relative_rad}, ralative_rad_2 is {ralative_rad_2}") # 真实的坐标系转角可能存在正负关系
        get_offset(relative_rad[0],np.array(start_pos)[[0,2]],start_gps)
        get_offset(relative_rad[0],np.array(start_pos)[[0,2]],start_gps,rotation_matrix)


    print("over")