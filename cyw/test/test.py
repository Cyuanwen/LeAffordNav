from habitat.utils.visualizations import maps
import cv2
import pickle
import os
import numpy as np
import math


data_dir = "cyw/test_data/top_down_map_data"

def draw_top_down_map(info, output_size):
    # return maps.colorize_draw_agent_and_fit_to_height(
    #     info, output_size
    # )
    return maps.colorize_draw_agent_and_fit_to_height_test(
        info, output_size
    )

def rotate_matrix_numbers(matrix, angle, center):
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

import cv2
import numpy as np

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


if __name__ == "__main__":
    pkl_datas = os.listdir(data_dir)
    for i,pkl_data in enumerate(pkl_datas):
        with open(os.path.join(data_dir,pkl_data),"rb") as f:
            pkl_data = pickle.load(f)
        top_down_map_1, top_down_map_2 = draw_top_down_map(pkl_data, 640)
        rotated_matrix = rotate_matrix_numbers(
            matrix = pkl_data['map'],
            angle = -math.degrees(pkl_data['agent_angle'][0]), # 地图要按照机器人旋转方向的反方向旋转
            center= pkl_data['agent_map_coord'],
        )
        # rotated_matrix = rotate_matrix_numbers(
        #     matrix = pkl_data['map'],
        #     angle = 90, # 地图要按照机器人旋转方向的反方向旋转
        #     center= pkl_data['agent_map_coord'],
        # )
        pkl_data['map'] = rotated_matrix
        top_down_map_1, top_down_map_2 = draw_top_down_map(pkl_data, 640)
        cv2.imwrite(f"cyw/test_data/top_down_map_1/top_down_map_{i}.jpg",top_down_map_1)
        cv2.imwrite(f"cyw/test_data/top_down_map_2/obstacle_map_{i}.jpg",top_down_map_2)

