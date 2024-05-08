import numpy as np
from PIL import Image
import pickle
import cv2


# semantic_labels = semantic_category_mapping.goal_id_to_goal_name
# # 创建一个空白的图例图像
# legend_img = np.zeros((len(semantic_labels) * 20, 100), dtype=np.uint8)

# # 在图例图像上绘制每个颜色块和对应的标签
# for idx,name in semantic_labels.items():
#     legend_img[idx * 20:(idx + 1) * 20,:] = palette[idx]
#     print(f"{idx}:{palette[idx]}")
# semantic_map_vis = Image.new(
#             "P", (legend_img.shape[1], legend_img.shape[0])
#         )
# semantic_map_vis.putpalette(palette)
# semantic_map_vis.putdata(legend_img.flatten().astype(np.uint8))
# semantic_map_vis = semantic_map_vis.convert("RGB")
# semantic_map_vis = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]
# cv2.imwrite("cyw/img/test_image/vis_legend.jpg",semantic_map_vis)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# # 定义数字调色板
# palette = [
#     0,  # 0: 背景
#     1,  # 1: 物体类别1
#     2,  # 2: 物体类别2
#     # ... 其他物体类别的数字
# ]

# # 定义物体类别名称列表
# class_names = [
#     "背景",
#     "物体类别1",
#     "物体类别2",
#     # ... 其他物体类别的名称
# ]

# # # 创建数字和物体类别的映射字典
# # number_to_class = {number: class_name for number, class_name in zip(palette, class_names)}
# class_names = semantic_category_mapping.goal_id_to_goal_name
# # 创建图例
# legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=cm.tab10(number)) for number in palette[1:len(class_names)+1]]
# plt.legend(legend_patches, class_names)

# # 显示图例
# plt.axis('off')
# plt.show()

# print('over')

import numpy as np
import matplotlib.pyplot as plt

# 假设有一个协方差矩阵
cov_matrix = np.array([[1.0, 0.5, 0.3],
                       [0.5, 1.0, 0.2],
                       [0.3, 0.2, 1.0]])

# 使用热图可视化协方差矩阵
plt.imshow(cov_matrix, cmap='Blues', interpolation='nearest')

# 添加颜色条
plt.colorbar( cmap='Blues')

# 自定义横轴和纵轴标签
variables = ['Variable 1', 'Variable 2', 'Variable 3']
plt.xticks(range(len(variables)), variables)
plt.yticks(range(len(variables)), variables)

plt.title('Covariance Matrix')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.show()
print("over")