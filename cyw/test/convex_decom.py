'''
凸分解
'''
# from shapely.geometry import Polygon

# # 定义一个多边形，这里以一个简单的五边形为例
# polygon = Polygon([(0, 0), (1, 2), (2, 2), (2, 1), (1, 0)])

# # 进行凸分解
# convex_decomposition = list(polygon.convex_decompose())

# # 输出每个凸多边形的顶点
# for poly in convex_decomposition:
#     print(list(poly.exterior.coords))


# import pyclipper

# # 定义一个多边形，这里以一个简单的五边形为例
# polygon = [(0, 0), (1, 2), (2, 2), (2, 1), (1, 0)]

# # 创建 Pyclipper 实例
# pc = pyclipper.Pyclipper()

# # 添加多边形
# pc.AddPath(polygon, pyclipper.PT_SUBJECT, True)

# # 执行凸分解
# solution = pc.Execute(pyclipper.CT_INTERSECTION)

# # 输出每个凸多边形的顶点
# for polygon in solution:
#     print(polygon)

# from scipy.spatial import ConvexHull

# hull = ConvexHull(polygon)


# import pyclipper
 
# # 定义一个多边形，使用XY坐标表示
# polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
 
# # 创建Clipper实例
# c = pyclipper.Clipper()
 
# # 添加多边形到Clipper
# subj = pyclipper.Polygon(polygon)
# c.AddPath(subj, pyclipper.PT_SUBJ, True)
 
# # 执行凸分解
# solution = pyclipper.PolyTree()
# c.Execute(pyclipper.CT_UNION, solution, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
 
# # 处理解决方案
# convex_solution = []
# for i in solution.Childs:
#     convex_solution.extend(i.Contour)
 
# # 输出结果
# print(convex_solution)



import cv2
import pickle
import numpy as np
from scipy.stats import mode
from home_robot.utils.image import smooth_mask


def get_test_mask():
    # Create an image
    r = 100
    mask = np.zeros((4 * r, 4 * r), dtype=np.uint8)

    # Create a sequence of points to make a contour
    vert = [None] * 6
    vert[0] = (3 * r // 2, int(1.34 * r))
    vert[1] = (1 * r, 2 * r)
    vert[2] = (3 * r // 2, int(2.866 * r))
    vert[3] = (5 * r // 2, int(2.866 * r))
    vert[4] = (3 * r, 2 * r)
    vert[5] = (5 * r // 2, int(1.34 * r))
    # Draw it in mask
    for i in range(6):
        cv2.line(mask, vert[i], vert[(i + 1) % 6], (255), 63)
    return mask


# mask = get_test_mask()
with open('cyw/test_data/place_point/reachable_point_cloud.pkl','rb') as f:
    reachable_point_cloud = pickle.load(f)
z = np.copy(reachable_point_cloud[:,:,2])
z = np.round(z,2)
z_mode = mode(z.reshape(-1,1)).mode
h_plant = z == z_mode # a mask
h_plant = smooth_mask(
    h_plant, np.ones((5, 5), np.uint8), num_iterations=5
)[1]
mask = h_plant*255
"""
Get the maximum/largest inscribed circle inside mask/polygon/contours.
Support non-convex/hollow shape
"""
dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
_, radius, _, center = cv2.minMaxLoc(dist_map)

result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.circle(result, tuple(center), int(radius), (0, 0, 255), 2, cv2.LINE_8, 0)

# minEnclosingCircle directly by cv2
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
center2, radius2 = cv2.minEnclosingCircle(np.concatenate(contours, 0))
cv2.circle(result, (int(center2[0]), int(center2[1])), int(radius2), (0, 255, 0,), 2)

cv2.imshow("mask", mask)
cv2.imshow("result", result)
cv2.imwrite("cyw/test_data/place_point/result.jpg", result)
cv2.waitKey()