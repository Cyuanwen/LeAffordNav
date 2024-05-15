'''
yolov8物体分割，使用多边形框出物体（多边形，不是矩形）；
将分割结果与标准ground truth结果进行对比，计算yolo物体分割精度；
精度评价指标：IoU（Intersection over Union）交并比
标准ground truth结果格式：label.txt，每一行为一个物体信息：category x1 y1 x2 y2 x3 y3 x4 y4 ... （使用归一化坐标）
例如：
0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2 0.12 0.12 // 0表示类别，为多边形边界的归一化坐标，每两个为一个点的坐标
0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4 0.32 0.32
'''

from ultralytics import YOLO
from PIL import Image
import os

images_dir = '../data/images/val'
labels_dir = '../data/labels/val'
output_dir = '../output'

# 加载模型
model = YOLO('./models/yolov8m-seg.pt')

# 检测图片
results = model(images_dir)

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(os.path.join(output_dir, f"output_{i}.jpg"))
