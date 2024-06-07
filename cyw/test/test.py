from PIL import Image
import numpy as np

# # /raid/home-robot/src/home_robot/home_robot/perception/detection/yolo/yolov8_perception.py
# from home_robot.perception.detection.yolo.yolov8_perception import YOLOPerception

# model = YOLOPerception(
#   confidence_threshold = 0.75
# )
# rgb = Image.open("/raid/cyw/detection/ovmm_data/dataset_debug/332799.png")
# rgb = np.asarray(rgb)
# model.predict2(rgb)
import pickle
data_dir = "cyw/datasets/datasets_v1/recep_data/train"

with open(f"./{data_dir}/recep_position.pickle", "rb") as handle:
    receptacle_positions = pickle.load(handle)


print("pver")


