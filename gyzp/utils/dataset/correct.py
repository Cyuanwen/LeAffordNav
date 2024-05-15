import os

import cv2
import numpy as np
from tqdm import tqdm


def check_image_size(image_path):
    """
    检查图片的尺寸，输出img_x, img_y
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_x, img_y = img.shape
    print("img_x: {}, img_y: {}".format(img_x, img_y))
    return img_x, img_y


def correct_label(label_file_path, img_x, img_y):
    with open(label_file_path, "r") as f:
        lines = f.readlines()
    ratio = img_x / img_y
    with open(label_file_path, "w") as f:
        for line in lines:
            line = line.strip().split()
            tag = line[0]
            contour = line[1:]
            contour = np.array([float(i) for i in contour])
            contour[0::2] *= ratio
            contour[1::2] /= ratio
            contour = ["{:.6f}".format(i) for i in contour]
            line = tag + " " + " ".join(contour)
            f.write(line + "\n")

def delete_low_dim_segment(label_file_path, img_x, img_y):
    """
    有些yolo标签的维度较低，仅有一个点或两个点，删除这类多边形（至少有三个点才能保留）
    """
    with open(label_file_path, "r") as f:
        lines = f.readlines()
    with open(label_file_path, "w") as f:
        for line in lines:
            line = line.strip().split()
            tag = line[0]
            contour = line[1:]
            contour = np.array([float(i) for i in contour])

            # countour每个float，clip(0, 1) 限制在0-1之间
            contour = np.clip(contour, 0, 1)

            contour = contour.reshape(-1, 2)
            if contour.shape[0] >= 3:
                contour = contour.flatten()
                contour = ["{:.6f}".format(i) for i in contour]
                line = tag + " " + " ".join(contour)
                f.write(line + "\n")

def correct_labels(labels_dir_list, img_x, img_y):
    for labels_dir in labels_dir_list:
        for label_file in tqdm(os.listdir(labels_dir)):
            # correct_label(os.path.join(labels_dir, label_file), img_x, img_y)
            delete_low_dim_segment(os.path.join(labels_dir, label_file), img_x, img_y)


def main():
    labels_dir_list = [
        "/raid/home-robot/gyzp/data/labels/train",
        "/raid/home-robot/gyzp/data/labels/val",
    ]
    image_path = "/raid/home-robot/gyzp/data/images/train/1.png"
    img_x, img_y = check_image_size(image_path)
    correct_labels(labels_dir_list, img_x, img_y)


if __name__ == "__main__":
    main()
