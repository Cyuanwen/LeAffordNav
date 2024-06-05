'''
给定 semantic, 抽取 lable, 并进行验证
'''
from array import array
import numpy as np
import cv2
import os


def extract_labels(
    sematntic:array,
    label_save_path:str
):
    '''
    抽取轮廓信息，归一化，并保存到指定文件夹
    '''
    exclude_tags = [0, 1, 23]
    target_tags = np.unique(sematntic)
    target_tags = [tag for tag in target_tags if tag not in exclude_tags]
    target_tags.sort()
    img_x, img_y = sematntic.shape
    if not target_tags:
        return
        # NOTE 这样会导致如果semantic全是背景，就没有对应的标签
        # traverse all tags
    for tag in target_tags:
        target = np.zeros_like(sematntic, dtype=np.uint8)
        target[sematntic == tag] = 1
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_countour_count = 0
        with open(label_save_path, "a") as f:
            for contour in contours:
                if contour.shape[0] < 3:  # at least 3 points
                    # print("invalid contour", contour)
                    # print("shape", contour.shape)
                    # print("tag", tag)
                    continue
                valid_countour_count += 1
                contour = contour.flatten().astype(np.float32)
                contour[0::2] /= img_y  # scale y
                contour[1::2] /= img_x  # scale x
                contour = contour.tolist()
                contour = ["{:.6f}".format(i) for i in contour]
                line = str(tag - 2) + " " + " ".join(contour)
                f.write(line + "\n")
        if valid_countour_count == 0:
            # print("no valid contour for tag\n\n", tag)
            continue


if __name__ == "__main__":
    data_path = "/raid/home-robot/scene_ep_data/103997460_171030507"
    semantic_files = os.listdir(os.path.join(data_path,"semantic"))
    if not os.path.exists(os.path.join(data_path,"labels")):
        os.mkdir(os.path.join(data_path,"labels"))
    for semantic_file in semantic_files:
        semantic = np.load(os.path.join(data_path,"semantic",semantic_file))
        label_path = os.path.join(data_path,"labels",semantic_file.replace("npy","txt"))
        extract_labels(semantic,label_path)


