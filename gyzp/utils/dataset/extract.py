from distutils.log import info
from itertools import count
import os

import cv2
import numpy as np
from PIL import Image


def extract_labels(
    semantic_map,
    image,
    label_save_path,
    image_save_path,
    marked_image_save_path=None,
    depth_map=None,
    info_save_path=None,
):
    exclude_tags = [0, 1, 23]

    target_tags = np.unique(semantic_map)

    target_tags = [tag for tag in target_tags if tag not in exclude_tags]
    target_tags.sort()

    if not target_tags:
        return

    img_x, img_y = semantic_map.shape

    if not os.path.exists(os.path.dirname(label_save_path)):
        os.makedirs(os.path.dirname(label_save_path))
        with open(label_save_path, "w") as f:
            pass

    if not os.path.exists(os.path.dirname(image_save_path)):
        os.makedirs(os.path.dirname(image_save_path))

    if marked_image_save_path and not os.path.exists(marked_image_save_path):
        os.makedirs(marked_image_save_path)

    if depth_map is not None and info_save_path:
        if not os.path.exists(os.path.dirname(info_save_path)):
            os.makedirs(os.path.dirname(info_save_path))
        with open(info_save_path, "w") as f:
            pass

    # Save image
    Image.fromarray(image, "RGB").save(image_save_path)

    # image id
    id = label_save_path.split("/")[-1].split(".")[0]
    
    # traverse all tags
    for tag in target_tags:
        target = np.zeros_like(semantic_map, dtype=np.uint8)
        target[semantic_map == tag] = 1
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

        if marked_image_save_path:
            res = cv2.drawContours(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), contours, -1, (0, 255, 0), 1
            )
            cv2.imwrite(
                os.path.join(
                    marked_image_save_path,
                    str(id) + "-" + str(tag) + ".png",
                ),
                res,
            )

        if info_save_path:  # line : tag pixel_count avg_depth
            with open(info_save_path, "a") as f:
                pixel_count = np.sum(target)
                avg_depth = np.mean(depth_map[target == 1]) if depth_map is not None else -1
                line = str(tag) + " " + str(pixel_count) + " " + "{:.4f}".format(avg_depth)
                f.write(line + "\n")

def extract_goal_object(
    semantic_map,
    image,
    label_save_path,
    image_save_path,
    marked_image_save_path,
    goal_object_name,
    goal_object_id,
    pixel_count_save_path,
):
    exclude_tags = list(range(2, 24))

    target_tags = np.unique(semantic_map)

    target_tags = [tag for tag in target_tags if tag not in exclude_tags]
    target_tags.sort()

    if not target_tags:
        return

    img_x, img_y = semantic_map.shape

    if not os.path.exists(os.path.dirname(label_save_path)):
        os.makedirs(os.path.dirname(label_save_path))

    if not os.path.exists(os.path.dirname(image_save_path)):
        os.makedirs(os.path.dirname(image_save_path))

    if marked_image_save_path and not os.path.exists(marked_image_save_path):
        os.makedirs(marked_image_save_path)
        
    if not os.path.exists(os.path.dirname(pixel_count_save_path)):
        os.makedirs(os.path.dirname(pixel_count_save_path))

    Image.fromarray(image, "RGB").save(image_save_path)

    with open(label_save_path, "w") as f:
        for tag in target_tags:
            target = np.zeros_like(semantic_map, dtype=np.uint8)
            target[semantic_map == tag] = 1
            contours, hierarchy = cv2.findContours(
                target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            pixel_count = np.sum(target)
            with open(pixel_count_save_path, "a") as ff:
                ff.write(str(pixel_count) + "\n")

            if marked_image_save_path:
                res = cv2.drawContours(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), contours, -1, (0, 255, 0), 1
                )
                cv2.imwrite(
                    os.path.join(
                        marked_image_save_path,
                        "id"
                        + label_save_path.split("/")[-1].split(".")[0]
                        + "-"
                        + goal_object_name
                        + ".png",
                    ),
                    res,
                )

            for contour in contours:
                contour = contour.flatten().astype(np.float32)
                contour[0::2] /= img_y  # scale y
                contour[1::2] /= img_x  # scale x
                contour = contour.tolist()
                contour = ["{:.6f}".format(i) for i in contour]
                line = str(goal_object_id) + " " + " ".join(contour)
                f.write(line + "\n")
