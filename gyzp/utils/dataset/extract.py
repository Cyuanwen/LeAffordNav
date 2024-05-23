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
    depth_save_path=None,
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

    if not os.path.exists(os.path.dirname(image_save_path)):
        os.makedirs(os.path.dirname(image_save_path))

    if marked_image_save_path and not os.path.exists(marked_image_save_path):
        os.makedirs(marked_image_save_path)

    if depth_map is not None and depth_save_path:
        if not os.path.exists(os.path.dirname(depth_save_path)):
            os.makedirs(os.path.dirname(depth_save_path))

    Image.fromarray(image, "RGB").save(image_save_path)

    
    for tag in target_tags:
        target = np.zeros_like(semantic_map, dtype=np.uint8)
        target[semantic_map == tag] = 1
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if marked_image_save_path:
            res = cv2.drawContours(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), contours, -1, (0, 255, 0), 1
            )
            cv2.imwrite(
                os.path.join(
                    marked_image_save_path,
                    "id"
                    + label_save_path.split("/")[-1].split(".")[0]
                    + "-tag"
                    + str(tag)
                    + ".png",
                ),
                res,
            )

        if depth_map is not None and depth_save_path:
            depth = np.zeros_like(depth_map, dtype=np.float32)
            depth[semantic_map == tag] = depth_map[semantic_map == tag]
            depth = depth.flatten().astype(np.float32)
            depth = depth[depth != 0]
            with open(depth_save_path, "a") as ff:
                for d in depth:
                    ff.write(str(d) + "\n")

        for contour in contours:
            contour = contour.flatten().astype(np.float32)
            contour[0::2] /= img_y  # scale y
            contour[1::2] /= img_x  # scale x
            contour = contour.tolist()
            contour = ["{:.6f}".format(i) for i in contour]

            # save labels
            with open(label_save_path, "a") as f:
                line = str(tag - 2) + " " + " ".join(contour)
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
