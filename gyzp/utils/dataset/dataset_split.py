import os
import random
import shutil
from math import isclose
from typing import List, Optional

from tqdm import tqdm


def dataset_split(
    source_images_dir: str,
    source_labels_dir: str,
    dest_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    """
    Split a dataset into train, val and test sets
    :param source_images_dir: the source directory of the dataset
    :param source_labels_dir: the source directory of the labels
    :param dest_dir: the destination directory of the dataset
    :param train_ratio: the ratio of the train set
    :param val_ratio: the ratio of the val set
    :param test_ratio: the ratio of the test set
    :param shuffle: whether to shuffle the dataset before splitting
    :param seed: the random seed
    """
    assert isclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ), "The sum of train_ratio, val_ratio and test_ratio must be 1.0."

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(os.path.join(dest_dir, "images")):
        os.makedirs(os.path.join(dest_dir, "images"))
    if not os.path.exists(os.path.join(dest_dir, "images", "train")):
        os.makedirs(os.path.join(dest_dir, "images", "train"))
    if not os.path.exists(os.path.join(dest_dir, "images", "val")):
        os.makedirs(os.path.join(dest_dir, "images", "val"))
    if not os.path.exists(os.path.join(dest_dir, "labels")):
        os.makedirs(os.path.join(dest_dir, "labels"))
    if not os.path.exists(os.path.join(dest_dir, "labels", "train")):
        os.makedirs(os.path.join(dest_dir, "labels", "train"))
    if not os.path.exists(os.path.join(dest_dir, "labels", "val")):
        os.makedirs(os.path.join(dest_dir, "labels", "val"))
    if not os.path.exists(os.path.join(dest_dir, "images", "test")):
        os.makedirs(os.path.join(dest_dir, "images", "test"))
    if not os.path.exists(os.path.join(dest_dir, "labels", "test")):
        os.makedirs(os.path.join(dest_dir, "labels", "test"))

    images = os.listdir(source_images_dir)
    if shuffle:
        if seed:
            random.seed(seed)
        random.shuffle(images)

    train_num = int(len(images) * train_ratio)
    val_num = int(len(images) * val_ratio)
    test_num = int(len(images) * test_ratio)
    train_images = images[:train_num]
    val_images = images[train_num : train_num + val_num]
    test_images = images[train_num + val_num : train_num + val_num + test_num]

    print("train_num:", train_num)
    print("val_num:", val_num)
    print("test_num:", test_num)

    for image in tqdm(train_images):
        shutil.copy(
            os.path.join(source_images_dir, image),
            os.path.join(dest_dir, "images", "train", image),
        )
        shutil.copy(
            os.path.join(source_labels_dir, image.replace(".png", ".txt")),
            os.path.join(dest_dir, "labels", "train", image.replace(".png", ".txt")),
        )
    for image in tqdm(val_images):
        shutil.copy(
            os.path.join(source_images_dir, image),
            os.path.join(dest_dir, "images", "val", image),
        )
        shutil.copy(
            os.path.join(source_labels_dir, image.replace(".png", ".txt")),
            os.path.join(dest_dir, "labels", "val", image.replace(".png", ".txt")),
        )
    for image in tqdm(test_images):
        shutil.copy(
            os.path.join(source_images_dir, image),
            os.path.join(dest_dir, "images", "test", image),
        )
        shutil.copy(
            os.path.join(source_labels_dir, image.replace(".png", ".txt")),
            os.path.join(dest_dir, "labels", "test", image.replace(".png", ".txt")),
        )


if __name__ == "__main__":

    dataset_split(
        source_images_dir="/raid/home-robot/gyzp/data/images/train",
        source_labels_dir="/raid/home-robot/gyzp/data/labels/train",
        dest_dir="/raid/home-robot/gyzp/yolo/data",
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        shuffle=True,
        seed=None,
    )

    print("Done.")
