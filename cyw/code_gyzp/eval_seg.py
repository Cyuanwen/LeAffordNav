import os
from typing import *

import cv2
import numpy as np
from tqdm import tqdm


def read_labels_from_file(file_path):
    """
    description: 从文件中读取分割标签
    param: file_path: 标签文件路径
    return: labels: {"id": [polygon1, polygon2, ...]}
    note: 标签文件格式每行表示一个多边形，第一个数字为label_id，后面的数字为多边形的归一化顶点坐标。
    即已经采集的yolo标签文件格式。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels = dict()  # "id" : [polygon1, polygon2, ...]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            label_id, *polygon_vertex = line.split(" ")
            if label_id not in labels:
                labels[label_id] = []
            labels[label_id].append(
                np.array(polygon_vertex, dtype=np.float32).reshape(-1, 2)
            )
        return labels


def get_iou(
    pred_polygons: List[np.ndarray],
    gt_polygons: List[np.ndarray],
    image_size=(480, 640),
    info=None,
):
    """
    description: 计算两个多边形的iou
    param: pred_polygons size: (polygon_num, single_polygon_vertex_num, 2)
    param: gt_polygons size: (polygon_num, single_polygon_vertex_num, 2)
    return: iou
    """
    pred_masks = np.zeros(image_size, dtype=np.uint8)
    gt_masks = np.zeros(image_size, dtype=np.uint8)

    # 传入的多边形为归一化之后的01坐标，需要乘以图片尺寸
    pred_polygons = [polygon * image_size for polygon in pred_polygons]
    gt_polygons = [polygon * image_size for polygon in gt_polygons]

    for polygon in gt_polygons:
        cv2.fillConvexPoly(gt_masks, polygon.astype(np.int32), 1)

    for polygon in pred_polygons:
        cv2.fillConvexPoly(pred_masks, polygon.astype(np.int32), 1)

    intersection = np.logical_and(pred_masks, gt_masks)
    union = np.logical_or(pred_masks, gt_masks)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else -1
    return iou


def calc_confusion_matrix(counts: Dict[str, int]):
    """
    description: 根据TP, FP, FN计算precision, recall, f1
    param: counts: {"TP": 0, "FP": 0, "FN": 0}
    return: precision, recall, f1, confusion_matrix
    """
    TP = counts["TP"]
    FP = counts["FP"]
    FN = counts["FN"]

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1, np.array([[TP, FP], [FN, 0]])


def eval_seg(pred_labels_dir, gt_labels_dir, iou_threshold=0.5):
    """
    description: 计算分割任务的评价指标
    param: pred_labels_dir: 预测标签目录
    param: gt_labels_dir: 真实标签目录
    param: iou_threshold: iou阈值, 大于该阈值则为TP
    return: None

    note: 标签文件格式每行表示一个多边形，第一个数字为label_id，后面的数字为多边形的归一化顶点坐标。
    例如：0 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1
    以上表示一个label_id为0的多边形，四个顶点分别为(0.1, 0.1), (0.1, 0.9), (0.9, 0.9), (0.9, 0.1)。
    """

    pred_files = os.listdir(pred_labels_dir)
    gt_files = os.listdir(gt_labels_dir)

    assert len(pred_files) == len(gt_files), "predict and gt files not match"

    # 对于每个标签，按照iou阈值计算，若iou大于阈值则为TP，否则为FP或FN
    confusions = dict()  # "label_id": {"TP": 0, "FP": 0, "FN": 0}

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files)):
        # print(f"pred_file: {pred_file}, gt_file: {gt_file}")

        pred_labels = read_labels_from_file(os.path.join(pred_labels_dir, pred_file))
        gt_labels = read_labels_from_file(os.path.join(gt_labels_dir, gt_file))

        pred_label_ids = set(pred_labels.keys())
        gt_label_ids = set(gt_labels.keys())

        label_ids = list(pred_label_ids | gt_label_ids)

        for label_id in label_ids:
            pred_polygons = pred_labels.get(label_id, [])
            gt_polygons = gt_labels.get(label_id, [])
            iou = get_iou(pred_polygons, gt_polygons)
            if iou == -1:
                continue

            if label_id not in confusions:
                confusions[label_id] = {"TP": 0, "FP": 0, "FN": 0}

            if iou > iou_threshold:
                confusions[label_id]["TP"] += 1
            else:
                if len(pred_polygons) > 0:
                    confusions[label_id]["FP"] += 1
                if len(gt_polygons) > 0:
                    confusions[label_id]["FN"] += 1

    # 对每个label_id计算各自的precision, recall, f1
    print("\n========= summary for each label =========")
    sorted_label_ids = sorted(confusions.keys(), key=lambda x: int(x))
    for label_id in sorted_label_ids:
        confusion = confusions[label_id]
        precision, recall, f1, confusion_matrix = calc_confusion_matrix(confusion)
        print(
            f"label_id: {label_id:<10} precision: {precision:<10.4f} recall: {recall:<10.4f} f1: {f1:<10.4f} (TP: {confusion_matrix[0, 0]}, FP: {confusion_matrix[0, 1]}, FN: {confusion_matrix[1, 0]}, Sum: {confusion_matrix.sum()})"
        )

    # 计算总的precision, recall, f1
    print("\n========= summary for all labels =========")
    TP = sum([confusion["TP"] for confusion in confusions.values()])
    FP = sum([confusion["FP"] for confusion in confusions.values()])
    FN = sum([confusion["FN"] for confusion in confusions.values()])
    precision, recall, f1, confusion_matrix = calc_confusion_matrix(
        {"TP": TP, "FP": FP, "FN": FN}
    )
    print(
        f"precision: {precision:<10.4f} recall: {recall:<10.4f} f1: {f1:<10.4f} (TP: {confusion_matrix[0, 0]}, FP: {confusion_matrix[0, 1]}, FN: {confusion_matrix[1, 0]}, Sum: {confusion_matrix.sum()})"
    )


if __name__ == "__main__":
    pred_labels_dir = "./example data/labels/gt"
    gt_labels_dir = "./example data/labels/pred"
    iou_threshold = 0.9
    eval_seg(pred_labels_dir, gt_labels_dir, iou_threshold)
