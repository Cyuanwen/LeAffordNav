"""
对比gt 和 模型检测结果，并记录检测不对（包括误识别、漏检）的 rgb,真实分割
将小物体和大物体的error分别记录下来（因为小物体和大物体分别用不同的模型）
目标物体？

保存路径格式：
record_root_path
├── goal
│   ├── error_labels  # 误识别pred标签, （每行一个物体：id name error_type) .txt
│   ├── images        # 原始图片rgb格式 .png 
│   └── labels        # gt标签, yolo格式 .txt 
│                     #  (每行一个物体: id x1 y1 x2 y2 x3 y3 ...，其中xy为归一化坐标)
└── receptacle
    ├── error_labels
    ├── images
    └── labels

错误检查：使用gt和pred的masks iou > threshold, 则认为检测正确

错误类型定义：
    漏检：gt有，pred没有 (error_labels: label_id obj_name missing)
    错检：iou < threshold, gt和pred都有 (error_labels: label_id obj_name wrong)
    多检：gt没有，pred有 (error_labels: label_id obj_name more)
"""

import json
import os
import string
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


class DetectionError:
    def __init__(
        self,
        record_root_path: str = "./detect_error_record",
        iou_threshold: float = 0.5,
    ):
        self.gt_id2name: Dict[int, str] = {}  # {gt_obj_id: obj_name}
        self.pred_id2name: Dict[int, str] = {}  # {pred_obj_id: obj_name}
        self.recep_name_list: List[str] = []  # 大物体名称列表
        self.name2label: Dict[str, int] = {}  # {obj_name: recode_label_id}
        # 使用gt cat map默认初始化上述4个变量
        self._init_config("projects/real_world_ovmm/configs/example_cat_map.json")

        self.record_root_path = record_root_path
        self.iou_threshold = iou_threshold

        self.goal_error_count = 0  # 小物体误识别计数
        self.receptacle_error_count = 0  # 大物体误识别计数

    def compare(
        self,
        pred_masks: np.ndarray,
        pred_class_dics: np.ndarray,
        gt_masks: np.ndarray,
        gt_class_dics: np.ndarray,
        rgb_image: np.ndarray,
        goal_id1_name: str,
    ):
        """
        Description:
            比较gt 和 模型检测结果，若检查错误，则记录错误信息
        Args:
            pred_masks: ndarray (pred_num, h, w) -> {0,1}
            pred_class_dics: ndarray (pred_num) -> class_id(int)
            gt_masks: ndarray (gt_num, h, w) -> {0,1}
            gt_class_dics: ndarray (gt_num) -> class_id(int)
        """
        # 设置goal_id1_name, 设置唯一一个小物体的id-name映射
        self.pred_id2name[1] = goal_id1_name
        self.gt_id2name[1] = goal_id1_name

        # 将gt和pred的class_id转换为class_name
        pred_class_names = [self.pred_id2name[class_id] for class_id in pred_class_dics]
        gt_class_names = [self.gt_id2name[class_id] for class_id in gt_class_dics]
        all_class_names = set(pred_class_names + gt_class_names)

        error_info: List[Dict[str, str]] = []  # 错误信息 {class_name, error_type}

        # 遍历所有出现的物体类别
        for class_name in all_class_names:
            # 获取gt和pred的mask
            gt_mask = gt_masks[gt_class_names == class_name]
            pred_mask = pred_masks[pred_class_names == class_name]

            # 划分错误类型
            if gt_mask.size == 0:
                # 多检
                # 多检大概率对应着存在漏检，因此这里不做记录
                # error_info.append({"class_name": class_name, "error_type": "more"})
                # self.record_error("more", class_name, gt_mask, pred_mask, rgb_image)
                pass
            elif pred_mask.size == 0:
                # 漏检
                error_info.append({"class_name": class_name, "error_type": "missing"})
            else:
                # 计算iou，判断是否为错检
                iou = self._compute_iou(gt_mask, pred_mask)
                if iou < self.iou_threshold:
                    # 错检
                    error_info.append({"class_name": class_name, "error_type": "wrong"})

        # 记录错误信息
        if len(error_info) > 0:
            # 分别记录大物体和小物体的错误信息
            goal_error_info = [
                info
                for info in error_info
                if info["class_name"] in self.recep_name_list
            ]
            receptacle_error_info = [
                info
                for info in error_info
                if info["class_name"] not in self.recep_name_list
            ]

            # 获取gt的标签(yolo格式)
            gt_labels = "".join(
                self._get_mask_contours_and_labels(gt_mask, class_name)[1]
                for class_name, gt_mask in zip(gt_class_names, gt_masks)
            )

            # 记录错误信息
            self._record_error_info(goal_error_info, class_name, rgb_image, gt_labels)
            self._record_error_info(
                receptacle_error_info, class_name, rgb_image, gt_labels
            )

    def _init_config(self, example_cat_map_path: str):
        """
        Description:
            初始化id-name映射和name-label映射
        """
        with open(example_cat_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj_category_to_obj_category_id = data[
            "obj_category_to_obj_category_id"
        ]  # {goal_name: goal_id}
        recep_category_to_recep_category_id = data[
            "recep_category_to_recep_category_id"
        ]  # {recep_name: recep_id}

        for recep_name, recep_id in recep_category_to_recep_category_id.items():
            self.gt_id2name[recep_id + 2] = recep_name
            self.pred_id2name[recep_id + 2] = recep_name
            self.recep_name_list.append(recep_name)
            self.name2label[recep_name] = recep_id

        for goal_name, goal_id in obj_category_to_obj_category_id.items():
            self.name2label[goal_name] = goal_id

    def _record_error_info(
        self,
        error_info: List[Dict[str, str]],
        class_name: str,
        rgb_image: np.ndarray,
        gt_labels: str,
    ):
        """
        Description:
            记录错误信息，保存rgb图片、gt标签、错误标签
        Args:
            error_info: 错误信息 {class_name, error_type}
            class_name: 物体类别
            rgb_image: rgb图片
            gt_labels: gt标签
        """
        if len(error_info) == 0:
            return

        # 判断是大物体还是小物体, 设置保存路径, 计数
        is_receptacle = class_name in self.recep_name_list
        if is_receptacle:
            error_count = self.receptacle_error_count
            self.receptacle_error_count += 1
            error_root_path = os.path.join(self.record_root_path, "receptacle")
        else:
            error_count = self.goal_error_count
            self.goal_error_count += 1
            error_root_path = os.path.join(self.record_root_path, "goal")

        # 保存记录文件名
        record_name = str(error_count)

        # 保存rgb图片
        error_image_path = os.path.join(
            error_root_path, "error_images", record_name + ".png"
        )
        os.makedirs(os.path.dirname(error_image_path), exist_ok=True)
        Image.fromarray(rgb_image, "RGB").save(error_image_path)

        # 保存gt标签
        gt_label_path = os.path.join(error_root_path, "labels", record_name + ".txt")
        os.makedirs(os.path.dirname(gt_label_path), exist_ok=True)
        with open(gt_label_path, "w", encoding="utf-8") as f:
            f.write(gt_labels)

        # 保存错误标签
        error_label_path = os.path.join(
            error_root_path, "error_labels", record_name + ".txt"
        )
        os.makedirs(os.path.dirname(error_label_path), exist_ok=True)
        with open(error_label_path, "w", encoding="utf-8") as f:
            for info in error_info:
                f.write(
                    str(self.name2label[info["class_name"]])
                    + " "
                    + info["class_name"]
                    + " "
                    + info["error_type"]
                    + " "
                    + gt_labels
                    + "\n"
                )

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Description:
            计算两个mask的iou
        Args:
            mask1: ndarray (h, w) -> {0,1}
            mask2: ndarray (h, w) -> {0,1}
        Returns:
            iou: iou值
        """
        assert mask1.shape == mask2.shape

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union != 0 else 0

        return iou

    def _get_mask_contours_and_labels(self, mask: np.ndarray, class_name: str):
        """
        Description:
            获取mask的轮廓
        Args:
            mask: ndarray (h, w) -> {0,1}
        Returns:
            contours: 轮廓
            label_str: yolo格式标签
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        label_id = self.name2label[class_name]

        valid_countour_count = 0
        label_str = ""
        img_x, img_y = mask.shape
        for contour in contours:
            if contour.shape[0] < 3:  # at least 3 points
                continue
            valid_countour_count += 1
            contour = contour.flatten().astype(np.float32)
            contour[0::2] /= img_y  # scale y
            contour[1::2] /= img_x  # scale x
            contour = contour.tolist()
            contour = ["{:.6f}".format(i) for i in contour]
            line = str(label_id) + " " + " ".join(contour) + "\n"

        return contours, label_str
