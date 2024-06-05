from typing import Optional,List
from ultralytics import YOLO
from PIL import Image

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.utils import filter_depth, overlay_masks

import numpy as np
import torch
import cv2

CHECKPOINT_FILE = "/raid/cyw/detection/ovmm_data/runs/segment/train/weights/best.pt"

class YOLOPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = None,
    ):
        '''load trained yolov8 model for inference
        
        Arguments:
            config_file: path to model config
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information     
        '''
        self.verbose = verbose
        if checkpoint_file is None:
            checkpoint_file = CHECKPOINT_FILE
        if self.verbose:
            print(
                f"Loading yolov8 with config={config_file} and checkpoint={checkpoint_file}"
            )
        self.model = YOLO(model = checkpoint_file,verbose=self.verbose)
        self.sem_gpu_id = sem_gpu_id
        self.confidence_threshold = confidence_threshold
    
    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        pass

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        if isinstance(obs.rgb, torch.Tensor):
            # rgb = obs.rgb.numpy()
            pass
            # TODO reshape
        elif isinstance(obs.rgb, np.ndarray):
            rgb = obs.rgb
        else:
            raise ValueError(
                f"Expected obs.rgb to be a numpy array or torch tensor, got {type(obs.rgb)}"
            )
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = obs.depth
        height, width, _ = image.shape
        pred = self.model(source=image,conf=self.confidence_threshold,device=f"cuda:{self.sem_gpu_id}")
        pred = pred[0]
        if obs.task_observations is None:
            obs.task_observations = {}
        if draw_instance_predictions:
            im_bgr = pred.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            obs.task_observations["semantic_frame"] = im_rgb
        else:
            obs.task_observations["semantic_frame"] = None
        # Sort instances by mask size
        if pred.masks is not None:
            masks = pred.masks.data.cpu().numpy()
            class_idcs = pred.boxes.cls.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
        else:
            masks = np.zeros((1,height, width))
            class_idcs = [0] # 0实际对应的bathub,但这里mask和scores都为0
            scores = [0] 


        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        obs.semantic = semantic_map.astype(int)
        obs.instance = instance_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = class_idcs
        obs.task_observations["instance_scores"] = scores

        return obs

    # 测试代码用，测试完毕可删除
    def predict2(
        self,
        rgb,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        # if isinstance(rgb, torch.Tensor):
        #     # rgb = obs.rgb.numpy()
        #     pass
        #     # TODO reshape
        # elif isinstance(rgb, np.ndarray):
        #     rgb = rgb
        # else:
        #     raise ValueError(
        #         f"Expected obs.rgb to be a numpy array or torch tensor, got {type(obs.rgb)}"
        #     )
        # image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = rgb
        height, width, _ = image.shape
        pred = self.model(source=image,conf=self.confidence_threshold,device=f"cuda:{self.sem_gpu_id}")
        pred = pred[0]
        if draw_instance_predictions:
            im_bgr = pred.plot()  # BGR-order numpy array
            # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            im_rgb = cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
            
        # Sort instances by mask size
        masks = pred.masks.data.cpu().numpy()
        class_idcs = pred.boxes.cls.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()
        
        # if depth_threshold is not None and depth is not None:
        #     masks = np.array(
        #         [filter_depth(mask, depth, depth_threshold) for mask in masks]
        #     )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        # obs.semantic = semantic_map.astype(int)
        # obs.instance = instance_map.astype(int)
        # if obs.task_observations is None:
        #     obs.task_observations = dict()
        # obs.task_observations["instance_map"] = instance_map
        # obs.task_observations["instance_classes"] = class_idcs
        # obs.task_observations["instance_scores"] = scores

        return None