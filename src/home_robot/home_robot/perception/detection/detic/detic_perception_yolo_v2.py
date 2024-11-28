# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
    由 src/home_robot/home_robot/perception/detection/detic/detic_perception.py 修改而来
    使用yolo识别大物体，detic识别小物体
    detic识别中：0 object, 1 start recep, 2 end recep
    yolo: 按照配置文件
    gt: 0 backgroud, 1 object, 2…… recep

    由于vocab变量外部代码也会使用，这里将yolo的结果映射到 detic vocab
    TODO: 让detic识别task related objects, rooms (rooms单独识别), yolo识别recep
    相比于 src/home_robot/home_robot/perception/detection/detic/detic_perception_yolo_v1.py： 使用detic单独识别recep, room, 并用yolo识别所有不相关的物体
'''

import argparse
import enum
import pathlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.utils import filter_depth, overlay_masks
# @cyw
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from home_robot.perception.wrapper import ROOMS
visualize = False
debug = False
# YOLO_EXTRA = ["sink","trunk","filing_cabinet","wardrobe"] #训练场景没有这三类数据，yolo识别不了
# 根据最新版yolo验证结果，增加几类
YOLO_EXTRA = ["sink","trunk","filing_cabinet","wardrobe","serving_cart","stand"] #训练场景没有这三类数据，yolo识别不了
# @gyzp
# sys.path.append(r"gyzp/utils/preception")
# from detect_error import DetectionErrorWrapper

sys.path.append(str(Path(__file__).resolve().parent.parent))
from detect_error import DetectionErrorWrapper

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "Detic/third_party/CenterNet2/")
)
from centernet.config import add_centernet_config  # noqa: E402

from home_robot.perception.detection.detic.Detic.detic.config import (  # noqa: E402
    add_detic_config,
)
from home_robot.perception.detection.detic.Detic.detic.modeling.text.text_encoder import (  # noqa: E402
    build_text_encoder,
)
from home_robot.perception.detection.detic.Detic.detic.modeling.utils import (  # noqa: E402
    reset_cls_test,
)

BUILDIN_CLASSIFIER = {
    "lvis": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}

YOLO_CHECKPOINT_FILE = "cyw/data/models/perception/train2/weights/best.pt"
RECORD_PATH_FILE = 'gyzp/output/detect_error/yolo_only'

def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


class DeticPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        vocabulary="coco",
        custom_vocabulary="",
        checkpoint_file=None,
        yolo_checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = None,
        yolo_confidence_threshold: Optional[float] = None,
        yolo_main: bool = False,
        log_detect: bool = False,
        log_dir: Optional[str]=None,
        add_rooms:bool=False,
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            vocabulary: currently one of "coco" for indoor coco categories or "custom"
             for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
            yolo_main: use yolo as main detector
            log: log detect info, will collect error detect data
            add_rooms: whether to detect rooms
        """
        self.verbose = verbose
        if config_file is None:
            config_file = str(
                Path(__file__).resolve().parent
                / "Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
            )
        if checkpoint_file is None:
            checkpoint_file = str(
                Path(__file__).resolve().parent
                / "Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
            )
        if self.verbose:
            print(
                f"Loading Detic with config={config_file} and checkpoint={checkpoint_file}"
            )

        string_args = f"""
            --config-file {config_file} --vocabulary {vocabulary}
            """

        if vocabulary == "custom":
            assert custom_vocabulary != ""
            string_args += f""" --custom_vocabulary {custom_vocabulary}"""

        string_args += f""" --opts MODEL.WEIGHTS {checkpoint_file}"""

        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""

        string_args = string_args.split()
        args = get_parser().parse_args(string_args)
        cfg = setup_cfg(
            args, verbose=verbose, confidence_threshold=confidence_threshold
        )

        assert vocabulary in ["coco", "custom"]
        if args.vocabulary == "custom":
            if "__unused" in MetadataCatalog.keys():
                MetadataCatalog.remove("__unused")
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(",")
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        elif args.vocabulary == "coco":
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]
            self.categories_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
                60: 6,  # table
                69: 7,  # oven
                71: 8,  # sink
                72: 9,  # refrigerator
                73: 10,  # book
                74: 11,  # clock
                75: 12,  # vase
                41: 13,  # cup
                39: 14,  # bottle
            }
        self.num_sem_categories = len(self.categories_mapping)

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE
        self.predictor = DefaultPredictor(cfg)
    
        if type(classifier) == pathlib.PosixPath:
            classifier = str(classifier)
        reset_cls_test(self.predictor.model, classifier, num_classes)

        # 设置单独识别房间的模型
        # @cyw
        self.add_rooms = add_rooms
        if self.add_rooms:
            self.room_predictor = DefaultPredictor(cfg)
        
        if args.vocabulary == "custom":
            self.room_classes = ["."] + ROOMS + ["other"]
            room_classifier = get_clip_embeddings(self.room_classes)
        else:
            raise NotImplementedError(
                "Detic does not have support rooms coco vocab"
            )
        num_room_classes = len(self.room_classes)
        reset_cls_test(self.room_predictor.model, room_classifier, num_room_classes)

        # 加载yolo 模型
        if yolo_checkpoint_file is None:
            yolo_checkpoint_file = YOLO_CHECKPOINT_FILE
        if self.verbose:
            print(
                f"Loading yolov8 with config={config_file} and checkpoint={checkpoint_file}"
            )
        self.yolo_model = YOLO(model = yolo_checkpoint_file,verbose=self.verbose)
        self.sem_gpu_id = sem_gpu_id
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.yolo_names = {
                            0: "bathtub",
                            1: "bed",
                            2: "bench",
                            3: "cabinet",
                            4: "chair",
                            5: "chest_of_drawers",
                            6: "couch",
                            7: "counter",
                            8: "filing_cabinet",
                            9: "hamper",
                            10: "serving_cart",
                            11: "shelves",
                            12: "shoe_rack",
                            13: "sink",
                            14: "stand",
                            15: "stool",
                            16: "table",
                            17: "toilet",
                            18: "trunk",
                            19: "wardrobe",
                            20: "washer_dryer",
        }
        yolo_name2id = {name:idx for idx,name in self.yolo_names.items()}
        self.yolo_extra_id = [yolo_name2id[name] for name in YOLO_EXTRA ]
        self.yolo_mian = yolo_main
        self.log_detect = log_detect
        # @gyzp error dectection
        if self.log_detect:
            self.error_detector = DetectionErrorWrapper(record_root_path=log_dir, verbose=True)
            print(f"the error detect result will be saved in {log_dir}")
        

    def reset_vocab(self, new_vocab: List[str], 
        simple_vocab:Optional[List[str]] = None,
        vocab_type="custom"
        ):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            simple_vocab: simple task related vocab
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        if self.verbose:
            print(f"Resetting vocabulary to {new_vocab}")
        MetadataCatalog.remove("__unused")
        if vocab_type == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = new_vocab
            # classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.name2id = {name:idx for idx,name in enumerate(self.metadata.thing_classes)}
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        else:
            raise NotImplementedError(
                "Detic does not have support for resetting from custom to coco vocab"
            )
        self.num_sem_categories = len(self.categories_mapping)

        self.yolo_mapping = {
            i:self.name2id[name] for i, name in self.yolo_names.items() \
                if name in self.name2id.keys()
        }

        self.room_mapping = {
                i: self.name2id[self.room_classes[i]] for i in range(len(self.room_classes))
            }
        # @cyw
        if simple_vocab is None:
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            num_classes = len(self.metadata.thing_classes)
            self.detic_mapping = None #不需要映射
        else:
            classifier = get_clip_embeddings(simple_vocab)
            num_classes = len(simple_vocab)
            self.detic_mapping = {
                i:self.name2id[simple_vocab[i]] for i in range(len(simple_vocab))
            }
        reset_cls_test(self.predictor.model, classifier, num_classes)
    
    # @cyw
    def set_episode_key(self,current_episode_key):
        '''
            设置log记录目录
        '''
        if self.log_detect:
            self.error_detector.set_record_dir(current_episode_key)
    
    def get_yolo_results(self,yolo_pred):
        '''
            将yolo_pred进行过滤，并重新映射：过滤出metadata.thing_class里面的类别，并将class_idcs映射到总体vocab
            NOTE 这里只返回yolo识别到的 vocab 指定的物体类别
        '''
        if yolo_pred.masks is not None:
            yolo_masks = yolo_pred.masks.data.cpu().numpy()
            yolo_class_idcs = yolo_pred.boxes.cls.cpu().numpy()
            yolo_scores = yolo_pred.boxes.conf.cpu().numpy()
            yolo_index = []
            class_idcs = []
            for idx, yolo_class_id in enumerate(yolo_class_idcs):
                if yolo_class_id in self.yolo_mapping:
                    yolo_index.append(idx)
                    class_idcs.append(self.yolo_mapping[yolo_class_id])
            if len(yolo_index)!=0:
                masks = yolo_masks[yolo_index,:,:]
                scores = yolo_scores[yolo_index]
                return masks,class_idcs,scores
        return None,None,None
    
    def mapping_class_idcs(self,class_idcs,mapping):
        '''将 class_idcs 根据 mapping 中的对应关系进行映射
        '''
        if mapping is None:
            # 不需要映射
            return class_idcs
        new_class_ids = []
        for idx in class_idcs:
            new_class_ids.append(mapping[int(idx)])
        return new_class_ids

    def filter_class_ids(self,masks,class_idcs,scores,keep_ids):
        '''过滤一些类别id
            Arguments:
                masks:
                class_ids:原本类别ids
                scores:
                keep_ids:要保留的ids
            Return:
                masks,class_ids,scores
        '''
        index = []
        new_class_ids = []
        for idx, class_id in enumerate(class_idcs):
            if class_id in keep_ids:
                index.append(idx)
                new_class_ids.append(class_id)
        masks_new = masks[index,:,:]
        scores_new = scores[index]
        return masks_new,new_class_ids,scores_new

    # @cyw
    def combine_results(self,pred,yolo_pred,yolo_main,height, width):
        '''将detic的预测结果和yolo的预测结果结合起来
            Arguments:
                pred: detic 预测结果，需将id映射到全局vocab
                yolo_pred: yolo 预测结果，已将id映射到全局vocab
                yolo_main: True: yolo的结果为主导，detic补充；False: detic 结果为主导，yolo补充
                room_pred:预测的房间结果
        '''
        # Sort instances by mask size
        masks = pred["instances"].pred_masks.cpu().numpy()# n,w,h
        class_idcs = pred["instances"].pred_classes.cpu().numpy() # n
        scores = pred["instances"].scores.cpu().numpy() # n

        # 将结果合并起来
        if yolo_main:
            yolo_extra_ids = [self.yolo_mapping[idx] for idx in self.yolo_extra_id if idx in self.yolo_mapping]
            yolo_keep_ids = [idx for idx in self.yolo_mapping.values() if idx not in yolo_extra_ids]
            if self.detic_mapping is not None:
                detic_keep_ids = [idx for idx in self.detic_mapping.values() if idx not in yolo_keep_ids]
            else:
                detic_keep_ids = [idx for idx in self.categories_mapping.values() if idx not in yolo_keep_ids]
        else:
            if self.detic_mapping is not None:
                detic_keep_ids = [idx for idx in self.detic_mapping.values()]
                yolo_keep_ids = [idx for idx in self.yolo_mapping.values() if idx not in detic_keep_ids]
            else:
                detic_keep_ids = self.categories_mapping.values()
                yolo_keep_ids = []
        # yolo 结果
        yolo_masks,yolo_class_idcs,yolo_scores = self.get_yolo_results(yolo_pred)
        # 进行过滤
        masks,class_idcs,scores = self.filter_class_ids(masks,class_idcs,scores,detic_keep_ids)
        if yolo_masks is not None and len(yolo_keep_ids)!=0:
            yolo_masks,yolo_class_idcs,yolo_scores = self.filter_class_ids(yolo_masks,yolo_class_idcs,yolo_scores,yolo_keep_ids)
            masks = np.concatenate((masks,yolo_masks),axis=0)
            class_idcs = np.concatenate((class_idcs,yolo_class_idcs),axis=0)
            scores = np.concatenate((scores,yolo_scores),axis=0)
        return masks, class_idcs, scores


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
            yolo_main: if true 只用detic的小物体识别结果，else 用yolo补充detic的识别

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        if draw_instance_predictions:
            raise NotImplementedError
        if isinstance(obs.rgb, torch.Tensor):
            rgb = obs.rgb.numpy()
        elif isinstance(obs.rgb, np.ndarray):
            rgb = obs.rgb
        else:
            raise ValueError(
                f"Expected obs.rgb to be a numpy array or torch tensor, got {type(obs.rgb)}"
            )
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = obs.depth
        height, width, _ = image.shape
        pred = self.predictor(image)
        if self.add_rooms:
            room_pred = self.room_predictor(image)

        # 标签重映射
        if self.detic_mapping is not None:
            pred["instances"].pred_classes = torch.tensor(self.mapping_class_idcs(pred["instances"].pred_classes.cpu().numpy(),self.detic_mapping),device=f"cuda:{self.sem_gpu_id}")
        if self.add_rooms:
            room_pred["instances"].pred_classes = torch.tensor(self.mapping_class_idcs(room_pred["instances"].pred_classes.cpu().numpy(),self.room_mapping),device=f"cuda:{self.sem_gpu_id}")

        # 读取yolo预测
        yolo_pred = self.yolo_model(source=image,conf=self.yolo_confidence_threshold,device=f"cuda:{self.sem_gpu_id}")
        yolo_pred = yolo_pred[0]

        if obs.task_observations is None:
            obs.task_observations = {}

        if visualize:
            # 可视化detic结果
            visualizer = Visualizer(
                image[:, :, ::-1], self.metadata, instance_mode=self.instance_mode
            )
            visualization = visualizer.draw_instance_predictions(
                predictions=pred["instances"].to(self.cpu_device)
            ).get_image()
            if self.add_rooms:
                room_visualization = visualizer.draw_instance_predictions(
                    predictions=room_pred["instances"].to(self.cpu_device)
                ).get_image()
            else:
                room_visualization = np.zeros_like(visualization)
            # 可视化 yolo结果
            im_bgr = yolo_pred.plot()  # BGR-order numpy array
            # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            im_rgb = im_bgr[..., ::-1]

            # 将三张图片拼接在一起
            combined_image = cv2.hconcat([visualization, room_visualization, im_rgb])

            # 在窗口中显示拼接后的图片
            cv2.imshow('Combined Images', combined_image)
            cv2.waitKey()
        
        masks, class_idcs, scores = self.combine_results(pred,yolo_pred,self.yolo_mian,height, width)
        # 单独获取room_predict
        if self.add_rooms:
            room_masks = room_pred["instances"].pred_masks.cpu().numpy()# n,w,h
            room_class_idcs = room_pred["instances"].pred_classes.cpu().numpy() # n
            room_scores = room_pred["instances"].scores.cpu().numpy() # n
            if debug:
                room_names = [self.metadata.thing_classes[idx] for idx in room_class_idcs ]
                print(f"the room is {room_names}")

        # @gyzp : detect error
        if self.log_detect:
            # 分析yolo识别的容器
            g_pred_masks = yolo_pred.masks.data.cpu().numpy() if yolo_pred.masks is not None else None
            g_pred_class_dics = yolo_pred.boxes.cls.cpu().numpy() + 2 if yolo_pred.boxes.cls is not None else None
            self.error_detector(obs = obs, pred_masks = g_pred_masks, pred_class_dics = g_pred_class_dics, goal_id1_name = self.metadata.thing_classes[1])
            # 分析detic识别的小物体
            detic_masks = pred["instances"].pred_masks.cpu().numpy()# n,w,h
            detic_class_idcs = pred["instances"].pred_classes.cpu().numpy() # n
            self.error_detector(obs = obs, pred_masks = detic_masks, pred_class_dics = detic_class_idcs, goal_id1_name = self.metadata.thing_classes[1],target_obj=["goal"])


        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )
            if self.add_rooms:
                room_masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in room_masks]
            )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))
        if self.add_rooms:
            room_semantic_map, room_instance_map = overlay_masks(room_masks, room_class_idcs, (height, width))
        # if visualize:
        #     sem_map_visulization = visualizer.draw_sem_seg(sem_seg=semantic_map)
        # 调用不了

        obs.semantic = semantic_map.astype(int)
        obs.instance = instance_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = class_idcs
        obs.task_observations["instance_scores"] = scores
        obs.task_observations["semantic_frame"] = None
        if self.add_rooms and len(room_class_idcs)!=0:
            obs.task_observations["room_semantic"] = room_semantic_map.astype(int)
        else:
            obs.task_observations["room_semantic"] = None
        return obs


def setup_cfg(
    args, verbose: bool = False, confidence_threshold: Optional[float] = None
):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    if confidence_threshold is None:
        confidence_threshold = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    if verbose:
        print("[DETIC] Confidence threshold =", confidence_threshold)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    # Fix cfg paths given we're not running from the Detic folder
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
        Path(__file__).resolve().parent / "Detic" / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
