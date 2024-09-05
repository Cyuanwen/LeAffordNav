#!/bin/bash

# 解析输入参数
SPLIT=$1
CUDA_DEVICE=$2
THREAD_NUM=$3

# 运行命令
# conda activate ovmm
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export HOME_ROBOT_ROOT=/raid/home-robot
# python projects/habitat_ovmm/place_data_collection.py \
#     --data_dir cyw/datasets/place_dataset_debug/$SPLIT/ \
#     --baseline_config_path cyw/configs/agent/heuristic_agent_esc_yolo_nav_place_cyw.yaml \
#     --env_config_path cyw/configs/debug_config/env/hssd_eval.yaml \
#     --thread_num $THREAD_NUM \
#     habitat.dataset.split=$SPLIT \
#     +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
#     +habitat/task.measurements.top_down_map.meters_per_pixel=0.05
# 奇怪，为啥上面这行代码不行？

# export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
# export HOME_ROBOT_ROOT=/raid/home-robot
# 检查是否包含 append 参数
if [[ $* == *append* ]]; then
    APPEND="--append"
    echo 'append data'
else
    APPEND=""
fi

python projects/habitat_ovmm/place_data_collection.py --data_dir cyw/datasets/place_dataset/$SPLIT/ --baseline_config_path cyw/configs/agent/heuristic_agent_esc_yolo_place_cyw.yaml --env_config_path cyw/configs/env/hssd_eval.yaml --thread_num $THREAD_NUM $APPEND habitat.dataset.split=$SPLIT +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map +habitat.task.measurements.top_down_map.meters_per_pixel=0.05 

# # example
# # export CUDA_VISIBLE_DEVICES=2,3 
# # bash cyw/scripts/place_data_collection_v1.sh val 1,2,3 0