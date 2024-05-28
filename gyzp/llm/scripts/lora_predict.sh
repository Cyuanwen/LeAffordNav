#!/bin/bash

cd "$(dirname "$0")"/..

cuda_visible_devices=0,1,2,3
train_python_file_path=./LLaMA-Factory/src/train.py
accelerate_config_file_path=./config/accelerate_single_config.yaml
predict_config_file_path=./config/lora_predict.yaml


CUDA_VISIBLE_DEVICES=$cuda_visible_devices accelerate launch --main_process_port 1234 \
    --config_file $accelerate_config_file_path \
    $train_python_file_path $predict_config_file_path 