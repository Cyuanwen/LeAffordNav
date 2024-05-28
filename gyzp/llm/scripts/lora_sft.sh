#!/bin/bash

cd "$(dirname "$0")"/..

cuda_visible_devices=0,1,2,3
train_python_file_path=./LLaMA-Factory/src/train.py
accelerate_config_file_path=./config/accelerate_single_config.yaml
sft_config_file_path=./config/lora_sft.yaml


# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file examples/accelerate/single_config.yaml \
#     src/train.py examples/lora_multi_gpu/llama3_lora_sft.yaml

CUDA_VISIBLE_DEVICES=$cuda_visible_devices accelerate launch \
    --config_file $accelerate_config_file_path \
    $train_python_file_path $sft_config_file_path