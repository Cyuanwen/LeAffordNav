#!/bin/bash

cd "$(dirname "$0)"/..

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ./config/accelerate_single_config.yaml \
    ./LLaMA-Factory/src/train.py ./config/full_predict.yaml
