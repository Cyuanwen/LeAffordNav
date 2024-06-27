#!/bin/bash

# pip install -U huggingface-hub

# set environment variable for the HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# Change directory to the script's directory
cd "$(dirname "$0")"/..

# Download the model
huggingface-cli download --resume-download Qwen/Qwen2-7B --local-dir ./temp/models/Qwen2-7B