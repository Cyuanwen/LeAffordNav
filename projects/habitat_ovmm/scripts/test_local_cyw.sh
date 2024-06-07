#!/usr/bin/env bash

DOCKER_NAME="ovmm_baseline_submission_v4_cmd"
SPLIT="minival"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
      --split)
      shift
      SPLIT="${1}"
      shift
      ;;
    *)
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
      -v $(realpath ../../data):/home-robot-v1/home-robot/data \
      --runtime=nvidia \
      --gpus all \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "LOCAL_ARGS='habitat.dataset.split=${SPLIT}'" \
      ${DOCKER_NAME} 
