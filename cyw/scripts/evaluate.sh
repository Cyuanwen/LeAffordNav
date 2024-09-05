## baseline
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path "cyw/configs/env/hssd_eval_print_img.yaml" --id_file "cyw/data/exp/random_num_100.json"

## baseline + gt seg
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path "cyw/configs/env/hssd_eval_gtseg_print_img.yaml" --id_file "cyw/data/exp/random_num_100.json"

## esc
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py --baseline_config_path "cyw/configs/agent/heuristic_agent_esc.yaml" --env_config_path "cyw/configs/env/hssd_eval_print_img.yaml" --id_file "cyw/data/exp/random_num_100.json" --EXP_NAME_suffix "esc"

## gt seg + hueristic gaze + cyw place
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py \
    --agent_type "gaze_place" \
    --baseline_config_path "cyw/configs/agent/gaze_place_cyw_agent.yaml" \
    --env_config_path "cyw/configs/env/hssd_eval_gtseg.yaml" \
    --id_file "cyw/data/exp/random_num_100.json" \
    --EXP_NAME_suffix "hueristic_gaze_cyw_place_v2"
# EXP_NAME_suffix + _v2 是因为 place 改过一个版本

## gt seg + hueristic nav + cyw place
python projects/habitat_ovmm/eval_baselines_agent.py \
    --agent_type "gaze_place" \
    --baseline_config_path "cyw/configs/agent/nav_place_cyw_agent.yaml" \
    --env_config_path "cyw/configs/env/hssd_eval_gtseg.yaml" \
    --id_file "cyw/data/exp/random_num_100.json" \
    --EXP_NAME_suffix "hueristic_gaze_cyw_place_v2"


## esc_yolo_detic + hueristic gaze + cyw place
export CUDA_VISIBLE_DEVICES=1,2,3
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py \
    --agent_type "gaze_place" \
    --baseline_config_path "cyw/configs/agent/esc_yolo_gaze_place_cyw_agent.yaml" \
    --env_config_path "cyw/configs/env/hssd_eval.yaml" \
    --id_file "cyw/data/exp/random_num_100.json" \
    --EXP_NAME_suffix "hueristic_gaze_cyw_place_v2"
# EXP_NAME_suffix + _v2 是因为 place 改过一个版本

## esc_yolo_detic + nav + cyw place
export HOME_ROBOT_ROOT=/raid/home-robot
export CUDA_VISIBLE_DEVICES=2,3
export HOME_ROBOT_ROOT=/raid/home-robot
conda activate ovmm
python projects/habitat_ovmm/eval_baselines_agent.py \
    --agent_type "gaze_place" \
    --baseline_config_path "cyw/configs/agent/esc_yolo_nav_place_cyw_agent.yaml" \
    --env_config_path "cyw/configs/env/hssd_eval.yaml" \
    --id_file "cyw/data/exp/random_num_100.json" \
    --EXP_NAME_suffix "hueristic_gaze_cyw_place_v3"

