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
    --EXP_NAME_suffix "hueristic_gaze_cyw_place"

