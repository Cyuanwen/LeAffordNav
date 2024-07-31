# 评估place policy

# baseline 策略
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/train/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_nav_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=train \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# heuristic_gaze policy + initial place policy
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/train/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_gaze_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=train \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# heuristic_gaze policy + cyw place policy
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/train/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_gaze_place_cyw.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=train \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# baseline but cyw_place_policy
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/train/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_nav_place_cyw.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=train \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# val split

# heuristic_gaze policy + cyw place policy
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/val/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_gaze_place_cyw.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=val \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# baseline policy
python projects/habitat_ovmm/place_policy_eval.py \
    --data_dir cyw/datasets/place_dataset_debug/val/ \
    --baseline_config_path cyw/configs/agent/heuristic_agent_nav_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=val \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05