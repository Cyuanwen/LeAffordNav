# train

# # rl policy
# python projects/habitat_ovmm/place_data_collection.py \
#     --data_dir cyw/datasets/place_dataset/train \
#     --baseline_config_path cyw/configs/agent/rl_agent_place.yaml \
#     --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
#     habitat.dataset.split=train +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
#     +habitat.task.measurements.top_down_map.meters_per_pixel=0.05


## heuristic_agent_place
python projects/habitat_ovmm/place_data_collection.py \
    --data_dir cyw/datasets/place_dataset/train \
    --baseline_config_path cyw/configs/agent/heuristic_agent_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=train +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05


# val
## heuristic_agent_place
python projects/habitat_ovmm/place_data_collection.py \
    --data_dir cyw/datasets/place_dataset/val \
    --baseline_config_path cyw/configs/agent/heuristic_agent_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=val \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05

# val debug数据（代码内部修改epnum=2）
python projects/habitat_ovmm/place_data_collection.py \
    --data_dir cyw/datasets/place_dataset_debug/val \
    --baseline_config_path cyw/configs/agent/heuristic_agent_place.yaml \
    --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml \
    habitat.dataset.split=val \
    +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
    +habitat.task.measurements.top_down_map.meters_per_pixel=0.05