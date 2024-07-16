import pandas as pd
import json
import numpy as np

assign_idx = "cyw/datasets/place_dataset_v1/train/heuristic_agent_place/assigned_idx.csv"
ep_idx = "cyw/datasets/place_dataset_v1/train/episode_ids.json"

assign_data = pd.read_csv(assign_idx)
with open(ep_idx,"r") as f:
    ep_idx = json.load(f)

assign_ep_idx = np.array(assign_data['episode_id'])
assign_ep_idx = np.unique(assign_ep_idx)
max_episode_idx = 0
for assign_ep_id in assign_ep_idx:
    index = ep_idx.index(str(assign_ep_id))
    if index > max_episode_idx:
        max_episode_idx = index

max_ep = ep_idx[max_episode_idx]
print(f"max_episode is {max_ep}")
