import pickle
import json
from pathlib import Path
import os

# pkl_file = "cyw/datasets/place_dataset_debug/val/heuristic_agent_place_test/place_waypoint.pkl"
pkl_file = "cyw/datasets/place_dataset_debug/val/heuristic_agent_place/place_waypoint.pkl"
with open(pkl_file,"rb") as f:
    data = pickle.load(f)

pkl_dir = str(Path(pkl_file).resolve().parent)
json_file =  os.path.join(pkl_dir,"place_waypoint.json")
with open(json_file,"w") as f:
    json.dump(data,f,indent=2)

print("over")