'''
在结果文件中过滤出想要的episode，以便分析
'''
# "END.ovmm_find_recep_phase_success": 1.0,
# "END.ovmm_place_object_phase_success": 0.0

import json
from tqdm import tqdm

def judge(item):
    if item["END.ovmm_find_recep_phase_success"] == 1 and item["END.ovmm_place_object_phase_success"] == 0:
        return True
    else:
        return False

if __name__ == "__main__":
    episode_file = "datadump/results/eval_hssd_cyw_gtseg_print_img/episode_results.json"
    with open(episode_file,"r") as f:
        episode_results = json.load(f)
    episodes = []
    for episode, result in tqdm(episode_results.items()):
        if judge(result):
            episodes.append(episode)
    print(episodes)
    print(f"total length {len(episodes)}")

    
