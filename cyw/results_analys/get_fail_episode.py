'''
从结果文件中，找出因为各种原因失败的轨迹，以便于复现这些轨迹，找找原因
'''
import json
import os
TEMP_DIR = "cyw/temp"

def get_episodes(data,condition_fn):
    episodes = []
    for key in data:
        item = data[key]
        if condition_fn(item):
            episodes.append(key.split("_")[-1])
    return episodes

def place_fail(item):
    '''
        "PLACE.ovmm_find_recep_phase_success": 1.0,
        "PLACE.ovmm_place_object_phase_success": 0.0,
    '''
    if item['END.ovmm_find_recep_phase_success'] and not item['END.ovmm_place_object_phase_success']:
        return True
    else:
        return False

def pick_fail(item):
    '''
        "END.ovmm_find_object_phase_success": 1.0,
        "END.ovmm_pick_object_phase_success": 1.0,
    '''
    return item['END.ovmm_find_object_phase_success']  and not item['END.ovmm_pick_object_phase_success']

if __name__ == "__main__":
    result_file = "datadump/results/eval_hssd_cyw_esc_yolo_main/episode_results.json"
    with open(result_file,"r") as f:
        results = json.load(f)
    episodes = get_episodes(results, pick_fail)
    file_name = result_file.split("/")[-2]
    file_name = f"{file_name}_pick_fial.json"
    with open(os.path.join(TEMP_DIR,file_name),"w") as f:
        json.dump(episodes,f)

