import numpy as np
from tqdm import tqdm
import json
import os
import pickle

TEMP_DIR = "cyw/temp"

def str2array(str_data):
    # 去除字符串中的方括号和空格，并将剩余的数字转换为浮点数
    data_list = [float(x) for x in str_data.strip('[]').split()]

    # 将列表转换为 NumPy 数组
    data_array = np.array(data_list)
    return data_array


def get_replay_set(data_A, data_B):
    '''
        找出 data_A 失败，但是data_B成功的place 策略设置，包括
        episode, recep position, waypoint position
        记录为pickle形式，以便于复现结果，找出失败原因
        data数据形式： "scene_106366353_174226695ep_31039_[-11.09816   0.7747   -5.64654]_[-9.26784  0.16824 -5.25978]": 1,
    '''
    replay_data = {}
    for item_A, item_B in tqdm(zip(data_A,data_B)):
        if not item_A == item_B:
            break
        if data_A[item_A] == 0 and data_B[item_B] ==1:
            data_keys = item_A.split("_")
            episode = data_keys[-3]
            recep_pos = data_keys[-2]
            waypoint = data_keys[-1]
            if episode not in replay_data:
                replay_data[episode] = {recep_pos:[waypoint]}
            elif recep_pos not in replay_data[episode]:
                replay_data[episode][recep_pos] = [waypoint]
            else:
                replay_data[episode][recep_pos].append(waypoint)
    return replay_data

def get_set(data_A,fail=True):
    '''
        找出data_A中失败的例子，找找原因
        fail: 如果是True,找出失败的例子，否则，找出成功的例子
    '''
    replay_data = {}
    for item_A in tqdm(data_A):
        if fail:
            condition = 0
        else:
            condition =1
        if data_A[item_A] == condition:
            data_keys = item_A.split("_")
            episode = data_keys[-3]
            recep_pos = data_keys[-2]
            waypoint = data_keys[-1]
            if episode not in replay_data:
                replay_data[episode] = {recep_pos:[waypoint]}
            elif recep_pos not in replay_data[episode]:
                replay_data[episode][recep_pos] = [waypoint]
            else:
                replay_data[episode][recep_pos].append(waypoint)
    return replay_data


if __name__ == "__main__":
    compare = False
    fail =True
    data_A_file = "cyw/datasets/place_dataset_debug/train/heuristic_agent_nav_place_cyw/success.json"
    data_B_file = "cyw/datasets/place_dataset_debug/train/heuristic_agent_nav_place/success.json"
    with open(data_A_file,"r") as f:
        data_A = json.load(f)
    with open(data_B_file,"r") as f:
        data_B = json.load(f)
    if compare:
        replay_data = get_replay_set(data_A,data_B)
        file_A_policy = data_A_file.split("/")[-2]
        file_B_policy = data_B_file.split("/")[-2]
        save_dir = os.path.join(TEMP_DIR,f"{file_A_policy}_vs_{file_B_policy}.pkl")
        with open(save_dir,"wb") as f:
            pickle.dump(replay_data,f)
    else:
        replay_data = get_set(data_A,fail=fail)
        file_A_policy = data_A_file.split("/")[-2]
        if fail:
            subfix = 'fail'
        else:
            subfix = "success"
        save_dir = os.path.join(TEMP_DIR,f"{file_A_policy}_{subfix}.pkl")
        with open(save_dir,"wb") as f:
            pickle.dump(replay_data,f)