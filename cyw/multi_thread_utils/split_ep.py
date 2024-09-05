'''
对episode进行划分，以便 并行运行
'''
import json
from pathlib import Path
import os

splits = 10 # 分为 splits 份

if __name__ == "__main__":
    data_file = "cyw/datasets/place_dataset/val/episode_ids.json"
    data_dir = str(Path(data_file).resolve().parent)
    data_dir = os.path.join(data_dir,'split_episode')
    os.makedirs(data_dir,exist_ok=True)
    with open(data_file,'r') as f:
        episodes = json.load(f)
    per_split_num = len(episodes)//splits
    ep_num = 0
    for i in range(splits):
        if i != splits -1:
            episodes_i = episodes[i*per_split_num:(i+1)*per_split_num]
        else:
            episodes_i = episodes[i*per_split_num:]
        ep_num += len(episodes_i)
        # 保存数据
        with open(os.path.join(data_dir,f'episode_ids_{i}.json'),'w') as f:
            json.dump(episodes_i,f)
    print(f"initial ep num is {len(episodes)}, ep_num is {ep_num}")


