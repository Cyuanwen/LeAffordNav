'''
将多个线程的数据合并起来
'''
import h5py
import os
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    data_root = "cyw/datasets/place_dataset/val/heuristic_agent_esc_yolo_place_cyw"
    thread_num = 10


    # """h5py文件"""
    # data_out = h5py.File(os.path.join(data_root,'data_out.hdf5'),"r+")
    # for thread in tqdm(range(thread_num)):
    #     h5py_file  = os.path.join(data_root,'multi_thread', f'data_out_{thread}.hdf5')
    #     file  = h5py.File(h5py_file, "r")
    #     # 遍历每个数据集
    #     for scene_name in file:
    #         scene_data = file[scene_name]
    #         # data_out.create_group(scene_name)
    #         for ep_name in scene_data:
    #             ep_data = scene_data[ep_name]
    #             # data_out.create_group(
    #             #     f"/{scene_name}/{ep_name}"
    #             # )
    #             for recep_name in ep_data:
    #                 recep_data = ep_data[recep_name]
    #                 scen_ep_recep_grp = data_out.create_group(f"/{scene_name}/{ep_name}/{recep_name}")
    #                 for name in recep_data:
    #                     data = recep_data[name][()]
    #                     if name == "start_top_down_map_s":
    #                         continue
    #                     if not name in scen_ep_recep_grp:
    #                         scen_ep_recep_grp.create_dataset(name, data=data)
    #                     else:
    #                         print('data repeat!')
    #     file.close()
    # data_out.flush()

    """pkl文件: [] 合并即可"""
    total_data = []
    for thread in tqdm(range(thread_num)):
        # cyw/datasets/place_dataset/val/heuristic_agent_esc_yolo_place_cyw/multi_thread/place_waypoint_6.pkl
        with open(os.path.join(data_root,'multi_thread',f'place_waypoint_{thread}.pkl'),'rb') as f:
            data = pickle.load(f)
        total_data += data
    # 写入文件
    out_pkl = os.path.join(data_root,"place_waypoint.pkl")
    print(f"save data to {out_pkl} **************")
    with open(out_pkl,'wb') as f:
        pickle.dump(total_data,f)


    

    
