'''
对数据进行后处理
目前主要是把pcd 中的 nan 数据替换为0
'''
import os
import h5py
import numpy as np


if __name__ == "__main__":
    data_dir = "cyw/datasets/place_dataset/val/heuristic_agent_esc_yolo_place_cyw"
    data_key = 'pcd_base_coord_s'
    hdf5_file = os.path.join(data_dir,'data_out.hdf5') 
    hdf5_file = h5py.File(hdf5_file,'r+')

    # scene_ep_recep_grp = self.hdf5_file[f"/scene_{scene_id}/ep_{episode_id}/{recep_position}"]
    for scene in hdf5_file:
        for ep in hdf5_file[scene]:
            for recep in hdf5_file[scene][ep]:
                scene_ep_recep_grp = hdf5_file[scene][ep][recep]
                if data_key not in scene_ep_recep_grp:
                    continue
                data = scene_ep_recep_grp[data_key][()]
                if np.isnan(data).any():
                    data = np.nan_to_num(data, nan=0)
                    scene_ep_recep_grp[data_key][()] = data