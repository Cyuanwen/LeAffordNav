'''
对容器位置进行聚类
recep_pos data orginaze：
receptacle_positions[scene_id][episode.goal_recep_category] = []
receptacle_positions[scene_id][episode.goal_recep_category].append(
                {
                    "recep_position": recep_position,
                    "view_point_positions":view_point_positions
                }
            )
recep_position = list(recep.position) # recep数据里面没有朝向
view_point_position = list(get_agent_state_position(env._dataset.viewpoints_matrix,view_point).position)
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import random
from tqdm import tqdm
import os
from pathlib import Path

random.seed(1234)

FIG_DIR = "cyw/test_data/viewpos_cluster"
show_image = False
MAX_RECEP_NUM = 4

def gather_recep_view_pos(scene_recep_pos):
    '''
        集中 scene recep 数据中的 recep_pos 和 viewpoint_pos
    '''
    recep_pos = []
    view_pos = []
    for item in scene_recep_pos:
        recep_pos  = recep_pos + item['recep_position']
        view_pos = view_pos + list(item['view_point_positions'])
    recep_pos = np.array(recep_pos).reshape(-1,3)[:,[0,2]]
    view_pos = np.array(view_pos)[:,[0,2]]
    return recep_pos, view_pos

'''可视化数据'''
def vis_pos(recep_pos, view_pos,idx,recep_name,fig_name='pos'):
    x1 = recep_pos[:,0]
    y1 = recep_pos[:,1]
    x2 = view_pos[:,0]
    y2 = view_pos[:,1]

    plt.figure()
    plt.scatter(x1,y1,color='red',label=recep_name)
    plt.scatter(x2,y2,color='black',label='view')
    plt.axis('equal')
    # 显示网格线
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{FIG_DIR}/{fig_name}_{idx}.jpg")
    # plt.show()
    plt.close()

def get_cluster_class(distance_threshold):
    '''
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#gallery-examples
    '''
    return AgglomerativeClustering(n_clusters=None,distance_threshold=distance_threshold)

def clustering(cluster_class, data,name=None,idx=None):
    if len(data) < 2:
        return np.array([1])
    clustering = cluster_class.fit(data)
    labels = clustering.labels_
    if show_image:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
        plt.axis('equal')
        # 显示网格线
        plt.grid(True)
        plt.colorbar()
        plt.savefig(f"{FIG_DIR}/{name}_{idx}.jpg")
        # plt.show()
        plt.close()
    return labels

def get_cluster_center(data,labels,method="mean"):
    '''
        计算每个类别的聚类中心
        method: mean, 均值； sample：随机采样
    '''
    represents = [] # 代表样本
    unique_label = np.unique(labels)
    for label in unique_label:
        data_i = data[labels==label]
        if method == "mean":
            sample_i = np.mean(data_i,axis=0)
        elif method == "sample":
            sample_i = random.sample(data_i,1)
        else:
            raise NotImplementedError
        represents.append(sample_i)
    represents = np.array(represents)
    represents.reshape(-1,2)
    return represents

# def sample_recep(labels,max_num):
#     '''
#         给定聚类结果，返回 (num_recep,) list 代表该容器是否被选中
#     '''
#     label_unique = np.unique(labels)
#     if len(label_unique) <= max_num:
#         label_choice = label_unique
#     else:
#         label_choice - random.sample(label_unique,max_num)
#     for label_i in label_choice:
# TODO 暂不考虑，考虑多线程加速


def main(data,data_dir):
    idx = 0
    viewpoint_num = 0
    # recep_cluster = get_cluster_class(distance_threshold=1)
    # view_cluster = get_cluster_class(distance_threshold=0.25) # 90000+
    view_cluster = get_cluster_class(distance_threshold=0.5) # 50000左右
    data_new = {}
    for scene in tqdm(data):
        data_new[scene] = {}
        for recep in data[scene]:
            data_new[scene][recep] = []
            recep_pos, _ = gather_recep_view_pos(data[scene][recep])
            # recep_labels = clustering(recep_cluster,recep_pos,'recep_pos',idx)
            for scene_ep_pos in data[scene][recep]:
                view_pos = list(scene_ep_pos['view_point_positions'])
                view_pos = np.array(view_pos)
                height = view_pos[0,1]
                view_pos = view_pos[:,[0,2]]
                labels = clustering(view_cluster,view_pos,'view_point',idx)
                view_represents =  get_cluster_center(view_pos,labels,method='mean')
                # array_3d = np.insert(array_2d, 1, fixed_number, axis=1)
                view_represents = np.insert(view_represents,1,height,axis=1)
                data_new[scene][recep].append(
                    {
                    "recep_position": scene_ep_pos['recep_position'],
                    "view_point_positions":view_represents
                    }
                )
                viewpoint_num += len(view_represents)
            # 可视化
            if show_image:
                recep_represents,view_represents = gather_recep_view_pos(data_new[scene][recep])
                vis_pos(recep_pos, view_pos,idx,recep,'pos')
                vis_pos(recep_represents,view_represents,idx,recep,'pos_represent')
                idx += 1
    print(f"the viewpoint_num is {viewpoint_num}")
    # 保存数据
    data_file = os.path.join(data_dir,'recep_position_cluster.pickle')
    with open(data_file,'wb') as f:
        pickle.dump(data,f)
    print(f"data save in {data_file}*************")


if __name__ == "__main__":
    # data_file = "cyw/datasets/place_dataset/val/recep_position.pickle"
    data_file = "cyw/datasets/place_dataset/train/recep_position.pickle"
    with open(data_file,"rb") as f:
        recep_pos = pickle.load(f)
    data_dir = str(Path(data_file).resolve().parent)
    main(recep_pos,data_dir)
    print("over")