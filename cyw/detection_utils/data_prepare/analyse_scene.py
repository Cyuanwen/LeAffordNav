'''
分析数据
1. 分析训练场景的物体类别，验证场景的物体类别
2. 分析数据集各类物体分布占比
'''
import pickle
import os
import json
from tqdm import tqdm

recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']
recep_category_21_id2name = {idx:name for idx,name in enumerate(recep_category_21)}


class ana_object:
    def __init__(self,
        data_dir:str
    ) -> None:
        self.data_dir = data_dir
    
    def ana_scene_object(self,receptacle_positions):
        '''
            分析各个场景的 goal_recep_category 分布（receptacle_positions是按照goal_recep_category来采集的）
        '''
        goal_recep_categorys = {}
        for scene_id, receptacle_positions in receptacle_positions.items():
            goal_recep_categorys[scene_id] = list(receptacle_positions.keys())
        
        # 保存数据
        with open(os.path.join(data_dir,"goal_recep_categorys.json"),"w") as f:
            json.dump(goal_recep_categorys,f,indent=2)
        
        # 是否所有类别数据都包括在内？
        recep_categorys = set()
        for receps in goal_recep_categorys.values():
            for recep in receps:
                recep_categorys.add(recep)
        recep_category_21_set = set(recep_category_21)
        if recep_categorys == recep_category_21_set:
            print("the recep data contain all 21 recep")
        else:
            miss_recep = recep_category_21_set - recep_categorys
            print(f"those receps {miss_recep} are missing")
    
    def get_object_ids(self,labels_file):
        '''根据labels，获得rgb中有哪些物体
        '''
        object_ids = []
        with open(labels_file,"r") as f:
            lines = f.readlines()
            for line in lines:
                object_ids.append(line.split()[0])
        return object_ids

    
    def ana_scene_data(self,label_in_scene:bool):
        '''
            分析采集到的场景的数据，各个类别的分布
            label_in_scene:label文件夹在scene文件夹里面
            if true: labels
                        scene
            else:
                    scene
                        labels
        '''
        scene_objects_all = {}
        if label_in_scene:
            scenes = os.listdir(self.data_dir)
            for scene in scenes:
                if os.path.isfile(os.path.join(self.data_dir,scene)):
                    break
                scene_objects = {i:0 for i in range(len(recep_category_21))}
                labels = os.listdir(os.path.join(self.data_dir,scene,"labels"))
                for label in tqdm(labels):
                    object_ids = self.get_object_ids(os.path.join(self.data_dir,scene,"labels",label))
                    for idx in object_ids:
                        scene_objects[int(idx)] += 1
                scene_objects_all[scene] = scene_objects
        else:
            scenes = os.listdir(os.path.join(self.data_dir,"labels"))
            for scene in scenes:
                scene_objects = {i:0 for i in range(len(recep_category_21))}
                labels = os.listdir(os.path.join(self.data_dir,"labels",scene))
                for label in tqdm(labels):
                    object_ids = self.get_object_ids(os.path.join(self.data_dir,"labels",scene,label))
                    for idx in object_ids:
                        scene_objects[idx] += 1
                scene_objects_all[scene] = scene_objects
        # 汇总所有场景
        objects = {i:0 for i in range(len(recep_category_21))}
        for scene_object_counts in scene_objects_all.values():
            for idx, count in scene_object_counts.items():
                objects[idx] += count
        scene_objects_all["total"] = objects
        # 保存文件
        with open(os.path.join(self.data_dir,"scene_objexts_count.json"),"w") as f:
            json.dump(scene_objects_all,f,indent=2)
    
    def __call__(self):
        if os.path.exists(os.path.join(self.data_dir,"recep_position.pickle")):
            label_in_scene = True
            with open(f"./{self.data_dir}/recep_position.pickle", "rb") as handle:
                receptacle_positions = pickle.load(handle)
            self.ana_scene_object(receptacle_positions)
        else:
            label_in_scene = False
        self.ana_scene_data(label_in_scene=label_in_scene)

if __name__ == "__main__":
    # data_dir = "cyw/datasets/datasets_v1/recep_data/train"
    data_dir = "cyw/datasets/datasets_v1/recep_data/val"
    analyser = ana_object(data_dir=data_dir)
    analyser()


    









