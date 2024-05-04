'''
测试psl，参考ESC代码
ESC 源代码使用了两种方式：PSL推理，优化概率；计算分数，选择分数最高的
或许可两者都尝试？

官方代码示例：https://github.com/linqs/psl-examples/tree/221ad2b86da8b698b9051665fd94e1ce2ade5a7d/citeseer

贪婪的找最近的frontier或者找最靠近object的frontier并不是最优，能否在更大范围搜索？

NOTE: PSL运行逻辑：将代码里面的数据写到 /tmp/psl-python/<psl_name>下,调用命令行
写入数据的时候注意数据类型，有可能整型的数据在和其它数据合并的时候会变为浮点型，这时候无法推理

TODO: 让大模型给出不可能在的地方，简化提示试一下
这些先验应该随着探索过程不断更新
距离不能按照min max归一化，这样会导致距离最小的score为0，最大的为1，可能会导致永远都选择距离最小的
'''
from array import array
import enum
from types import SimpleNamespace
from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

import json
import pandas as pd
import numpy as np

debug = True

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]

# ADDITIONAL_PSL_OPTIONS = {
#     'runtime.log.level': 'INFO'
#     # 'runtime.db.type': 'Postgres',
#     # 'runtime.db.pg.name': 'psl',
# }

config = {
    "co_occur_file": "cyw/data/co_occur_llama3_7b.json",
    "reasoning": "object", # the method to reasoning, choose from ["both","room","object"],
    "PSL_infer": "optimal", # the method to infer, choose from ["optimal","approximation"]
    "probability_neg_recep": 0.1, #分给负相关物体的概率
    "probability_neg_room": 0.1, #分给负相关房间的概率
    "probability_pos_recep": 0.9, 
    "probability_pos_room": 0.9, 
    "probability_other_recep": 0.3, 
    "probability_other_room": 0.3,
    "verbose":True
}
config = SimpleNamespace(**config)

recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']

recep_category_21_to_idx = {recep:i for i, recep in enumerate(recep_category_21)}

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']

def get_dist_score(dist_matrix:array, verbose: bool=False,
    min_dist=None, max_dist=None):
    '''
    由距离矩阵获得距离分数：对整个矩阵归一化,取反
    '''
    if min_dist is None:
        min_dist = np.min(dist_matrix)
    if max_dist is None:
        max_dist = np.max(dist_matrix)
    if verbose:
        print(f"minmal distance {min_dist}, maxmal distance {max_dist}")
    dist_matrix_score = 1 - (dist_matrix - min_dist) / (max_dist - min_dist)
    return dist_matrix_score


class psl_agent(object):
    '''
    wrapper for PSL
    '''
    def __init__(self,
    config,
    ) -> None:
        self.config = config
        # 所有的设置在config文件指明
        with open(config.co_occur_file,'r') as f:
            self.co_occur = json.load(f)
        # 计算共现矩阵 co_occur_room_mtx, co_occur_obj_mtx: 每一行代表一个recep
        co_occur_room_mtx = np.zeros((len(recep_category_21),len(rooms)))
        co_occur_obj_mtx = np.zeros((len(recep_category_21),len(recep_category_21)))
        for id_rec in range(len(recep_category_21)):
            rec = recep_category_21[id_rec]
            rec_co_occur = self.co_occur[rec]
            for id_room in range(len(rooms)):
                if rooms[id_room] in  rec_co_occur['pos_rooms']:
                    co_occur_room_mtx[id_rec,id_room] = 2
                elif rooms[id_room] in rec_co_occur['neg_rooms']:
                    co_occur_room_mtx[id_rec][id_room] = 0
                else:
                    co_occur_room_mtx[id_rec][id_room] = 1
            for id_obj in range(len(recep_category_21)):
                if recep_category_21[id_obj] in rec_co_occur['pos_objs']:
                    co_occur_obj_mtx[id_rec,id_obj] = 2
                elif recep_category_21[id_obj] in rec_co_occur['neg_objs']:
                    co_occur_obj_mtx[id_rec,id_obj] = 0
                else:
                    co_occur_obj_mtx[id_rec,id_obj] = 1
        co_occur_room_mtx[co_occur_room_mtx==2] = getattr(config,'probability_pos_room',1)
        co_occur_room_mtx[co_occur_room_mtx==1] = getattr(config,"probability_other_room",0.5)
        co_occur_room_mtx[co_occur_room_mtx==0] = getattr(config,"probability_neg_room",0)
        co_occur_obj_mtx[co_occur_obj_mtx==2] = getattr(config,"probability_pos_recep",1)
        co_occur_obj_mtx[co_occur_obj_mtx==1] = getattr(config,"probability_other_recep",0.5)
        co_occur_obj_mtx[co_occur_obj_mtx==0] = getattr(config,"probability_neg_recep",0)
        np.fill_diagonal(co_occur_obj_mtx,1)
        
        self.co_occur_room_mtx = co_occur_room_mtx
        self.co_occur_obj_mtx = co_occur_obj_mtx

        if debug:
            co_occur_room_df = pd.DataFrame(co_occur_room_mtx,index=recep_category_21,columns=rooms)
            co_occur_obj_df = pd.DataFrame(co_occur_obj_mtx,index=recep_category_21,columns=recep_category_21)
            # 保存datafreme
            co_occur_room_df.to_csv("cyw/data/co_occur_room_df.csv")
            co_occur_obj_df.to_csv("cyw/data/co_occur_obj_df.csv")

        self.reasoning = config.reasoning
        self.psl_model = PSLModel('OVMM-PSL')  ## important: please use different name here for different process in the same machine. eg. objnav, objnav2, ...
        # 多线程应该怎么办？只有选用optimal infer，这个名字才会有意义，会创建一个文件夹，存储中间数据，否则这个名字没有意义
        self.PSL_infer = config.PSL_infer

        if self.PSL_infer == "optimal":
            # Add Predicates
            self.add_predicates(self.psl_model)

            # Add Rules
            self.add_rules(self.psl_model)

        self.verbose = self.config.verbose

        # 方便使用的变量
        self.rooms_idxs = np.array(range(len(rooms)))
        self.obj_idxs = np.array(range(len(recep_category_21)))
        self.room_num = len(rooms)
        self.obj_num = len(recep_category_21)
    

    def add_predicates(self, model:PSLModel):
        """
        add predicates for ADMM PSL inference
        """
        if self.reasoning in ['both', 'object']:

            predicate = Predicate('IsNearObj', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('ObjCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        if self.reasoning in ['both', 'room']:

            predicate = Predicate('IsNearRoom', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('RoomCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model:PSLModel):
        """
        add rules for ADMM PSL inference
        """
        if self.reasoning in ['both', 'obj']:
            model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
            model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        if self.reasoning in ['both', 'room']:
            model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
            model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def clear_data(self):
        '''
        删除PSL模型里面的数据
        '''
        if self.verbose:
            print("clear data ......")
        for predicate in self.psl_model.get_predicates().values():
            predicate.clear_data()
            if self.verbose:
                print(f"delete data of {predicate.name()}")
    
    def set_target (self, target:str):
        '''
        set target object
        '''
        self.target = target
        target_idx = recep_category_21_to_idx[target]

        if self.PSL_infer == "optimal":
            self.clear_data()

            if self.reasoning in ["both","room"]:
                # RoomCooccur(R)
                co_room_scores = self.co_occur_room_mtx[target_idx,:]
                data = np.stack((self.rooms_idxs,co_room_scores)).transpose()
                data = pd.DataFrame(data, columns = list(range(2)))
                # columns一定要按照0，1，2……命名，否则会出错
                data[0] = data[0].astype(int)
                self.psl_model.get_predicate('RoomCooccur').add_data(Partition.OBSERVATIONS, data)
            if self.reasoning in ["both","object"]:
                # ObjCooccur(O)
                co_obj_score = self.co_occur_obj_mtx[target_idx,:]
                data = np.stack((self.obj_idxs,co_obj_score)).transpose()
                data = pd.DataFrame(data,columns=list(range(2)))
                data[0] = data[0].astype(int)
                self.psl_model.get_predicate('ObjCooccur').add_data(Partition.OBSERVATIONS, data)
        else:
            # RoomCooccur(R)
            self.co_room_scores = self.co_occur_room_mtx[target_idx,:]
            # ObjCooccur(O)
            self.co_obj_score = self.co_occur_obj_mtx[target_idx,:]
    
    def infer_optimal(
        self,near_room_frontier:array, 
        near_object_frontier:array,
        dist_frontier:array
    ):
        '''
        严格按照最优化PSL目标函数的方式计算：耗时
        near_room_frontier: 前端点和房间是否靠近，near_room_frontier(i,j)表示第j个前端点附近是否有房间i, 0,1变量
        near_object_frontier: 前端点和物体是否靠近，near_object_frontier(i,j)表示第j个前端点附近是否有物体i, 0,1变量
        注意：不是距离，仅仅判断有没有出现，因为距离很难计算
        dist_frotier: frontier距离当前位置的距离，shape:(num_frontier,1)
        TODO: 加上obj room的判断
        '''
        assert len(near_room_frontier) == len(rooms), "the length of distance matrix between rooms and frontiers is not equal as the length of target_co_occur_rooms"
        assert len(near_object_frontier) == len(recep_category_21), "the length of distance matrix between objects and frontiers is not equal as the length of target_co_occur_objects"
        assert near_room_frontier.shape[1] == near_object_frontier.shape[1] == len(dist_frontier), "the frontier number in dist_room_frontier is not equall with which in dist_object_frontier"

        frontier_idxs = np.array(range(len(dist_frontier)))

        #  clear the data before
        for predicate in self.psl_model.get_predicates().values():
            if predicate.name() in ['ISNEARROOM', 'ISNEAROBJ','CHOOSE', 'SHORTDIST']:
                predicate.clear_data()
        if self.reasoning in ["both","room"]:
            # IsNearRoom(R,F)
            data = np.stack((np.repeat(self.rooms_idxs,frontier_num),np.tile(frontier_idxs,self.room_num),near_room_frontier.flatten())).transpose()
            data = pd.DataFrame(data,columns=list(range(3)))
            data[0] = data[0].astype(int)
            data[1] = data[1].astype(int)
            self.psl_model.get_predicate('IsNearRoom').add_data(Partition.OBSERVATIONS, data)
        if self.reasoning in ["both","object"]:
            # IsNearObj(O,F)
            data = np.stack((np.repeat(self.obj_idxs,frontier_num),np.tile(frontier_idxs,self.obj_num), near_object_frontier.flatten())).transpose()
            data = pd.DataFrame(data,columns=list(range(3)))
            data[0] = data[0].astype(int)
            data[1] = data[1].astype(int)
            self.psl_model.get_predicate('IsNearObj').add_data(Partition.OBSERVATIONS, data)

        # ShortDist(F)
        dist_score = get_dist_score(dist_frontier)
        data = np.stack((frontier_idxs,dist_score)).transpose()
        data = pd.DataFrame(data,columns=list(range(2)))
        data[0] = data[0].astype(int)
        self.psl_model.get_predicate('ShortDist').add_data(Partition.OBSERVATIONS, data)

        # Choose(F)
        data = pd.DataFrame(frontier_idxs, columns = list(range(1)))
        self.psl_model.get_predicate('Choose').add_data(Partition.TARGETS, data)

        # infer
        result = self.psl_model.infer(additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)
        for key, value in result.items():
            result_dt_frame = value
            scores = result_dt_frame.loc[:,'truth']
            idx_frontier = frontier_idxs[np.argmax(scores)]   
            print(f"the best frontier:{idx_frontier}")
    
    def infer_approximation(
        self,near_room_frontier:array, 
        near_object_frontier:array,
        dist_frontier:array
    ):
        '''
        近似计算PSL结果：按照规则打分（实际上，规则已经蕴藏在打分计算方式里面了）计算速度快
        near_room_frontier: 前端点和房间是否靠近，near_room_frontier(i,j)表示第j个前端点附近是否有房间i, 0,1变量
        near_object_frontier: 前端点和物体是否靠近，near_object_frontier(i,j)表示第j个前端点附近是否有物体i, 0,1变量
        注意：不是距离，仅仅判断有没有出现，因为距离很难计算
        dist_frotier: frontier距离当前位置的距离，shape:(num_frontier,1)
        
        ESC源代码里面：物体和房间为什么要分开计算
        '''
        assert len(near_room_frontier) == len(rooms), "the length of distance matrix between rooms and frontiers is not equal as the length of target_co_occur_rooms"
        assert len(near_object_frontier) == len(recep_category_21), "the length of distance matrix between objects and frontiers is not equal as the length of target_co_occur_objects"
        assert near_room_frontier.shape[1] == near_object_frontier.shape[1] == len(dist_frontier), "the frontier number in dist_room_frontier is not equall with which in dist_object_frontier"

        num_frontiers = len(dist_frontier)
        scores = np.zeros((1, num_frontiers))

        if self.reasoning in ["both","room"]:
            # ## PSL score calculation using one hot constraint: the score of selecting a frontier
            co_room_scores = self.co_room_scores.reshape((9,1))
            score_room_1 = np.clip(co_room_scores + near_room_frontier -1, 0, 10)
            score_room_2 = 1- np.clip(co_room_scores - near_room_frontier + 1, -10, 1)
            score_room = (score_room_1 - score_room_2).sum(axis = 0) #(1,num_frontier)
            scores += score_room #(1,num_frontier)
        if self.reasoning in ["both","obj"]:
            # ## PSL score calculation using one hot constraint: the score of selecting each frontier
            co_obj_score = self.co_obj_score.reshape((21,1))
            score_obj_1 = np.clip(co_obj_score + near_object_frontier - 1, 0, 10)
            score_obj_2 = 1 - np.clip(co_obj_score - near_object_frontier + 1, -10, 1)
            score_obj = (score_obj_1 - score_obj_2).sum(axis = 0) #(1,num_frontier)
            scores += score_obj #(1,num_frontier)
        dist_score = get_dist_score(dist_frontier).reshape((1,3)) #(1,num_frontier)
        if self.reasoning == "both":
            scores += 2 * dist_score
        else:
            scores += dist_score
        idx_frontier = np.argmax(scores)
        print(f"the best frontier:{idx_frontier}")
    
    def infer(
        self,near_room_frontier:array, 
        near_object_frontier:array,
        dist_frontier:array
    ):
        '''
        计算PSL结果，可在配置文件里面指定按照最优方式计算还是按照近似方式计算，两者结果似乎相差不多，近似计算更快
        near_room_frontier: 前端点和房间是否靠近，near_room_frontier(i,j)表示第j个前端点附近是否有房间i, 0,1变量
        near_object_frontier: 前端点和物体是否靠近，near_object_frontier(i,j)表示第j个前端点附近是否有物体i, 0,1变量
        注意：不是距离，仅仅判断有没有出现，因为距离很难计算
        dist_frotier: frontier距离当前位置的距离，shape:(num_frontier,1)
        '''
        if self.PSL_infer == "optimal":
            self.infer_optimal(near_room_frontier,near_object_frontier,dist_frontier)
        else:
            self.infer_approximation(near_room_frontier,near_object_frontier,dist_frontier)


if __name__ == "__main__":
    psl_agent_1 = psl_agent(config)
    psl_agent_1.set_target("bed")
    frontier_num = 3
    obj_num = 21
    room_num = 9

    values = [0,1]
    probabilities = [0.6,0.4]
    
    np.random.seed(123)
    near_room_frontier = np.random.choice(values,size=(room_num,frontier_num),p=probabilities)
    near_object_frontier = np.random.choice(values,size=(obj_num,frontier_num),p=probabilities)
    print("the room frontier==============")
    print(near_room_frontier)
    print("the object frontier===============")
    print(near_object_frontier)
    dist_frontier = np.random.rand(frontier_num)
    print("distance of frontier===========")
    print(dist_frontier)

    psl_agent_1.infer(near_room_frontier,near_object_frontier,dist_frontier)



        



    