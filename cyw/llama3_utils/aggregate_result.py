'''
将所有结果聚合起来，并进行初步过滤
NOTE: 过滤很有必要，llama3-7b会输出一些没有指定的词汇
'''
import json

target_room_file = "/raid/home-robot/cyw/data/target_room.json"
target_obj_file = "cyw/data/target_object.json"
target_neg_room_file = "cyw/data/target_negative_room.json"
target_neg_obj_file = "cyw/data/target_negative_object.json"

recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']
# 来自于 projects/real_world_ovmm/configs/example_cat_map.json

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']


with open(target_room_file,'r') as f:
    target_room = json.load(f)

with open(target_obj_file,'r') as f:
    target_obj = json.load(f)

with open(target_neg_room_file,'r') as f:
    target_neg_room = json.load(f)

with open(target_neg_obj_file,'r') as f:
    target_neg_obj = json.load(f)
co_occur_data = {}
for recep in recep_category_21:
    if recep == "toilet":
        print("debug")
    rec_rooms = target_room[recep]
    ini_room_len = len(rec_rooms)
    rec_rooms = [rec_room for rec_room in rec_rooms if rec_room in rooms]
    if len(rec_rooms) != ini_room_len:
        print("the length of rec_rooms not equall with initial length")
    
    neg_rec_rooms = target_neg_room[recep]
    ini_room_len = len(neg_rec_rooms)
    neg_rec_rooms = [rec_room for rec_room in neg_rec_rooms if rec_room in rooms and rec_room not in rec_rooms]
    if len(neg_rec_rooms) != ini_room_len:
        print("the length of neg_rec_rooms not equall with initial length")
    
    recp_objs = target_obj[recep]
    ini_obj_len = len(recp_objs)
    recp_objs = [rec_obj for rec_obj in recp_objs if rec_obj in recep_category_21 and rec_obj != recep]
    if len(recp_objs) != ini_obj_len:
        print("the length of recp_objs not equall with initial length")
    
    neg_recp_objs = target_neg_obj[recep]
    ini_obj_len = len(neg_recp_objs)
    neg_recp_objs = [rec_obj for rec_obj in neg_recp_objs if rec_obj in recep_category_21 and recp_objs != recep and rec_obj not in recp_objs]
    if len(neg_recp_objs) != ini_obj_len:
        print("the length of neg_recp_objs not equall with initial length")
    
    co_occur_data[recep] = {
        "pos_rooms": rec_rooms,
        "neg_rooms": neg_rec_rooms,
        "pos_objs": recp_objs,
        "neg_objs": neg_recp_objs
    }

with open("cyw/data/co_occur_llama3_7b.json",'w') as f:
    json.dump(co_occur_data,f,indent=4)

