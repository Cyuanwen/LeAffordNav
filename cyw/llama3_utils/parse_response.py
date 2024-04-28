'''
解析llama生成的内容
'''
import json
import re


recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']
# 来自于 projects/real_world_ovmm/configs/example_cat_map.json

responses_file = "cyw/data/response.json"

with open(responses_file,'r') as f:
    responses = json.load(f)

target_object = {}
target_room = {}

for target, response in zip(recep_category_21,responses):
    response = response['message']['content']
    # # 使用正则表达式在所有单词上面加上双引号
    # response = re.sub(r'(\b\w+\b)', r'"\1"', response)
    # 这样会使得living room被分开
    try:
        response = json.loads(response)
    except:
        print(f"the target {target} can't be parsed")
    if isinstance(response,list):
        # llm可能会多输出一个[]
        response = response[0]
    target_object[target] = response['objects']
    target_room[target] = response['rooms']

with open("cyw/data/target_object.json",'w') as f:
    json.dump(target_object,f,indent=4)

with open("cyw/data/target_room.json",'w') as f:
    json.dump(target_room,f,indent=4)
