'''
生成批量的prompt
询问LLM，target不可能在哪些物体附近

NOTE: curl 的 content 里面的内容如果包含" '等，需要用\ 转义符
这样太复杂的提示感觉还是不太行，生成的东西有可能跳出要求的格式
'''
import json
import subprocess


recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']
# 来自于 projects/real_world_ovmm/configs/example_cat_map.json

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']

template = "You are a robot assistant. Your task is to use your prior knowledge about indoor environment to help robot to choose where to go to find the {object_name} as soon as possible. \
In order to be efficient, you need to give some locations where the {object_name} cannot possibly be. Please follow my instructions step-by-step: \
1. List which of the following rooms the {object_name} could not be in: [bedroom, living room, bathroom, kitchen, dining room, office room, gym, lounge, laundry room] \
2. List which of the following objects the {object_name} could not be near: [bathtub, bed, bench, cabinet, chair, chest_of_drawers, couch, counter, filing_cabinet, hamper, serving_cart, shelves, shoe_rack, sink, stand, stool, table, toilet, trunk, wardrobe, washer_dryer] \
Note: In order not to miss some important locations where the {object_name} really are, you should only list locations where you are pretty sure the {object_name} can not be."
out_format = r"Your output should be a quoted by brackets as a dict {\"rooms\":[room1, room2,...], \"objects\":[object1, object2,...]}.Pleasr give the result directly, do not output extra content"

messages = []

for object_name in recep_category_21:
    prompt = template.format(object_name=object_name)
    prompt = prompt + out_format
    messages.append(prompt)


with open("cyw/data/message_negative.json",'w') as f:
    json.dump(messages,f,indent=2)

results = []
for message in messages:
    curl_command = f'curl http://localhost:11434/api/chat -d \'{{"model": "llama3:8b", "messages": [{{"role": "user", "content": "{message}"}}], "stream": false,"options":{{"seed":245}}}}\''
    # 执行 curl 命令
    # print(curl_command)
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    # 输出命令执行结果
    result = json.loads(result.stdout)
    results.append(result)

with open("cyw/data/response_negative.json",'w') as f:
    json.dump(results,f,indent=2)