'''
生成批量的prompt
'''
import json
import subprocess


recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']
# 来自于 projects/real_world_ovmm/configs/example_cat_map.json

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']

template = "You are a robot assistant. Your task is to use your prior knowledge about indoor environment to help robot to choose where to go to find the {object_name} as soon as possible. \
To select where to go, please follow my instructions step-by-step: \
1. List which of the following rooms the {object_name} might be in: [bedroom, living room, bathroom, kitchen, dining room, office room, gym, lounge, laundry room] \
2. List which of the following objects the {object_name} may be near: [bathtub, bed, bench, cabinet, chair, chest_of_drawers, couch, counter, filing_cabinet, hamper, serving_cart, shelves, shoe_rack, sink, stand, stool, table, toilet, trunk, wardrobe, washer_dryer] \
Note: In order for the robot to find obejct as quickly as possible, you should not leave out locations where the target object might be, but it is not like listing all the locations above either. "
out_format = r"Your output should be a quoted by brackets as a dict {\"rooms\":[room1, room2,...], \"objects\":[object1, object2,...]}.Do not output extra content!"

messages = []

for object_name in recep_category_21:
    prompt = template.format(object_name=object_name)
    prompt = prompt + out_format
    messages.append(prompt)


with open("cyw/data/message.json",'w') as f:
    json.dump(messages,f,indent=2)

results = []
for message in messages:
    curl_command = f'curl http://localhost:11434/api/chat -d \'{{"model": "llama3:8b", "messages": [{{"role": "user", "content": "{message}"}}], "stream": false,"options":{{"seed":123}}}}\''
    # 执行 curl 命令
    # print(curl_command)
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    # 输出命令执行结果
    result = json.loads(result.stdout)
    results.append(result)

with open("cyw/data/response.json",'w') as f:
    json.dump(results,f,indent=2)