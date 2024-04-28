#!/bin/bash
'''
对shell文件不太熟悉，代码还是错的
'''

json_file="/raid/home-robot/cyw/data/message.json"

# 使用jq命令读取JSON文件内容
result=$(jq '.' "$json_file")

# 获取列表的长度
length=$(echo "$result" | jq 'length')

# # 遍历列表
# for index in $(seq 0 $(($length - 1))); do
#   # 获取每个元素
#   element=$(echo "$result" | jq -r ".[$index]")

#   # 在这里执行你想要对每个元素执行的操作
#   echo "Element $index: $element"
# done


# 循环发送每条消息并将结果追加到文件
# 遍历列表
for index in $(seq 0 $(($length - 1))); do
    # 获取每个元素
    element=$(echo "$result" | jq -r ".[$index]")

    # 在这里执行你想要对每个元素执行的操作
    echo "Element $index: $element"
    curl http://localhost:11434/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama3:8b",
            "prompt": '"$element"',
            "options": {
                "seed": 123,
            }
        }' >> /raid/home-robot/cyw/data/response.json
done

curl http://localhost:11434/api/chat -d '{
  "model": "llama3:8b",
  "messages": [
    {
      "role": "user",
      "content": You are a robot assistant. Your task is to use your prior knowledge about indoor environment to help robot to choose where to go to find the bathtub as soon as possible. To select where to go, please follow my instructions step-by-step: 1. List which of the following rooms the bathtub might be in: [bedroom, living room, bathroom, kitchen, dining room, office room, gym, lounge, laundry room] 2. List which of the following objects the bathtub may be near: [bathtub, bed, bench, cabinet, chair, chest_of_drawers, couch, counter, filing_cabinet, hamper, serving_cart, shelves, shoe_rack, sink, stand, stool, table, toilet, trunk, wardrobe, washer_dryer] Note: In order for the robot to find obejct as quickly as possible, you should not leave out locations where the target object might be, but it is not like listing all the locations above either. Your output should be a quoted by brackets as a dict {\"rooms\":[room1, room2,...], \"objects\":[object1, object2,...]}.Do not output extra content!"
    }
  ],
  "stream": false
}' > /raid/home-robot/cyw/data/response.json
# 这里单引号和括号一起会被解析为主机名？