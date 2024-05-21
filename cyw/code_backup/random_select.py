'''
随机生成一些轨迹id,以供大批量实验
'''
import random
import json

random_seed = 1234
random_num = 100
min_num = 0
max_num = 1199

random.seed(random_seed)

random_ids = random.sample(range(min_num,max_num),random_num)

# 保存id号
with open("cyw/data/exp/random_num_100.json","w") as f:
    json.dump(random_ids,f)
