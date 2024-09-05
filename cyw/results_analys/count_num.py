'''
计数符合条件的数量
'''
import json

def place_success_nav_fail(item:dict):
    return item["place_success"] and not item['nav2place']

def count(data:dict,condition_fn):
    """
        "scene_102817140ep_144_[-4.70602  0.75083 -0.95558]_[-5.43044  0.17759 -2.70904]": {
        "place_success": 0,
        "robot_scene_colls": 0,
        "ovmm_placement_stability": 0,
        "nav2place": 0
    },
    计算数据中 place_success = 1 但是 nav2place = 0 的episode数量
    """
    count = 0
    for item in data.values():
        if condition_fn(item):
            count += 1
    return count

if __name__ == "__main__":
    data_file = "cyw/datasets/place_dataset_debug/val/heuristic_agent_nav_place_cyw/v2_1_success.json"
    with open(data_file,"r") as f:
        data = json.load(f)
    count_num = count(data,place_success_nav_fail)
    print(f"the count num is {count_num}")
    # TODO the count num is 176 , 总共 500+ 占比还是有点大，收集数据的时候应该考虑这一指标

