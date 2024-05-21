import json


def get_dict(file_path):
    with open(file_path, "r") as f:
        return json.load(f).get("obj_category_to_obj_category_id")


if __name__ == "__main__":
    file_path = "/mnt/d/Lab/HomeRobot/home-robot/projects/real_world_ovmm/configs/example_cat_map.json"
    print(get_dict(file_path))
    print(get_dict(file_path).get("apple"))
