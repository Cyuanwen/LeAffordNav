"""
向一个文件夹写入数据，用来测试多进程脚本
"""
import argparse
import json

data_file = "cyw/test_data/test"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_index",type=int)
    parser.add_argument("--to_index",type=int)
    args = parser.parse_args()

    episodes = list(range(args.from_index,args.to_index))
    print(f"episodes is {episodes}")

    with open(f"{data_file}/{args.from_index}_{args.to_index}.json","w") as f:
        json.dump(episodes,f)

        