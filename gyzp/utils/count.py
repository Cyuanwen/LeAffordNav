import os

if __name__ == "__main__":
    path = "/raid/home-robot/gyzp/data/data2/receptacle/val/images"
    dirs = os.listdir(path)
    total = 0
    print(f"dir num: {len(dirs)}")
    for d in dirs:
        if os.path.isdir(os.path.join(path, d)):
            files = os.listdir(os.path.join(path, d))
            print(f"{d}: {len(files)}")
            total += len(files)
    print(f"Total: {total}")