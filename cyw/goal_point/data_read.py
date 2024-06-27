'''
读取数据
'''
import pickle

if __name__ == "__main__":
    data_dir = "cyw/datasets/place_dataset/train/rl_agent_place_test.pkl"
    with open(data_dir,"rb") as f:
        data = pickle.load(f)
    
    print("over")