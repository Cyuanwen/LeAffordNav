import numpy as np

co_occur_mtx = np.load('cyw/ESC/ablations/npys/npys/deberta_predict.npy')

co_occur_room_mtx = np.load('cyw/ESC/ablations/npys/npys/deberta_predict_room.npy')

print("over")

categories_21 = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa',
'bed', 'chest_of_drawers', 'plant', 'sink', 'toilet', 'stool',
'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes']

recep_category_to_recep_category_id ={
    "bathtub": 0,
    "bed": 1,
    "bench": 2,
    "cabinet": 3,
    "chair": 4,
    "chest_of_drawers": 5,
    "couch": 6,
    "counter": 7,
    "filing_cabinet": 8,
    "hamper": 9,
    "serving_cart": 10,
    "shelves": 11,
    "shoe_rack": 12,
    "sink": 13,
    "stand": 14,
    "stool": 15,
    "table": 16,
    "toilet": 17,
    "trunk": 18,
    "wardrobe": 19,
    "washer_dryer": 20
}

recep_category_21 = ['bathtub', 'bed', 'bench', 'cabinet', 'chair', 'chest_of_drawers', 'couch', 'counter', 'filing_cabinet', 'hamper', 'serving_cart', 'shelves', 'shoe_rack', 'sink', 'stand', 'stool', 'table', 'toilet', 'trunk', 'wardrobe', 'washer_dryer']

# print(recep_category_21)