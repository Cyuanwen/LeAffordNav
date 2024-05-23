




















names = {
  0: "bathtub",
  1: "bed",
  2: "bench",
  3: "cabinet",
  4: "chair",
  5: "chest_of_drawers",
  6: "couch",
  7: "counter",
  8: "filing_cabinet",
  9: "hamper",
  10: "serving_cart",
  11: "shelves",
  12: "shoe_rack",
  13: "sink",
  14: "stand",
  15: "stool",
  16: "table",
  17: "toilet",
  18: "trunk",
  19: "wardrobe",
  20: "washer_dryer"}

all_receptacles = [
    "cabinet",
    "stool",
    "trunk",
    "shoe_rack",
    "chest_of_drawers",
    "table",
    "toilet",
    "serving_cart",
    "bed",
    "washer_dryer",
    "hamper",
    "stand",
    "bathtub",
    "couch",
    "counter",
    "shelves",
    "chair",
    "bench",
]

recep_21 = set(names.values())
recep = set(all_receptacles)

diff = recep_21 - recep

print(diff)
