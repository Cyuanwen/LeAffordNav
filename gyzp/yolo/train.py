from ultralytics import YOLO

# Load a model
model = YOLO(
    "/raid/home-robot/gyzp/yolo/models/yolov8m-seg.pt"
)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./config/recep_seg_yolo_train.yaml", epochs=50, imgsz=640)
