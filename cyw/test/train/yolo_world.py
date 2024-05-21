from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld('yolov8s-worldv2.pt')

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='cyw/test_data/coco8/coco8.yaml', epochs=100, imgsz=640)
# results = model.train(data='cyw/data/detection/config/recep_seg_yolo_train.yaml', epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model('path/to/bus.jpg')
print("over")