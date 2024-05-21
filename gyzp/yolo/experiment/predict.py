from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('./runs/segment/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image