from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(data='src/colony.yaml', epochs=5)
