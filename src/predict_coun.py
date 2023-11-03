import supervision as sv
from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('src/best.pt')

IMAGE = 'docs/datasets/colony/train/IMG_4642.JPG'
result = model(IMAGE)[0]


results = model.predict(source=IMAGE, save=True, conf=0.10)  # save plotted images

detections = sv.Detections.from_ultralytics(result)
print(len(detections))

