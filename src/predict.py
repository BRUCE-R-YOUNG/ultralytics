from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("src/best.pt")

# from ndarray
im1 = Image.open("docs/datasets/colony/train/IMG_4494.JPG")
results = model.predict(source=im1, save=True, conf=0.15)  # save plotted images

