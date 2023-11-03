from typing import List
import numpy as np
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='image path')
args = parser.parse_args()

image_path = args.image_path

class ShelfDetector:
    def __init__(self, model_path: str, confidence: float = 0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect_shelves(self, image_path: str) -> List[int]:
        result = self.model.predict(source=image_path, conf=self.confidence, save=False)
        arrxy = result[0].boxes.xyxy
        coordinates = np.array(arrxy)

        x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2
        y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2
        midpoints = np.column_stack((x_coords, y_coords))

        sorted_midpoints = midpoints[midpoints[:,1].argsort()]
        rounded_n_sorted_arr = np.round(sorted_midpoints).astype(int)

        group_sizes = []
        objects = 0
        for i in range(1, len(rounded_n_sorted_arr)):
            if rounded_n_sorted_arr[i][1] - rounded_n_sorted_arr[i-1][1] > 130:
                group_sizes.append(objects + 1)
                objects = 0
            else:
                objects += 1
        
        group_sizes.append(objects + 1)
        return group_sizes

detector = ShelfDetector('best.pt')
result = detector.detect_shelves(image_path)
for i, size in enumerate(result):
    print(f"{i+1}. There are {size} products on the shelf")