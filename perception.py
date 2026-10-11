from ultralytics import YOLO
import cv2
import numpy as np
from common import CAM, RMAX, pixel_to_rover

# Load a lightweight, pre-trained YOLO model (nano version for CPU speed)
model = YOLO('yolo26n.pt') 

def detect_hazards(frame):
    # Run the model on the current camera frame
    results = model(frame, stream=True)
    
    detections = []
    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Format this data to match what localize.py expects
            w = x2 - x1
            h = y2 - y1
            
            # (Note: You'll need to map standard YOLO classes like 'rock' to your specific CLASSES list)
            if class_name in ["rock", "crater", "ice"]: 
                 detections.append({
                     "cls": class_name, 
                     "conf": int(confidence * 100), 
                     "x": x1, 
                     "y": y1, 
                     "u": x1 + w//2, 
                     "v": y1 + h//2, 
                     "w": w, 
                     "h": h
                 })
    return detections