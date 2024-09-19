import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Read an image using OpenCV
source = cv2.imread("C:/Users/varsh/OneDrive/Pictures/Screenshots/Screenshot 2024-09-12 141209.png")
#source="0"
# Run inference on the source
results = model(source)  # list of Results objects

plt.imshow(source)
