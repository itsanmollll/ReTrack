import torch
import cv2

# Load YOLOv5 model (pre-trained on COCO dataset)
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

if __name__ == "__main__":
    model = load_yolov5_model()
    print("YOLOv5 model loaded successfully!")
