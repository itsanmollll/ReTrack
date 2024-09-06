import cv2
import torch

def load_model(model_name):
    return torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

def draw_boxes(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj[:5]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
