# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# import streamlit as st
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torchvision.utils import draw_bounding_boxes
# import cv2

# # Load YOLOv5 from torch hub
# @st.cache_resource
# def load_model():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)  # You can use 'yolov5m' or 'yolov5l' for larger models
#     model.eval()  # Set to evaluation mode
#     return model

# model = load_model()

# def create_image_with_bboxes(img, prediction): 
#     # Convert the image to tensor
#     img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
#     # Draw bounding boxes
#     img_with_bboxes = img_tensor.clone()
#     for box, label in zip(prediction["boxes"], prediction["labels"]):
#         color = "red" if label == "person" else "green"
#         img_with_bboxes = draw_bounding_boxes(img_with_bboxes, torch.tensor([box]), labels=[label], colors=[color], width=2)
    
#     # Convert back to numpy for display
#     img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
#     return img_with_bboxes_np

# def make_prediction(img): 
#     # Perform inference
#     results = model(img)
    
#     # Extracting the bounding boxes, labels, and scores
#     predictions = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]
    
#     # Convert to a dictionary format
#     prediction_dict = {
#         "boxes": predictions[:, :4].cpu().numpy(),  # Bounding box coordinates
#         "scores": predictions[:, 4].cpu().numpy(),  # Confidence scores
#         "labels": [model.names[int(cls)] for cls in predictions[:, 5].cpu().numpy()]  # Class labels
#     }
    
#     return prediction_dict

# def process_video(input_video_path, output_video_path):
#     # Print the input video path for debugging
#     print(f"Input video path: {input_video_path}")
    
#     # Open the video file
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
    
#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     # Print video properties for debugging
#     print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert frame to PIL Image
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         # Make prediction
#         prediction = make_prediction(img)
        
#         # Draw bounding boxes
#         img_with_bboxes = create_image_with_bboxes(img, prediction)
        
#         # Convert back to OpenCV format
#         frame_with_bboxes = cv2.cvtColor(np.array(img_with_bboxes), cv2.COLOR_RGB2BGR)
        
#         # Write the frame
#         out.write(frame_with_bboxes)
    
#     # Release everything if job is finished
#     cap.release()
#     out.release()
#     print("Video processing complete.")

# # Example usage
# input_video_path = 'test_videos/test.mp4'
# output_video_path = 'output_video.mp4'
# process_video(input_video_path, output_video_path)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import cv2

# Load Faster R-CNN from torchvision
@st.cache_resource
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()  # Set to evaluation mode
    return model, weights

model, weights = load_model()
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]  # Dictionary with keys "boxes", "labels", "scores"
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): 
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, 
        boxes=prediction["boxes"], 
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]], 
        width=2
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return img_with_bboxes_np

def process_video(input_video_path, output_video_path):
    # Print the input video path for debugging
    print(f"Input video path: {input_video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Print video properties for debugging
    print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Make prediction
        prediction = make_prediction(img)
        
        # Draw bounding boxes
        img_with_bboxes = create_image_with_bboxes(img, prediction)
        
        # Convert back to OpenCV format
        frame_with_bboxes = cv2.cvtColor(np.array(img_with_bboxes), cv2.COLOR_RGB2BGR)
        
        # Write the frame
        out.write(frame_with_bboxes)
    
    # Release everything if job is finished
    cap.release()
    out.release()
    print("Video processing complete.")

# Example usage
input_video_path = 'test_videos/test.mp4'
output_video_path = 'output_video.mp4'
process_video(input_video_path, output_video_path)