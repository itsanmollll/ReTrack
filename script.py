# import streamlit as st
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torchvision.utils import draw_bounding_boxes


# # Load YOLOv5 from torch hub
# @st.cache_resource
# def load_model():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # You can use 'yolov5m' or 'yolov5l' for larger models
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

# # def create_image_with_bboxes(img, prediction): 
# #     # Convert the image to tensor
# #     img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
# #     # Draw bounding boxes
# #     img_with_bboxes = torch.tensor(np.array(img)).transpose(2, 0, 1)
# #     for box, label in zip(prediction["boxes"], prediction["labels"]):
# #         color = "red" if label == "person" else "green"
# #         img_with_bboxes = draw_bounding_boxes(img_with_bboxes, torch.tensor([box]), labels=[label], colors=[color], width=2)
    
# #     # Convert back to numpy for display
# #     img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
# #     return img_with_bboxes_np

# # Dashboard UI
# st.title("YOLOv5 Object Detector :tea: :coffee:")
# upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

# if upload:
#     img = Image.open(upload)

#     prediction = make_prediction(img)  # Dictionary with "boxes", "labels", "scores"
#     img_with_bbox = create_image_with_bboxes(np.array(img), prediction)

#     # Display the image with bounding boxes
#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(111)
#     plt.imshow(img_with_bbox)
#     plt.xticks([], [])
#     plt.yticks([], [])
#     ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

#     st.pyplot(fig, use_container_width=True)

#     # Show predictions excluding boxes (for clarity)
#     del prediction["boxes"]
#     st.header("Predicted Probabilities")
#     st.write(prediction)
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"] ## ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stopsign',]
img_preprocess = weights.transforms() ## Scales values from 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval(); ## Setting Model for Evaluation/Prediction   
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]                       ## Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
    return img_with_bboxes_np

## Dashboard
st.title("Objects Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image :", type=["png", "jpg", "jpeg"]) ## Image as Bytes 

if upload:
    img = Image.open(upload)

    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## Give image as (3,W,H) Channel first

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox);
    plt.xticks([],[]);
    plt.yticks([],[]);
    ax.spines[["top", "left", "right", "bottom"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    prediction["scores"] = prediction["scores"].detach().numpy().tolist()

    st.header("Prediction Probabilities")
    st.write(prediction)