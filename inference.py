import cv2
import torch
from utils import draw_boxes, load_model
from track import Tracker

# Load YOLOv5 pre-trained model
model = load_model('yolov5s')

# Initialize tracker (DeepSORT, ByteTrack, etc.)
tracker = Tracker()

def detect_and_track(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Person Detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        # Step 2: Person Tracking
        tracked_objects = tracker.update(detections)

        # Step 3: Draw bounding boxes and IDs
        frame = draw_boxes(frame, tracked_objects)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_and_track('test_videos/test_video.mp4', 'outputs/output_test_video.mp4')
