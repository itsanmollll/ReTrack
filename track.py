from deepsort.deep_sort import DeepSort

class Tracker:
    def __init__(self):
        # Initialize the DeepSORT tracker
        self.deepsort = DeepSort(model_path="path_to_deepsort_model")

    def update(self, detections):
        # Filter out non-person detections if necessary
        person_detections = [d for d in detections if d[5] == 0]  # Assuming class 0 is 'person'
        
        # Track persons using DeepSORT
        outputs = self.deepsort.update(person_detections)
        return outputs
