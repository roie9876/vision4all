import cv2
import numpy as np
from ultralytics import YOLO
from centroid_tracker import CentroidTracker  # Ensure this import matches your file structure

class YOLOModel:
    def __init__(self):
        # Load the YOLOv8 model; adjust the model path as needed.
        self.model = YOLO('yolov8s.pt')
        self.object_classes = ['car', 'person', 'dog', 'cat']
        # (Optional) Create a mapping of class IDs if needed.
        self.object_class_ids = [cls_id for cls_id, cls_name in self.model.names.items() 
                                 if cls_name in self.object_classes]

    def count_objects(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Each detection: [x1, y1, x2, y2, conf, cls_id]

        counts = {"human": 0, "car": 0, "animal": 0}
        for obj in detections:
            _, _, _, _, _, cls_id = obj
            cls_id = int(cls_id)
            cls_name = self.model.names[cls_id]
            if cls_name == 'person':
                counts["human"] += 1
            elif cls_name == 'car':
                counts["car"] += 1
            elif cls_name in ['dog', 'cat']:
                counts["animal"] += 1

        return counts

    def process_video_with_counts(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error opening video: {input_video_path}")
            return None

        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        # Create a separate centroid tracker for each class.
        trackers = {
            "human": CentroidTracker(maxDisappeared=40),
            "car": CentroidTracker(maxDisappeared=40),
            "animal": CentroidTracker(maxDisappeared=40)
        }
        # Sets to hold unique object IDs seen for each class.
        unique_ids = {
            "human": set(),
            "car": set(),
            "animal": set()
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection on the current frame.
            results = self.model(frame)
            detections = results[0].boxes.data.cpu().numpy()  # Each detection: [x1, y1, x2, y2, conf, cls_id]

            # Organize bounding boxes by class.
            boxes = {"human": [], "car": [], "animal": []}
            for obj in detections:
                x1, y1, x2, y2, conf, cls_id = obj
                cls_id = int(cls_id)
                cls_name = self.model.names[cls_id]
                if cls_name == 'person':
                    boxes["human"].append((int(x1), int(y1), int(x2), int(y2)))
                elif cls_name == 'car':
                    boxes["car"].append((int(x1), int(y1), int(x2), int(y2)))
                elif cls_name in ['dog', 'cat']:
                    boxes["animal"].append((int(x1), int(y1), int(x2), int(y2)))

            # Update each tracker and record the unique object IDs.
            for cls in trackers:
                objects = trackers[cls].update(boxes[cls])
                for object_id in objects.keys():
                    unique_ids[cls].add(object_id)

            # (Optional) Draw the tracked centroids and IDs on the frame.
            for cls in trackers:
                for object_id, centroid in trackers[cls].objects.items():
                    cv2.putText(frame, f"{cls}:{object_id}", (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            out.write(frame)

        cap.release()
        out.release()

        # The final count is the number of unique IDs recorded per class.
        final_counts = {cls: len(unique_ids[cls]) for cls in unique_ids}
        return output_video_path, final_counts

# Create an instance of YOLOModel for export
yolo_model = YOLOModel()

# Export the model as needed
__all__ = ['yolo_model']