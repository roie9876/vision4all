import cv2
import numpy as np
from ultralytics import YOLO
from tracker import CentroidTracker  # Assumes you have a working centroid tracker

class YOLOModel:
    def __init__(self):
        # Load the YOLOv8 model (adjust the model path if needed)
        self.model = YOLO('yolo11l.pt')
        # Classes we use for counting (this list is for direct counting without tracking)
        self.object_classes = ['car', 'person', 'dog', 'cat']
        self.object_class_ids = [cls_id for cls_id, cls_name in self.model.names.items() 
                                 if cls_name in self.object_classes]
        # Classes that you want to track over time
        self.tracked_classes = ['person', 'car', 'truck', 'motorcycle', 'boat', 'sheep', 'cow']

    def count_objects(self, frame):
        """Counts objects in a single frame without tracking."""
        results = self.model(frame, conf=0.6)
        detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls_id]
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
        """
        Processes the video, overlays tracking information, and returns final object counts.
        A matching threshold is used so that if a detection is too far away from the tracker's
        last known position (e.g. due to a camera jump), it is not associated with the tracker.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error opening video: {input_video_path}")
            return None

        # Prepare video writer using H.264 encoder (avc1)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        tracker = CentroidTracker(max_disappeared=30)
        # Use a dictionary to accumulate votes for each tracker (object ID)
        objectID_to_class_votes = {}

        # Define a matching threshold (in pixels). Adjust this value based on your scenario.
        match_threshold = 600  # For example, 100 pixels

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection on the current frame
            results = self.model(frame, conf=0.6)
            detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls_id]

            # Collect detections for tracked classes
            tracked_boxes = []
            class_for_box = []
            for obj in detections:
                x1, y1, x2, y2, conf, cls_id = obj
                cls_name = self.model.names[int(cls_id)]
                if cls_name in self.tracked_classes:
                    # Convert bounding box to (x, y, width, height)
                    tracked_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                    class_for_box.append(cls_name)

            # Update the tracker with the current detections
            objects = tracker.update(tracked_boxes)

            # To ensure one-to-one matching, keep track of which detections have been used
            used_detections = set()

            # For each tracked object, find the best matching detection (if any)
            for objectID, centroid in objects.items():
                best_idx = None
                best_dist = float('inf')
                for i, (bx, by, bw, bh) in enumerate(tracked_boxes):
                    if i in used_detections:
                        continue
                    center_x = bx + bw / 2
                    center_y = by + bh / 2
                    # Calculate Euclidean distance squared
                    dist = (centroid[0] - center_x) ** 2 + (centroid[1] - center_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                # Only accept the match if it is within the threshold
                if best_idx is not None and best_dist < (match_threshold ** 2):
                    used_detections.add(best_idx)
                    cls_name = class_for_box[best_idx]
                    # Record a vote for this object ID.
                    if objectID not in objectID_to_class_votes:
                        objectID_to_class_votes[objectID] = {}
                    objectID_to_class_votes[objectID][cls_name] = objectID_to_class_votes[objectID].get(cls_name, 0) + 1

                # Draw the tracker ID and centroid on the frame
                cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

            out.write(frame)

        cap.release()
        out.release()

        # Allow time for the output video to finalize.
        import time
        time.sleep(0.5)

        # After processing all frames, decide on the final class for each tracked object
        final_counts = {"human": 0, "car": 0, "animal": 0}
        for objectID, votes in objectID_to_class_votes.items():
            # Choose the class with the highest vote for this object
            final_class = max(votes, key=votes.get)
            if final_class == "person":
                final_counts["human"] += 1
            elif final_class in ["car", "truck", "motorcycle", "boat"]:
                final_counts["car"] += 1
            elif final_class in ["dog", "cat", "sheep", "cow"]:
                final_counts["animal"] += 1

        return output_video_path, final_counts

# Create an instance of YOLOModel for export
yolo_model = YOLOModel()

# Export the model as needed (for example, in other modules)
__all__ = ['yolo_model']