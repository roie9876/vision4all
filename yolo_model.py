import cv2
from ultralytics import YOLO
import numpy as np
import os

class YOLOModel:
    def process_video(self, input_video_path, output_video_path):
        model = YOLO('yolov8s.pt')
        
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_video_path.replace('.mp4', '.webm'), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        object_classes = ['car', 'person', 'dog', 'cat']
        object_class_ids = [cls_id for cls_id, cls_name in model.names.items() if cls_name in object_classes]

        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        writing_segment = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                results = model(frame)
                detected_objects = results[0].boxes.data.cpu().numpy()
                detected_objects = [obj for obj in detected_objects if int(obj[5]) in object_class_ids]

                if detected_objects:
                    out.write(frame)
                    writing_segment = True

                for obj in detected_objects:
                    x1, y1, x2, y2, conf, cls = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], model.names[int(obj[5])]
                    label = f'{cls} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                if writing_segment:
                    writing_segment = False

            prev_gray = gray.copy()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return output_video_path.replace('.mp4', '.webm')

# Create an instance of YOLOModel
yolo_model = YOLOModel()

# Ensure yolo_model is exported
__all__ = ['yolo_model']