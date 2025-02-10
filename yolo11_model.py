import cv2
from ultralytics import solutions
import streamlit as st
import os

def count_objects_in_region(video_path, output_video_path, model_path):
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error reading video file: {video_path}")
        return None
    # Retrieve video properties.
    w, h, fps = (int(cap.get(x)) for x in 
                 (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Define a region (adjust as needed).
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    object_counts = {"human": 0, "car": 0, "animal": 0}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = counter.count(frame)
        video_writer.write(frame)

        # Update object counts
        for obj in counter.objects:
            cls_name = obj['name']
            if cls_name == 'person':
                object_counts["human"] += 1
            elif cls_name == 'car':
                object_counts["car"] += 1
            elif cls_name in ['dog', 'cat']:
                object_counts["animal"] += 1

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    return object_counts

def main():
    st.title("YOLO11 Object Counting")
    
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(video_path)
        
        output_video_path = os.path.join(temp_dir, f"processed_{uploaded_file.name}")
        model_path = "yolo11n.pt"  # Ensure this path is correct
        
        if st.button("Process Video"):
            object_counts = count_objects_in_region(video_path, output_video_path, model_path)
            
            st.video(output_video_path)
            st.success("Video processed successfully!")
            
            if object_counts:
                st.write("Object counts:")
                st.json(object_counts)

if __name__ == "__main__":
    main()
