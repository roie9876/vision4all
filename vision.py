import streamlit as st
from PIL import Image
import requests
import io
import base64
import cv2  # OpenCV for video processing
import tempfile
import logging
import time
import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
import concurrent.futures  # <-- Add this import
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
# from video_summary import run_video_summary  # <-- remove this line
from detected_objects import run_detect_objects  # <-- updated import
from search_object_in_image import run_search_object_in_image  # <-- updated import
from dotenv import load_dotenv  # <-- new import
from utils import extract_frames, summarize_descriptions, detect_objects_in_image  # <-- updated import
from search_object_in_video import run_search_object_in_video  # <-- new import
from detect_change_in_video_and_summary import run_detect_change_in_video_and_summary  # <-- new import
from video_summary import handle_image_upload, handle_video_upload, describe_images, split_video_into_segments, describe_images_batch, summarize_video_segment  # <-- updated import

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Remove or modify the logging configuration
logging.basicConfig(level=logging.CRITICAL)  # Change logging level to CRITICAL to suppress DEBUG logs

# Setup retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def run_video_summary():
    st.title("Video/Image Summary")

    # Step 1: User uploads a video or image
    uploaded_file = st.file_uploader("Choose a video or image...", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        logging.debug(f"Uploaded file: {uploaded_file.name}")
        sample_rate = st.selectbox(
            "Select frame extraction rate:",
            options=[1, 2, 0.5, 4],
            format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
            index=0
        )
        logging.debug(f"Selected sample rate: {sample_rate}")
        start_time = time.time()
        if uploaded_file.type.startswith("video"):
            frames = handle_video_upload(uploaded_file, sample_rate)
            logging.debug(f"Extracted frames: {len(frames)}")

            if frames:
                # Example: Split the uploaded video into multiple segments and process concurrently
                segment_paths = split_video_into_segments(frames[0], segments_count=4)
                logging.debug(f"Segment paths: {segment_paths}")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    segment_summaries_futures = [executor.submit(summarize_video_segment, seg) for seg in segment_paths]
                    segment_summaries = [future.result() for future in concurrent.futures.as_completed(segment_summaries_futures)]

                # Combine summaries from all segments
                combined_summary = " ".join(segment_summaries)
                st.write("Combined Summary from all segments:")
                st.write(combined_summary)

                # Clean up temporary segment files
                for segment_path in segment_paths:
                    os.remove(segment_path)
            else:
                logging.debug("No frames extracted from the video.")
        else:
            frames = handle_image_upload(uploaded_file)
            logging.debug(f"Extracted frames: {len(frames)}")

        if frames:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Process frames in parallel
                objects_list = list(
                    executor.map(
                        lambda fp: detect_objects_in_image(Image.open(fp)),
                        frames
                    )
                )

            # Step 2: Analyze frames and generate descriptions
            images = [Image.open(frame_path) for frame_path in frames]
            descriptions = describe_images_batch(images, "Describe what you see in Hebrew:")

            # Step 3: Summarize descriptions
            summary = summarize_descriptions(descriptions)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write("Summary:")
            st.write(summary)
            st.write(f"Time taken to analyze the video: {elapsed_time:.2f} seconds")

            # Display the time taken in the UI
            st.write(f"Time taken to process the video: {elapsed_time:.2f} seconds")

            # Clean up temporary files
            temp_dir = os.path.join(os.getcwd(), 'temp')
            for file_path in frames:
                os.remove(file_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        else:
            logging.debug("No frames extracted from the image.")
    else:
        st.write("No file uploaded.")

# Streamlit UI
#st.title("Image/Video Description in Hebrew")

st.sidebar.title("Select an Application")
app_selection = st.sidebar.radio("Go to", ("Video and Image Summary", "Search Object in Image", "Detect Objects", "Search Object in Video", "Detect Change in Video and Summary"))

if app_selection == "Video and Image Summary":
    #st.subheader("Video and Image Summary")
    st.write("Upload a video or image to generate a summary description in Hebrew.")
    run_video_summary()
elif app_selection == "Search Object in Image":
    #st.subheader("Search Object in Image")
    st.write("Upload a reference image and a target image to detect if the specified object appears in both images.")
    run_search_object_in_image()
elif app_selection == "Detect Objects":
    #st.subheader("Detect Objects in Video/Image")
    st.write("Upload a video or image to detect and list all objects present.")
    run_detect_objects()
elif app_selection == "Search Object in Video":
    #st.subheader("Search Object in Video")
    st.write("Upload a reference image, describe the object, and then upload a video to search for that object.")
    run_search_object_in_video()
elif app_selection == "Detect Change in Video and Summary":
    #st.subheader("Detect Change in Video and Summary")
    st.write("Upload a static webcam video capture to detect changes and summarize them.")
    run_detect_change_in_video_and_summary()