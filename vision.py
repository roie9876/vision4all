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
import concurrent.futures
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import shutil  # Add this import for shutil

from dotenv import load_dotenv

# --------------------------------------------------
# Import the shared Azure OpenAI client
# and (optionally) the `deployment` name if needed.
from azure_openai_client import client, DEPLOYMENT

# Local imports (your own modules)
from detected_objects import run_detect_objects
from search_object_in_image import run_search_object_in_image
from search_object_in_video import run_search_object_in_video
from detect_change_in_video_and_summary import run_detect_change_in_video_and_summary
from video_summary_image import run_image_summary, handle_image_upload


from utils import (
    extract_frames,
    summarize_descriptions,
    detect_objects_in_image
    # We'll define describe_images_batch in this file now
)
from video_summary_video import (
    handle_video_upload,
    analyze_frames,
    split_video_into_segments,
    process_segment
)
# Load environment variables
load_dotenv()

# --------------------------------------------------
# Logging / Retry
# logging.basicConfig(
#     level=logging.DEBUG,  # Capture all logs
#     format='%(asctime)s - %(asctime)s - %(message)s',
#     handlers=[
#         logging.FileHandler("app.log"),
#         logging.StreamHandler()
#     ]
# )

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

# --------------------------------------------------
def run_video_summary():
    st.title("Video/Image Summary")

    uploaded_file = st.file_uploader(
        "Choose a video or image...",
        type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
        key="uploader1"   # <-- Add unique key
    )

    if uploaded_file is None:
        st.write("No file uploaded.")
        return

    # logging.debug(f"Uploaded file: {uploaded_file.name}")

    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[1, 2, 0.5, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=0
    )
    # logging.debug(f"Selected sample rate: {sample_rate}")

    if st.button("Process"):
        # Start timing if needed
        start_time = time.time()

        # 1. Save video to temporary location
        temp_dir = os.path.join(os.getcwd(), 'temp_video')
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # logging.debug(f"Video saved to {video_path}")

        # Display the uploaded video
        st.video(video_path)

        # 2. Split the video into segments
        segment_paths = split_video_into_segments(video_path, segment_length=10)
        # logging.debug(f"Segment paths: {segment_paths}")

        # 3. Process each segment in parallel
        descriptions = []
        total_tokens_sum = 0
        total_frames = 0  # Initialize total frames counter
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_segment, seg_path, sample_rate)
                for seg_path in segment_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                desc, tokens_used = future.result()  # Adjust to handle only two returned values
                descriptions.append(desc)
                total_tokens_sum += tokens_used
                # total_frames += frames_processed  # Remove this line if frames_processed is not returned

        # 4. Summarize the segment descriptions
        summary_text = summarize_descriptions(descriptions)

        # 5. Display results
        st.write("### Summary:")
        st.write(summary_text)
        st.write(f"Total tokens used: {total_tokens_sum}")
        # st.write(f"Total frames extracted: {total_frames}")  # Remove this line if frames_processed is not returned
        elapsed_time = time.time() - start_time
        st.write(f"Total time taken to analyze: {elapsed_time:.2f} seconds")  # Display total time taken
        # logging.info(f"Summary generated in {elapsed_time:.2f} seconds")

        # 6. Clean up temporary files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Use shutil.rmtree to delete the directory and its contents
        # logging.debug("Video summary process completed.")

# --------------------------------------------------
# Streamlit UI
st.sidebar.title("Select an Application")
app_selection = st.sidebar.radio(
    "Go to",
    (
        "Image Summary",
        "Video Summary",
        "Search Object in Image",
        "Detect Objects",
        "Search Object in Video",
        "Detect Change in Video and Summary"
    )
)

if app_selection == "Image Summary":
    st.write("Upload an image to generate a summary in Hebrew.")
    run_image_summary()

elif app_selection == "Video Summary":
    st.write("Upload a video to generate a summary in Hebrew.")
    run_video_summary()

elif app_selection == "Search Object in Image":
    st.write("Upload a reference image and a target image to detect if the specified object appears in both images.")
    run_search_object_in_image()

elif app_selection == "Detect Objects":
    st.write("Upload a video or image to detect and list all objects present.")
    run_detect_objects()

elif app_selection == "Search Object in Video":
    st.write("Upload a reference image, describe the object, and then upload a video to search for that object.")
    run_search_object_in_video()

elif app_selection == "Detect Change in Video and Summary":
    st.write("Upload a static webcam video capture to detect changes and summarize them.")
    run_detect_change_in_video_and_summary()
