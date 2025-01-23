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
from video_summary_video import run_video_summary  # Add this import

# Load environment variables
load_dotenv()

# --------------------------------------------------
# Logging / Retry
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
