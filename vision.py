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
# source ./venv/bin/activate
from azure_openai_client import client, DEPLOYMENT

# Local imports (your own modules)
from detected_objects import run_detect_objects
from search_object_in_image import run_search_object_in_image
from search_object_in_video import run_search_object_in_video
from detect_change_in_video_and_summary import run_detect_change_in_video_and_summary
from video_summary_image import run_image_summary, handle_image_upload
from video_summary_video import run_video_summary  # Add this import
from video_summary_with_object_count import run_video_summary_with_object_count

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

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Video Summary", "Detect Change in Video", "Video Summary with Object Count",
                                     "Detect Objects", "Search Object in Image", "Search Object in Video",
                                     "Image Summary"])
    
    if app_mode == "Video Summary":
        run_video_summary()
    elif app_mode == "Detect Change in Video":
        run_detect_change_in_video_and_summary()
    elif app_mode == "Video Summary with Object Count":
        run_video_summary_with_object_count()
    elif app_mode == "Detect Objects":
        run_detect_objects()
    elif app_mode == "Search Object in Image":
        run_search_object_in_image()
    elif app_mode == "Search Object in Video":
        run_search_object_in_video()
    elif app_mode == "Image Summary":
        run_image_summary()

if __name__ == "__main__":
    main()
