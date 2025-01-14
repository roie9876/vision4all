import streamlit as st
import tempfile
import os
import logging
import io
import base64
import requests
from PIL import Image
from utils import extract_frames, detect_objects_in_image
from openai import AzureOpenAI
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

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

def run_detect_objects():
    st.title("Detect Objects in Video/Image")

    # Step 1: User uploads a video or image
    uploaded_file = st.file_uploader("Choose a video or image...", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if uploaded_file.type in ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"]:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.video(video_path)
            frames = extract_frames(video_path)
            st.write(f"Extracted {len(frames)} frames from the video.")
        else:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
            temp_dir = tempfile.mkdtemp()
            frame_path = os.path.join(temp_dir, f"{uploaded_file.name}.png")
            img.save(frame_path)
            frames = [frame_path]
            st.write("Uploaded image.")

        # Step 2: Detect objects in frames
        content_prompt = st.text_input(
            "Enter the content prompt:",
            value="List all objects in this image in Hebrew, for each object add description like color, shape, material etc in Hebrew"
        )
        objects_list = []
        for frame_path in frames:
            objects = detect_objects_in_image(Image.open(frame_path))
            objects_list.append(objects)
        st.markdown("<h3>Detected objects in frames</h3>", unsafe_allow_html=True)

        # Step 3: Display detected objects
        for i, objects in enumerate(objects_list):
            items_html = ''.join(f"<li>{obj}</li>" for obj in objects)
            st.markdown(
                f"<strong>Frame {i+1}:</strong><ul>{items_html}</ul>",
                unsafe_allow_html=True
            )
