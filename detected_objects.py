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

    fps_options = [0.5, 1, 2, 5]
    selected_fps = st.selectbox("Choose frames per second (FPS)", fps_options, index=1)

    uploaded_file = st.file_uploader("Choose a video or image...", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    frames = None
    if uploaded_file is not None:
        if uploaded_file.type in ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"]:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.video(video_path)
            st.write(f"Extracting frames at {selected_fps} FPS.")
            frames = extract_frames(video_path, fps=selected_fps)
            st.write(f"Extracted {len(frames)} frames from the video.")
        else:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
            temp_dir = tempfile.mkdtemp()
            frame_path = os.path.join(temp_dir, f"{uploaded_file.name}.png")
            img.save(frame_path)
            frames = [frame_path]
            st.write("Uploaded image.")

    content_prompt = st.text_input(
        "Enter the content prompt:",
        value="List all objects in this image in Hebrew"
    )

    if st.button("Process") and frames is not None:
        objects_list = []
        for frame_path in frames:
            objects = detect_objects_in_image(Image.open(frame_path))
            objects_list.append(objects)

        # Removed the code that displays objects per frame

        # Aggregate all objects across frames
        all_objects = []
        for frame_objs in objects_list:
            all_objects.extend(frame_objs)
        unique_objects = list(set(all_objects))
        comma_sep_objects = ", ".join(unique_objects)

        # Summarize locally
        summary = summarize_descriptions([comma_sep_objects])

        st.markdown("<h3>All Objects (Comma-Separated)</h3>", unsafe_allow_html=True)
        st.write(comma_sep_objects)
        #st.markdown("<h3>Summary of All Objects</h3>", unsafe_allow_html=True)
        #st.write(summary)

def summarize_text(text):
    # ...logic from utils.summarize_text...
    # define a local variable to avoid the NameError
    summary = "תמצית"  # or any temporary placeholder
    return summary

def summarize_descriptions(descriptions):
    # ...logic from utils.summarize_descriptions...
    combined_text = " ".join(descriptions)
    initial_summary = summarize_text(combined_text)
    final_summary = summarize_text(initial_summary)
    return final_summary
