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
import concurrent.futures

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
        value="List objects in this image in Hebrew. Include: אנשים, סוגי יחדות, כומות, דרגות, סוג מדים, כלים צבאיים, סוגי נשקים, האם זה יום או לילה, מקום דור או פתוח, סוג מבנה, נוף הצילום כמו מדבר או ירוק"
    )

    if st.button("Process") and frames is not None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            objects_list = list(
                executor.map(
                    lambda fp: detect_objects_in_image(Image.open(fp)), 
                    frames
                )
            )
        
        # Aggregate all objects across frames using OpenAI summarization
        all_objects = []
        for frame_objs in objects_list:
            all_objects.extend(frame_objs)
        combined_objects = " ".join(all_objects)

        # Summarize locally to remove duplicates
        summary = summarize_text(combined_objects)

        st.markdown("<h3>All Objects (Comma-Separated)</h3>", unsafe_allow_html=True)
        st.write(summary)
        #st.markdown("<h3>Summary of All Objects</h3>", unsafe_allow_html=True)
        #st.write(summary)

def summarize_text(text):
    logging.info("Summarizing text")
    # Headers and payload for the request
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_OPENAI_KEY"),
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role": "user",
                "content": f"Summarize the following text in Hebrew in comma separated format, ensuring no duplicates: {text}"
            }
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    # Send request
    try:
        response = http.post(os.getenv("AZURE_OPENAI_ENDPOINT"), headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        logging.error(f"Failed to make the request. Error: {e}")
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Extract summary from response
    summary = response.json()['choices'][0]['message']['content']
    logging.info("Summary generated")
    return summary

def summarize_descriptions(descriptions):
    logging.info("Summarizing descriptions")
    combined_text = " ".join(descriptions)
    initial_summary = summarize_text(combined_text)
    final_summary = summarize_text(initial_summary)
    return final_summary
