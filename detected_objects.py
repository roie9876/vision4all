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

    # Allow multiple files
    uploaded_files = st.file_uploader(
        "Choose video files...", 
        type=["mp4", "avi", "mov", "mkv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Display loaded videos
        st.markdown("### Loaded Videos")
        loaded_videos = [uploaded_file.name for uploaded_file in uploaded_files]
        st.text_area("Loaded Videos", "\n".join(loaded_videos), height=200, disabled=True)

        if st.button("Process"):
            # Create placeholders for process status
            processing_box = st.empty()
            finished_box = st.empty()

            # Create directory for saving detected objects
            save_folder = os.path.join("detected objects", "processed_videos")
            os.makedirs(save_folder, exist_ok=True)

            processing_videos = []
            finished_videos = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for uploaded_file in uploaded_files:
                    video_name = uploaded_file.name
                    processing_videos.append(video_name)
                    loaded_videos.remove(video_name)
                    
                    processing_box.text_area("Processing Videos", "\n".join(processing_videos), height=200, disabled=True)
                    
                    # Save temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    frames = extract_frames(video_path, sample_rate=1.0)
                    # Detect objects in frames
                    objects_list = list(
                        executor.map(
                            lambda fp: detect_objects_in_image(Image.open(fp)),
                            frames
                        )
                    )
                    # Summarize using OpenAI
                    all_objects = []
                    for frame_objs in objects_list:
                        all_objects.extend(frame_objs)
                    combined_objects = " ".join(all_objects)
                    summary = summarize_text(combined_objects)
                    # Write to text file
                    text_file_path = os.path.join(save_folder, f"{video_name}.txt")
                    with open(text_file_path, "w", encoding="utf-8") as f:
                        f.write(summary)

                    processing_videos.remove(video_name)
                    finished_videos.append(video_name)
                    
                    finished_box.text_area("Finished Videos", "\n".join(finished_videos), height=200, disabled=True)

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
