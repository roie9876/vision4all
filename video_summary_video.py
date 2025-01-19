import streamlit as st
import tempfile
import os
import logging
import io
import base64
import requests
from PIL import Image
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import concurrent.futures
import cv2
import math

# Import your shared Azure OpenAI client
from azure_openai_client import client, DEPLOYMENT

# Local imports
from utils import (
    summarize_descriptions,
    extract_frames  # we still use for sub-video frames
)

load_dotenv()

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
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

total_tokens_used = 0

def resize_and_compress_image(image, max_size=(800, 800), quality=95):
    image.thumbnail(max_size, Image.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return Image.open(buffered)

def describe_image(image, content_prompt):
    global total_tokens_used

    image = resize_and_compress_image(image)

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    encoded_image = image_to_base64(image)

    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "\n"
                }
            ]
        }
    ]

    # logging.debug(f"Chat prompt: {chat_prompt}")

    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=chat_prompt,
            max_tokens=4096,
            temperature=0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        description = completion.choices[0].message.content
        if hasattr(completion, 'usage') and hasattr(completion.usage, 'total_tokens'):
            total_tokens = completion.usage.total_tokens
            # logging.info(f"Tokens used for this completion: {total_tokens}")
        else:
            # logging.warning("Total tokens used not found in the API response.")
            total_tokens = 0
    except Exception as e:
        # logging.error(f"Failed to generate completion. Error: {e}")
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description, total_tokens

def handle_video_upload(uploaded_file, frames_per_second):
    import os
    import io
    import base64
    import logging
    from PIL import Image
    # Read video frames using cv2
    temp_dir = os.path.join(os.getcwd(), 'temp_video')
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / frames_per_second)) if frames_per_second != 0 else 1
    frames_list = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_bgr)
            frames_list.append(pil_img)
        frame_count += 1
    cap.release()
    return video_path, frames_list

def analyze_frames(frames):
    start_time = time.time()

    descriptions = []
    total_tokens_used = 0
    for image in frames:
        description, tokens_used = describe_image(image, "Describe what you see in Hebrew:")
        descriptions.append(description)
        total_tokens_used += tokens_used

    summary = summarize_descriptions(descriptions)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return summary, elapsed_time, total_tokens_used

def split_video_into_segments(video_path, segment_length=10):
    # logging.debug("start split the video to segments")
    """
    Splits the video into segments of up to 'segment_length' seconds each,
    saves them in a local 'temp_segments' folder, and returns the segment paths.
    """
    import os
    import cv2
    folder_path = os.path.join(os.getcwd(), 'temp_segments')
    os.makedirs(folder_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_per_segment = segment_length * fps
    segment_paths = []
    segment_index = 0
    frame_counter = 0
    out = None

    # logging.debug(f"Splitting video into segments of {segment_length} seconds each.")
    # logging.debug(f"Frames per segment: {frames_per_segment}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % frames_per_segment == 0:
            if out:
                out.release()
            segment_filename = f"segment_{segment_index}.mp4"
            segment_path = os.path.join(folder_path, segment_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = frame.shape
            out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
            segment_paths.append(segment_path)
            # logging.debug(f"Created segment: {segment_path}")
            segment_index += 1
        out.write(frame)
        frame_counter += 1
    if out:
        out.release()
    cap.release()
    # logging.debug(f"Total segments created: {len(segment_paths)}")
    return segment_paths

def process_segment(segment_path, sample_rate):
    """
    Extract frames from a 10-second segment, describe them, and return the partial summary.
    """
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(segment_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / sample_rate)) if sample_rate != 0 else 1
    frames_extracted = []
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_bgr)
            frames_extracted.append(pil_img)
        success, frame = cap.read()
        frame_count += 1
    cap.release()

    # Analyze frames
    summary_text, _, tokens_used = analyze_frames(frames_extracted)
    return summary_text, tokens_used

def run_video_summary():
    st.title("Video Summary")

    # Restrict file upload to video formats
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is None:
        st.write("No video uploaded.")
        return

    # User selects how many frames per second to sample within each segment
    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[0.5, 1, 2, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )

    if st.button("Process"):
        # Log a message when user clicks 'Process'
        # logging.debug("Video file uploaded and sample rate selected.")

        # 1. Save the uploaded video to a temporary path
        temp_dir = os.path.join(os.getcwd(), 'temp_video')
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # logging.debug(f"Video saved to {video_path}")

        # 2. Call `split_video_into_segments` to split the video into 10-second segments
        segment_paths = split_video_into_segments(video_path, segment_length=10)
        # logging.debug(f"Segment paths: {segment_paths}")

        # 3. Process each segment in parallel
        descriptions = []
        total_tokens_sum = 0

        # Make sure `process_segment` is imported from video_summary_video.py
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_segment, seg_path, sample_rate)
                for seg_path in segment_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                desc, tokens_used = future.result()
                descriptions.append(desc)
                total_tokens_sum += tokens_used

        # 4. Summarize all partial descriptions
        summary_text = summarize_descriptions(descriptions)

        # 5. Display the final summary
        st.write("### Summary:")
        st.write(summary_text)
        st.write(f"Total tokens used: {total_tokens_sum}")

        # 6. Clean up the temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)
        # logging.debug("Video summary process completed.")
    st.title("Video Summary")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        st.write("No video uploaded.")
        return

    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[0.5, 1, 2, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )

    if st.button("Process"):
        # logging.debug("Video file uploaded and sample rate selected.")

        # Save the uploaded video to a temporary path
        temp_dir = os.path.join(os.getcwd(), 'temp_video')
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # logging.debug(f"Video saved to {video_path}")

        # Call split_video_into_segments to split the video into 10-second segments
        segment_paths = split_video_into_segments(video_path, segment_length=10)
        # logging.debug(f"Segment paths: {segment_paths}")

        # Process each segment in parallel
        descriptions = []
        total_tokens_sum = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_segment, seg_path, sample_rate) for seg_path in segment_paths]
            for future in concurrent.futures.as_completed(futures):
                desc, tokens = future.result()
                descriptions.append(desc)
                total_tokens_sum += tokens

        summary_text = summarize_descriptions(descriptions)
        st.write("Summary:")
        st.write(summary_text)
        st.write(f"Total tokens used: {total_tokens_sum}")

        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        # logging.debug("Video summary process completed.")

def run_video_summary_split(uploaded_file):
    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[0.5, 1, 2, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )
    # logging.debug(f"Selected sample rate: {sample_rate}")

    if uploaded_file.type.startswith("video"):
        video_path, frames = handle_video_upload(uploaded_file, sample_rate)
        # logging.debug(f"Extracted {len(frames)} frames from the video.")
    else:
        frames = handle_image_upload(uploaded_file)
        # logging.debug(f"Image-based frames: {len(frames)}")

    if not frames:
        return "No frames were extracted. Nothing to analyze.", 0, 0

    total_frames = len(frames)
    # logging.debug(f"Total frames extracted: {total_frames}")

    # Analyze frames
    summary, elapsed_time, total_tokens_used = analyze_frames(frames)

    # Clean up
    if uploaded_file.type.startswith("video"):
        if os.path.exists(video_path):
            os.remove(video_path)

    return summary, elapsed_time, total_tokens_used
