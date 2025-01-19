import streamlit as st
import tempfile
import os
import logging
import io
import base64
from PIL import Image
from utils import extract_frames, extract_video_segment
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import yolo_model
from openai import AzureOpenAI
import time
import cv2  # Add this import for OpenCV

# Setup logging configuration
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
#     logging.FileHandler("app.log"),
#     logging.StreamHandler()
# ])

# Test log message to ensure logging is working
# logging.debug("Logging is configured correctly in detect_change_in_video_and_summary.py")

# Load environment variables
load_dotenv()

# Configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version=api_version
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

def resize_and_compress_image(image, max_size=(800, 800), quality=95):
    image.thumbnail(max_size, Image.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return Image.open(buffered)

def describe_image(image):
    # logging.info("Describing image")
    
    # Resize and compress the image to reduce base64 size
    image = resize_and_compress_image(image)

    # Convert image to base64
    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    encoded_image = image_to_base64(image)

    # Prepare the chat prompt
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
                    "text": "Describe what you see in Hebrew:"
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

    # Generate the completion
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=4096,
            temperature=0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        # Try extracting usage if available
        if hasattr(completion, "usage") and completion.usage:
            st.session_state.total_tokens_used += completion.usage.total_tokens

        description = completion.choices[0].message.content
        # logging.info("Image description received")
    except Exception as e:
        # logging.error(f"Failed to generate completion. Error: {e}")
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description

def describe_images(images):
    # logging.info("Describing images")
    descriptions = []
    for image in images:
        description = describe_image(image)
        descriptions.append(description)
    # logging.info("Image descriptions received")
    return descriptions

def summarize_text(text):
    # logging.info("Summarizing text with OpenAI")
    prompt = (
        "Summarize the following text in Hebrew. focus on pepole description, cars etc:\n" + text
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": subscription_key,
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    try:
        response = http.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()

        # Try extracting usage if available
        if "usage" in response.json():
            st.session_state.total_tokens_used += response.json()["usage"]["total_tokens"]

    except requests.RequestException as e:
        # logging.error(f"Failed to make the request. Error: {e}")
        raise SystemExit(f"Failed to make the request. Error: {e}")

    summary = response.json()['choices'][0]['message']['content']
    # logging.info("Text summarization received")
    return summary

def detect_changes_with_openai(frames):
    # logging.info("Detecting changes with OpenAI")
    descriptions = describe_images(frames)
    
    # Split descriptions into chunks to avoid exceeding token limits
    chunk_size = 10  # Adjust chunk size as needed
    chunks = [descriptions[i:i + chunk_size] for i in range(0, len(descriptions), chunk_size)]
    
    summaries = []
    for chunk in chunks:
        prompt = (
            "Analyze the following frames and describe only if there is a major change like car/people movement in Hebrew. for car describ details like color, model etc. describe people clothes. everything in hebrew:\n"
            "if there is no change you must write in English 'no change'\n" + "\n".join(chunk)
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": subscription_key,
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 4096
        }

        try:
            response = http.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        except requests.RequestException as e:
            # logging.error(f"Failed to make the request. Error: {e}")
            raise SystemExit(f"Failed to make the request. Error: {e}")

        result = response.json()['choices'][0]['message']['content']
        summaries.append(result)
        # logging.info("Change detection result received for a chunk")

    # Combine all summaries into a final summary
    final_summary = "\n".join(summaries)
    
    # Summarize the final summary text
    summarized_text = summarize_text(final_summary)
    return summarized_text

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"

def create_segments(total_frames, frame_rate, segment_length):
    segments = []
    for i in range(0, total_frames, segment_length):
        start_frame = i
        end_frame = min(i + segment_length, total_frames)
        segments.append((start_frame, end_frame))
    return segments

def run_detect_change_in_video_and_summary():
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    st.subheader("Detect Change in Video and Summary")
    st.write("Upload video capture to detect changes and summarize them.")

    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        start_time = time.time()
        # Remove the redundant total_tokens_used initialization
        # total_tokens_used = 0

        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name
        st.video(video_path)

        # Process the video with YOLO to get only changes
        output_video_path = os.path.join(os.path.dirname(video_path), "output.webm")
        final_output_path = yolo_model.process_video(video_path, output_video_path)
        st.video(final_output_path)

        # Sample rate selection
        sample_rate = st.selectbox(
            "Select frame extraction rate:",
            options=[1, 2, 0.5, 4],
            format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
            index=0
        )

        # Summarize the changes from output.webm
        frames = extract_frames(final_output_path, sample_rate=sample_rate)
        st.write(f"Extracted {len(frames)} frames from the video.")  # display frames count
        summary = detect_changes_with_openai([Image.open(f) for f in frames])
        
        # Display the summary aligned to the right
        st.markdown("<div style='text-align: right; direction: rtl;'>" + summary + "</div>", unsafe_allow_html=True)

        # Clean up extracted frames
        for frame_path in frames:
            os.remove(frame_path)
        os.remove(video_path)  # Delete the original uploaded video file
        os.remove(final_output_path)

        end_time = time.time()
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        cost_per_1k_tokens = 0.0015
        # Use st.session_state.total_tokens_used for price calculation
        total_price = (st.session_state.total_tokens_used / 1000) * cost_per_1k_tokens

        # Display results in Streamlit
        st.write(f"Total analysis time: {elapsed_str}")
        st.write(f"Total tokens used: {st.session_state.total_tokens_used}")
        st.write(f"Approximate cost: ${total_price:.4f}")

def main():
    video_path = "path_to_your_video_file.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    segment_length = total_frames // 4  # Adjusted to create 4 segments from the total frames

    segments = create_segments(total_frames, frame_rate, segment_length)
    for idx, (start_frame, end_frame) in enumerate(segments):
        print(f"Creating segment {idx + 1}: start_frame={start_frame}, end_frame={end_frame}")
        # ...existing code for segment creation...
        if start_frame < total_frames:
            print(f"Segment {idx + 1} created successfully")
        else:
            print(f"Segment {idx + 1} creation failed")
    print(f"Total segments created: {len(segments)}")
    cap.release()  # Release the video capture object

# ...existing code...
