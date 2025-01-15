import io
import base64
import requests
import logging
import tempfile
import cv2
from PIL import Image
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("dotenv package not found. Skipping loading environment variables from .env file.")

# Configuration
API_KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Setup retry strategy
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

def extract_frames(video_path, fps=1.0):
    """
    Extract frames from a video at the specified fps.
    """
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate // fps) if fps > 0 else 1
    frames = []
    success, image = video.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, image)
            frames.append(temp_file.name)
        success, image = video.read()
        count += 1
    video.release()
    return frames

def extract_video_segment(video_path, start_time, end_time):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clamp timestamps
    if start_time < 0:
        start_time = 0
    if end_time < 0:
        end_time = 0
    if end_time <= start_time:
        logging.error("Invalid segment times")
        video.release()
        return None

    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    if start_frame >= total_frames:
        logging.error("Start frame exceeds total frames")
        video.release()
        return None
    if end_frame > total_frames:
        end_frame = total_frames

    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    segment_frames = []
    for _ in range(start_frame, end_frame):
        success, image = video.read()
        if not success or image is None:
            break
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, image)
        segment_frames.append(temp_file.name)
    video.release()
    
    if not segment_frames:
        logging.error("Failed to extract video segment: no frames extracted")
        return None
    
    first_frame = cv2.imread(segment_frames[0])
    segment_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (first_frame.shape[1], first_frame.shape[0]))
    for frame_path in segment_frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    
    # Clean up extracted frames
    for frame_path in segment_frames:
        os.remove(frame_path)
    
    return segment_path

def summarize_text(text):
    logging.info("Summarizing text")
    # Headers and payload for the request
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role": "user",
                "content": f"Summarize the following text in Hebrew: {text}"
            }
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    # Send request
    try:
        response = http.post(ENDPOINT, headers=headers, json=payload)
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

def detect_objects_in_image(image):
    logging.info("Detecting objects in image")
    # Convert image to bytes and encode to base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr).decode('ascii')

    # Headers and payload for the request
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        "messages": [
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
                        "text": "List all objects in this image in Hebrew"
                    },
                    {
                        "type": "text",
                        "text": "\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "\n"
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    # Send request
    try:
        response = http.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        logging.error(f"Failed to make the request. Error: {e}")
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Extract objects from response
    objects = response.json()['choices'][0]['message']['content']
    logging.info("Objects detected in image")
    return objects.split(", ")

def image_similarity(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    # Compute SSIM between two images
    score, _ = structural_similarity(img1_gray, img2_gray, full=True)
    return score > 0.9
