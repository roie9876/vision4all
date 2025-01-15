import streamlit as st
import tempfile
import os
import io
import base64
import requests
from PIL import Image
from utils import extract_frames, summarize_descriptions, extract_video_segment
from openai import AzureOpenAI
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import concurrent.futures
import logging

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log"),
    logging.StreamHandler()
])

# Test log message to ensure logging is working
logging.debug("Logging is configured correctly in video_summary.py")

# Load environment variables
load_dotenv()

# Initialize the Azure OpenAI client
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

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

def describe_image(image, content_prompt):
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
        description = completion.choices[0].message.content
    except Exception as e:
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description

def describe_images(images, content_prompt):
    descriptions = []
    for image in images:
        description = describe_image(image, content_prompt)
        descriptions.append(description)
    return descriptions

def describe_images_batch(images, content_prompt, batch_size=38):
    descriptions = []
    
    # Split images into batches
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        
        # Resize and compress images to reduce base64 size
        resized_images = [resize_and_compress_image(image) for image in batch]
        
        # Convert images to base64
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        encoded_images = [image_to_base64(image) for image in resized_images]

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
                        "text": content_prompt
                    },
                    {
                        "type": "image_urls",
                        "image_urls": [
                            {"url": f"data:image/jpeg;base64,{encoded_image}"} for encoded_image in encoded_images
                        ]
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
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(client.chat.completions.create,
                                           model=deployment,
                                           messages=chat_prompt,
                                           max_tokens=4096,
                                           temperature=0,
                                           top_p=0.95,
                                           frequency_penalty=0,
                                           presence_penalty=0,
                                           stop=None,
                                           stream=False) for _ in batch]
                for future in concurrent.futures.as_completed(futures):
                    completion = future.result()
                    batch_descriptions = [choice.message.content for choice in completion.choices]
                    descriptions.extend(batch_descriptions)
        except Exception as e:
            raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return descriptions

def handle_image_upload(uploaded_file):
    img = Image.open(uploaded_file)
    if img.mode == 'CMYK' or img.mode == 'RGBA':
        img = img.convert('RGB')  # Convert CMYK or RGBA to RGB
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    base_name, _ = os.path.splitext(uploaded_file.name)
    frame_path = os.path.join(temp_dir, f"{base_name}.jpeg")
    img.save(frame_path, format='JPEG')
    frames = [frame_path]
    return frames

def handle_video_upload(uploaded_file, sample_rate):
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    tfile = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)
    frames = extract_frames(video_path, sample_rate=sample_rate)
    return frames

def create_segments(total_frames, frame_rate, segment_length):
    if segment_length == 0:
        raise ValueError("Segment length must not be zero.")
    segments = []
    for i in range(0, total_frames, segment_length):
        start_frame = i
        end_frame = min(i + segment_length, total_frames)
        segments.append((start_frame, end_frame))
    return segments

def split_video_into_segments(video_path, segments_count=4):
    """
    Split the video into several equal segments based on the total duration.
    """
    import cv2
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust segments_count if it is greater than the total number of frames
    if segments_count > total_frames:
        segments_count = total_frames
    
    segment_length = total_frames // segments_count
    if segment_length == 0:
        segments_count = max(1, total_frames)  # Ensure at least one segment
        segment_length = total_frames // segments_count
        if segment_length == 0:
            raise ValueError("Segment length must not be zero. Reduce the number of segments or increase the video length.")
    
    segments = create_segments(total_frames, frame_rate, segment_length)

    logging.debug(f"Total frames: {total_frames}, Frame rate: {frame_rate}, Segment length: {segment_length}")

    segment_paths = []
    for idx, (start_frame, end_frame) in enumerate(segments):
        logging.debug(f"Creating segment {idx + 1}: start_frame={start_frame}, end_frame={end_frame}")
        temp_segment_path = extract_video_segment(video_path, start_frame / frame_rate, end_frame / frame_rate)
        if temp_segment_path:
            logging.debug(f"Segment {idx + 1} created at {temp_segment_path}")
            segment_paths.append(temp_segment_path)
        else:
            logging.debug(f"Segment {idx + 1} creation failed")
    video.release()
    logging.debug(f"Total segments created: {len(segment_paths)}")
    return segment_paths

def run_video_summary():
    st.title("Video/Image Summary")

    # Step 1: User uploads a video or image
    uploaded_file = st.file_uploader("Choose a video or image...", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        logging.debug(f"Uploaded file: {uploaded_file.name}")
        sample_rate = st.selectbox(
            "Select frame extraction rate:",
            options=[1, 2, 0.5, 4],
            format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
            index=0
        )
        logging.debug(f"Selected sample rate: {sample_rate}")
        start_time = time.time()
        if uploaded_file.type.startswith("video"):
            frames = handle_video_upload(uploaded_file, sample_rate)
            logging.debug(f"Extracted frames: {len(frames)}")

            # Example: Split the uploaded video into multiple segments and process concurrently
            segment_paths = split_video_into_segments(frames[0], segments_count=4)
            logging.debug(f"Segment paths: {segment_paths}")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                segment_summaries_futures = [executor.submit(summarize_video_segment, seg) for seg in segment_paths]
                segment_summaries = [future.result() for future in concurrent.futures.as_completed(segment_summaries_futures)]

            # Combine summaries from all segments
            combined_summary = " ".join(segment_summaries)
            st.write("Combined Summary from all segments:")
            st.write(combined_summary)

            # Clean up temporary segment files
            for segment_path in segment_paths:
                os.remove(segment_path)

        else:
            frames = handle_image_upload(uploaded_file)
            logging.debug(f"Extracted frames: {len(frames)}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process frames in parallel
            objects_list = list(
                executor.map(
                    lambda fp: detect_objects_in_image(Image.open(fp)),
                    frames
                )
            )

        # Step 2: Analyze frames and generate descriptions
        images = [Image.open(frame_path) for frame_path in frames]
        descriptions = describe_images_batch(images, "Describe what you see in Hebrew:")

        # Step 3: Summarize descriptions
        summary = summarize_descriptions(descriptions)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write("Summary:")
        st.write(summary)
        st.write(f"Time taken to analyze the video: {elapsed_time:.2f} seconds")

        # Display the time taken in the UI
        st.write(f"Time taken to process the video: {elapsed_time:.2f} seconds")

        # Clean up temporary files
        temp_dir = os.path.join(os.getcwd(), 'temp')
        for file_path in frames:
            os.remove(file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    else:
        st.write("No file uploaded.")

def summarize_frame(frame_image):
    # Implement your frame summarization logic here
    # For example, you can use an image captioning model
    return "A new object appeared in the frame"

def summarize_video_segment(segment_path):
    # Extract key frames from the video segment
    frames = extract_frames(segment_path, sample_rate=1)  # Sample 1 frame per second

    # Describe each frame
    images = [Image.open(frame_path) for frame_path in frames]
    descriptions = describe_images(images, "Describe what you see in Hebrew:")

    # Summarize the descriptions
    summary = summarize_descriptions(descriptions)
    return summary

def main():
    total_frames = 427  # This value should be dynamically set based on the actual number of frames extracted
    frame_rate = 25.0
    segment_length = 50  # Set the segment length to 50 frames
    segments = create_segments(total_frames, frame_rate, segment_length)
    for idx, (start_frame, end_frame) in enumerate(segments):
        print(f"Creating segment {idx + 1}: start_frame={start_frame}, end_frame={end_frame}")
        # ...existing code for segment creation...
        if start_frame < total_frames:
            print(f"Segment {idx + 1} created successfully")
        else:
            print(f"Segment {idx + 1} creation failed")
    print(f"Total segments created: {len(segments)}")
