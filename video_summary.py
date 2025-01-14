import streamlit as st
import tempfile
import os
import logging
import io
import base64
import requests
from PIL import Image
from utils import extract_frames, summarize_descriptions
from openai import AzureOpenAI
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

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
        
        # Log the number of images sent in this request
        logging.info(f"Sending {len(encoded_images)} images in this request to OpenAI")

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
    st.write("Uploaded image.")
    return frames

def handle_video_upload(uploaded_file, sample_rate):
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    tfile = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)
    frames = extract_frames(video_path, sample_rate=sample_rate)
    st.write(f"Extracted {len(frames)} frames from the video.")
    return frames

def run_video_summary():
    st.title("Video/Image Summary")

    # Step 1: User uploads a video or image
    uploaded_file = st.file_uploader("Choose a video or image...", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        sample_rate = st.selectbox(
            "Select frame extraction rate:",
            options=[1, 2, 0.5, 4],
            format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
            index=0
        )
        start_time = time.time()
        if uploaded_file.type.startswith("video"):
            frames = handle_video_upload(uploaded_file, sample_rate)
        else:
            frames = handle_image_upload(uploaded_file)

        # Step 2: Analyze frames and generate descriptions
        images = [Image.open(frame_path) for frame_path in frames]
        descriptions = describe_images_batch(images, "Describe what you see in Hebrew:")
        st.write("Analyzed frames and generated descriptions.")

        # Step 3: Summarize descriptions
        summary = summarize_descriptions(descriptions)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write("Summary:")
        st.write(summary)
        st.write(f"Time taken to analyze the video: {elapsed_time:.2f} seconds")
        logging.info(f"Summary generated. Time taken: {elapsed_time:.2f} seconds")  # Keep this line
        print(f"Debug: Time taken to analyze the video: {elapsed_time:.2f} seconds")

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
