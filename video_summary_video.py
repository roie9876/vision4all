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
import shutil

# Import your shared Azure OpenAI client
from azure_openai_client import client, DEPLOYMENT

# Local imports
from utils import (
    summarize_descriptions,
    extract_frames,  # we still use for sub-video frames
    call_azure_openai_with_retry  # <- import our new helper
)

load_dotenv()

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

    try:
        completion = call_azure_openai_with_retry({
            "model": DEPLOYMENT,
            "messages": chat_prompt,
            "max_tokens": 4096,
            "temperature": 0,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False
        })
        description = completion.choices[0].message.content
        if hasattr(completion, 'usage') and hasattr(completion.usage, 'total_tokens'):
            total_tokens = completion.usage.total_tokens
            st.write(f"Total tokens used for this completion: {total_tokens}")
        else:
            total_tokens = 0
    except Exception as e:
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description, total_tokens

def handle_image_upload(uploaded_file):
    img = Image.open(uploaded_file)  # Load the image here
    if img.mode == 'CMYK' or img.mode == 'RGBA':
        img = img.convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    base_name, _ = os.path.splitext(uploaded_file.name)
    frame_path = os.path.join(temp_dir, f"{base_name}.jpeg")
    img.save(frame_path, format='JPEG')
    return frame_path

def handle_video_upload(uploaded_file, frames_per_second):
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
    # ...existing code...

def summarize_image_analysis(image, description):
    global total_tokens_used  # Access the global token counter

    # Start timer
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0

    start_time = time.time()

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
                    "text": description
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
    response = call_azure_openai_with_retry({
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "messages": chat_prompt,
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "stream": False
    })

    # Try extracting usage if available
    if hasattr(response, "usage") and response.usage:
        st.session_state.total_tokens_used += response.usage.total_tokens

    # Parse the response
    try:
        summary_text = response.choices[0].message.content
    except Exception as e:
        summary_text = "Error occurred while summarizing the results."

    # End timer and calculate duration
    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Approximate cost calculation (example rate):
    cost_per_1k_tokens = 0.0015
    total_price = (st.session_state.total_tokens_used / 1000) * cost_per_1k_tokens

    return summary_text, elapsed_str, st.session_state.total_tokens_used, total_price

def run_video_summary():
    st.title("Video/Image Summary")
    # Delete 'temp_segments' folder at start
    temp_segments_dir = os.path.join(os.getcwd(), 'temp_segments')
    if os.path.exists(temp_segments_dir):
        shutil.rmtree(temp_segments_dir)

    uploaded_file = st.file_uploader(
        "Choose a video or image...",
        type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
        key="uploader1"
    )
    if uploaded_file is None:
        st.write("No file uploaded.")
        return

    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[1, 2, 0.5, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )

    content_prompt = st.text_input("Enter the content prompt:", value="Analyze the image and describe it in Hebrew. Focus on identifying and detailing cars, animals, and humans. try to determine the objects like cars or human or animal. For each detected object, provide as many details as possible: Cars: Describe the color, type, and model. Animals: Identify the type of animal. Humans: Describe what they are doing, what they are wearing, and if they have any weapons.even if object is not move, describe it in details.If the ai model is unsure whether an object is a cars, humans, or animals, make the best guess and provide an explanation.")

    if st.button("Process"):
        start_time = time.time()
        # 1. Save video
        temp_dir = os.path.join(os.getcwd(), 'temp_video')
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.video(video_path)

        # 2. Split video into segments
        segment_paths = split_video_into_segments(video_path, segment_length=10)
        descriptions = []
        total_tokens_sum = 0
        total_frames = 0

        # 3. Process each segment in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_segment, seg_path, sample_rate)
                for seg_path in segment_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                desc, tokens_used, frames_processed = future.result()
                descriptions.append(desc)
                total_tokens_sum += tokens_used
                total_frames += frames_processed

        # 4. Summarize
        summary_text = summarize_descriptions(descriptions, content_prompt=content_prompt)

        # 5. Display results
        st.write("### Summary:")
        st.write(summary_text)
        st.write(f"Total tokens used: {total_tokens_sum}")
        st.write(f"Total frames extracted: {total_frames}")
        price = calculate_price(total_tokens_sum)
        st.write(f"Price: ${price:.4f}")
        elapsed_time = time.time() - start_time
        st.write(f"Total time taken: {elapsed_time:.2f} seconds")

        # 6. Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_segments_dir):
            shutil.rmtree(temp_segments_dir)

def run_video_summary_split(uploaded_file):
    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[0.5, 1, 2, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )

    if uploaded_file.type.startswith("video"):
        video_path, frames = handle_video_upload(uploaded_file, sample_rate)
    else:
        frames = handle_image_upload(uploaded_file)

    if not frames:
        return "No frames were extracted. Nothing to analyze.", 0, 0

    total_frames = len(frames)

    # Analyze frames
    summary, elapsed_time, total_tokens_used = analyze_frames(frames)

    # Clean up
    if uploaded_file.type.startswith("video"):
        if os.path.exists(video_path):
            os.remove(video_path)

    return summary, elapsed_time, total_tokens_used

def batch_describe_images(images, content_prompt, batch_size=20):
    results = []
    # Concurrency for each batch
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {}
        for i in range(0, len(images), batch_size):
            # Build a single prompt with multiple images.
            chat_prompt = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }]
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            # Add prompt text for each image in the batch
            for idx, img in enumerate(images[i:i+batch_size]):
                # Resize and convert image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

                chat_prompt[1]["content"].append({
                    "type": "text",
                    "text": f"{content_prompt} (Image {i+idx+1})"
                })
                chat_prompt[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                })
            future = executor.submit(call_azure_openai_with_retry, {
                "model": DEPLOYMENT,
                "messages": chat_prompt,
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "stream": False
            })
            future_to_batch[future] = i
        for future in concurrent.futures.as_completed(future_to_batch):
            completion = future.result()
            description = completion.choices[0].message.content
            tokens_used = getattr(completion.usage, 'total_tokens', 0)
            results.append((description, tokens_used))
    return results

def analyze_frames(frames):
    start_time = time.time()
    descriptions = []
    total_tokens_used = 0
    # Use the updated batch_describe_images with concurrency
    batched_results = batch_describe_images(frames, "Describe in Hebrew:", batch_size=20)
    for desc_batch, tokens_used in batched_results:
        descriptions.extend(desc_batch)
        total_tokens_used += tokens_used
    summary = summarize_descriptions(descriptions)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return summary, elapsed_time, total_tokens_used

def split_video_into_segments(video_path, segment_length=10):
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
            segment_index += 1
        out.write(frame)
        frame_counter += 1
    if out:
        out.release()
    cap.release()
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
    return summary_text, tokens_used, len(frames_extracted)

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def calculate_price(tokens_used, rate_per_1000_tokens=0.0050):
    return tokens_used * rate_per_1000_tokens / 1000