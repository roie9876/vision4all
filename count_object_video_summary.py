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
from ultralytics import YOLO
import numpy as np
import json
import random
from datetime import datetime

# Import your shared Azure OpenAI client
from azure_openai_client import client, DEPLOYMENT

# Local imports
from utils import (
    summarize_descriptions,
    extract_frames  # we still use for sub-video frames
)
from yolo_model import yolo_model  # Ensure this import statement is correct

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
                {"type": "text", "text": "You are an AI assistant that helps people find information."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": "\n"}
            ]
        }
    ]

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
            st.write(f"Total tokens used for this completion: {total_tokens}")
        else:
            total_tokens = 0
    except Exception as e:
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description, total_tokens

def handle_image_upload(uploaded_file):
    img = Image.open(uploaded_file)
    if img.mode in ('CMYK', 'RGBA'):
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
    # (Your code for frame extraction can go here if needed)
    return video_path, frames_list

def summarize_image_analysis(image, description):
    global total_tokens_used  
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0

    start_time = time.time()
    image = resize_and_compress_image(image)

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    encoded_image = image_to_base64(image)

    chat_prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an AI assistant that helps people find information."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": description},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": "\n"}
            ]
        }
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    if hasattr(response, "usage") and response.usage:
        st.session_state.total_tokens_used += response.usage.total_tokens

    try:
        summary_text = response.choices[0].message.content
    except Exception as e:
        summary_text = "Error occurred while summarizing the results."

    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    cost_per_1k_tokens = 0.0015
    total_price = (st.session_state.total_tokens_used / 1000) * cost_per_1k_tokens

    return summary_text, elapsed_str, st.session_state.total_tokens_used, total_price

def split_video_into_segments(video_path, segment_length=10):
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

    summary_text, _, tokens_used, counts = analyze_frames(frames_extracted)
    return summary_text, tokens_used, len(frames_extracted), counts

def analyze_frames(frames):
    start_time = time.time()
    descriptions = []
    total_tokens_used_local = 0
    object_counts = {"human": 0, "car": 0, "animal": 0}
    batched_results = batch_describe_images(frames, "Describe in Hebrew:", batch_size=20)
    for desc_batch, tokens_used in batched_results:
        descriptions.extend(desc_batch)
        total_tokens_used_local += tokens_used
        for frame in frames:
            counts = yolo_model.count_objects(frame)
            for key in object_counts:
                object_counts[key] += counts.get(key, 0)
    summary = summarize_descriptions(descriptions)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return summary, elapsed_time, total_tokens_used_local, object_counts

def batch_describe_images(images, content_prompt, batch_size=20):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {}
        for i in range(0, len(images), batch_size):
            chat_prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an AI assistant that helps people find information."}]
                },
                {"role": "user", "content": []}
            ]
            for idx, img in enumerate(images[i:i+batch_size]):
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
            future = executor.submit(
                client.chat.completions.create,
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
            future_to_batch[future] = i
        for future in concurrent.futures.as_completed(future_to_batch):
            completion = future.result()
            description = completion.choices[0].message.content
            tokens_used = getattr(completion.usage, 'total_tokens', 0)
            results.append((description, tokens_used))
    return results

def random_start_time():
    day = random.randint(1, 6)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(2024, 10, day, hour, minute, second).isoformat()

def run_video_summary_with_object_count():
    st.title("Video Summary with Object Count")
    
    uploaded_files = st.file_uploader(
        "Choose video(s)...",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="uploader2"
    )
    if not uploaded_files:
        st.write("Please upload a video file to proceed.")
        return

    sample_rate = st.selectbox(
        "Select frame extraction rate:",
        options=[0.5, 1, 2, 4],
        format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
        index=1
    )
    
    st.markdown("### Loaded Videos")
    loaded_videos = [file.name for file in uploaded_files]
    st.text_area("Loaded Videos", "\n".join(loaded_videos), key="loaded_videos_area", height=150, disabled=True)
    
    processing_box = st.empty()
    finished_box = st.empty()
    
    processing_videos = []
    finished_videos = []
    
    # List to collect video info for JSON output
    json_results = []
    
    if st.button("Process Videos"):
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        update_counter = 0  # For unique keys
        
        # Process each video
        for uploaded_file in uploaded_files:
            video_name = uploaded_file.name
            processing_videos.append(video_name)
            
            update_counter += 1
            processing_box.empty()
            processing_box.text_area(
                "Processing Videos",
                "\n".join(processing_videos),
                key=f"processing_text_area_{update_counter}",
                height=150,
                disabled=True
            )
            
            # Save video temporarily (in temp_video folder)
            temp_dir = os.path.join(os.getcwd(), 'temp_video')
            os.makedirs(temp_dir, exist_ok=True)
            video_path = os.path.join(temp_dir, video_name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(video_path)
            
            # Process video with YOLO to obtain object counts.
            output_video_path = os.path.join(temp_dir, f"{video_name}_processed.webm")
            final_output_path, object_counts = yolo_model.process_video_with_counts(video_path, output_video_path)
            st.video(final_output_path)
            st.write("Object counts (YOLO):", object_counts)
            
            # Generate the video description summary via segmentation.
            segment_paths = split_video_into_segments(final_output_path, segment_length=10)
            descriptions = []
            total_tokens_sum = 0
            total_frames = 0
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_segment, seg_path, sample_rate)
                    for seg_path in segment_paths
                ]
                for future in concurrent.futures.as_completed(futures):
                    desc, tokens_used, frames_processed, counts = future.result()
                    descriptions.append(desc)
                    total_tokens_sum += tokens_used
                    total_frames += frames_processed
            
            summary_text = summarize_descriptions(descriptions)
            st.write("Video Description Summary:", summary_text)
            
            # Build JSON object for this video
            video_info = {
                "video_descriptions": summary_text,
                "video_url": "",
                "start_time": random_start_time(),
                "point_latitude": 33.12045961,
                "point_longitude": 35.18533746,
                "CameraID": f"id_{random.randint(1, 20)}",
                "video_name": video_name,
                "object_counts": object_counts
            }
            # Write JSON file with the same name as the video (e.g., 1.mp4 -> 1.json)
            base_name, _ = os.path.splitext(video_name)
            json_file_path = os.path.join(results_dir, f"{base_name}.json")
            with open(json_file_path, "w", encoding="utf-8") as jf:
                json.dump([video_info], jf, ensure_ascii=False, indent=2)
            
            processing_videos.remove(video_name)
            finished_videos.append(video_name)
            
            update_counter += 1
            processing_box.empty()
            processing_box.text_area(
                "Processing Videos",
                "\n".join(processing_videos),
                key=f"processing_text_area_{update_counter}",
                height=150,
                disabled=True
            )
            update_counter += 1
            finished_box.empty()
            finished_box.text_area(
                "Finished Videos",
                "\n".join(finished_videos),
                key=f"finished_text_area_{update_counter}",
                height=150,
                disabled=True
            )
            
            # Cleanup temporary video files.
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
        
        update_counter += 1
        processing_box.text("All videos processed!")

def main():
    # Uncomment one of the following to choose which functionality to run:
    # run_video_summary()         # For video/image summary without YOLO object counts
    run_video_summary_with_object_count()  # For video summary with YOLO object counts and description summary

if __name__ == "__main__":
    main()