#temp#
import streamlit as st
import tempfile
import os
import logging
# ---------- DEFAULT DEMO VIDEOS ----------
DEFAULT_BEFORE_VIDEO = "לפני - גובה עשרים מטר.mp4"
DEFAULT_AFTER_VIDEO  = "אחרי - גובה עשרים מטר.mp4"
# -----------------------------------------
import io
import base64
import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import concurrent.futures
import cv2
import shutil
import numpy as np  # <-- NEW
import re  # NEW – used for stripping ```json``` fences
from skimage.metrics import structural_similarity as ssim  # NEW
import torch, torchvision                                   # NEW
import openai

# source venv/bin/activate
# Import your shared Azure OpenAI client
from azure_openai_client import client, DEPLOYMENT
# ---------- GPT provider helpers ----------
def call_openai_com_with_retry(payload: dict, max_retries: int = 5, backoff: float = 2.0):
    """
    Send ChatCompletion to api.openai.com with retries.
    Forces model='o3' and uses OPENAI_API_KEY from environment.
    """
    import copy, os, time, logging
    import openai
    try:                                # OpenAI‑Python < 1.0
        from openai.error import OpenAIError
    except ImportError:                 # OpenAI‑Python ≥ 1.0
        from openai import OpenAIError

    payload = copy.deepcopy(payload)
    # Always use the `o3` model on api.openai.com
    payload["model"] = "o3"
    payload.pop("deployment_id", None)    # Azure-only key

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY לא הוגדר או לא אותחל – בדוק את קובץ .env")

    for attempt in range(1, max_retries + 1):
        try:
            # --- OpenAI‑Python backwards compatibility ---
            if hasattr(openai, "OpenAI"):          # new (≥1.0.0)
                client = openai.OpenAI(api_key=openai.api_key)
                return client.chat.completions.create(**payload)
            else:                                  # old (<1.0.0)
                return openai.ChatCompletion.create(**payload)
        except OpenAIError as e:
            logging.warning(f"OpenAI.com error: {e} (attempt {attempt}/{max_retries})")
            time.sleep(backoff * attempt)
    raise RuntimeError("Exceeded retries to api.openai.com")


def call_gpt_with_retry(payload: dict):
    """
    Dispatch to Azure OpenAI or api.openai.com according to
    st.session_state['api_provider'] (default 'Azure OpenAI').
    """
    provider = st.session_state.get("api_provider", "Azure OpenAI")
    if provider == "OpenAI.com":
        return call_openai_com_with_retry(payload)
    else:
        return call_azure_openai_with_retry(payload)
# ------------------------------------------

# Local imports
from utils import (
    summarize_descriptions,
    extract_frames,             # we still use for sub‑video frames
    call_azure_openai_with_retry  # Azure helper used by the dispatcher
)

load_dotenv()
st.write("API key starts with:", os.getenv("OPENAI_API_KEY")[:10])

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

# ---------- high-quality image parameters ----------
MAX_DIM_FOR_GPT = 2048        # longest side sent to GPT-4o
JPEG_QUALITY    = 95          # better quality for base64 encoding
# ---------------------------------------------------
# Minimum structural‑difference (1‑SSIM) לסינון רעש
MIN_SSIM_DIFF = 0.7   # 35 % difference threshold — reduces small colour/lighting artefactss



# ---------- NEW TILE-COMPARISON PROMPT (JSON) ----------



TILE_COMPARE_PROMPT = """
🟢 **התעלם לחלוטין**  
1. שינוי בצפיפות / זווית / צבע עשב או עשבוני.  
2. שינוי גוון, תאורה, צל, רעש מצלמה.  
3. ≤ 10 % פיקסלים שונים בכלל.  

🔴 **שינוי מהותי** – חייב לעמוד באחד:  
A. אובייקט קשיח (אבן, פסולת, ציוד, סימון כביש) הופיע/נעלם ומשתרע על ≥ 20 % משטח המסגרת.  
B. אובייקט קשיח זז ≥ 60 px **או** ≥ 30 % מגודלו.  

החזר JSON **תקני בלבד** (ללא מלל נוסף), למשל:

```json
{
  "change_detected": true,
  "reason": "אבן חדשה הופיעה",
  "bbox_before": [x1, y1, x2, y2],
  "bbox_after":  [x1p, y1p, x2p, y2p],
  "movement_px": 72,
  "changed_pixels_percent": 27,
  "confidence": 96
}
{
  "change_detected": false,
  "reason": "",
  "bbox_before": [],
  "bbox_after": [],
  "movement_px": 0,
  "changed_pixels_percent": 0,
  "confidence": 0
}
"""


# TILE_COMPARE_PROMPT = """
# אתה עוזר AI שתפקידו לעזור לאנשים למצוא מידע.
# בהקשר זה, תתבקש להשוות בין שתי תמונות חלקיות (tile) של אותו אזור שצולם מרחפן, שבו מופיעים כביש, צמחייה וסלעים.
# עליך להשוות בין התמונות ולדווח אך ורק על שינויים אשר מופעים בתוך המסגרת האדומה והם שינויים מהותיים בין התמונות.

# הגדרת "שינוי מהותי":
# הופעה – אובייקט בולט שקיים בתמונה אחת בלבד.
# היעלמות – אובייקט בולט שנעלם בתמונה השנייה.
# שינוי מיקום משמעותי – אובייקט בולט שקיים בשתי התמונות אך זז בצורה ניכרת (≥ 25 % מגודל האובייקט או ≥ 50 px).

# *** התעלם מהבדלי תאורה, צללים, שינויים בצבע, רעש דיגיטלי ותזוזה טבעית קלה של עלים או עשב ***

# החזר תגובה בפורמט JSON (בעברית):
# {
#   "change_detected": true/false,
#   "description": "תיאור קצר של השינוי (אם קיים)",
#   "confidence": 0-100,
#   "criteria": {
#     "object_definition": "אובייקט בולט הוא אובייקט שגודלו לפחות 20% משטח התמונה",
#     "movement_threshold": "שינוי מיקום ייחשב משמעותי אם האובייקט זז ב-≥ 25 % מגודלו או ב-≥ 50 px",
#     "ignore_small_objects": true,
#     "ignore_insignificant_changes": "שינויים בתאורה, צללים, צבע, רעש דיגיטלי ותזוזה טבעית קלה של צמחייה אינם נחשבים מהותיים"
#   }
# }

# אם לא זוהה שינוי מהותי החזר:
# {
#   "change_detected": false,
#   "description": "",
#   "confidence": 0
# }
# """
# Alias for backward compatibility
COMMON_HEBREW_PROMPT = TILE_COMPARE_PROMPT



SHARPNESS_THRESHOLD = 120.0     # NEW – default variance-of-Laplacian limit

def resize_and_compress_image(image, max_dim: int = MAX_DIM_FOR_GPT):
    """Return the same PIL image if it already fits within `max_dim`,
    otherwise down-scale (LANCZOS) so its longest side == max_dim.
    No premature JPEG save – compression happens only when converting
    to base64 for GPT, with quality=JPEG_QUALITY.
    """
    if max(image.size) <= max_dim:
        return image
    img = image.copy()
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img

def describe_image(image, content_prompt):
    global total_tokens_used
    image = resize_and_compress_image(image)

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=JPEG_QUALITY)
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
            ]
        }
    ]
    try:
        completion = call_gpt_with_retry({
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames_list.append(frame)
        frame_count += 1
    cap.release()
    return video_path, frames_list

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
        img.save(buffered, format="JPEG", quality=JPEG_QUALITY)
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
                }
            ]
        }
    ]
    # Generate the completion
    response = call_gpt_with_retry({
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
    prompt_text_area()          # << add UI once
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
        summary_text = summarize_descriptions(
            descriptions,
            content_prompt=get_user_prompt()
        )
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

def _resize_for_prompt(img: Image.Image, max_dim: int = MAX_DIM_FOR_GPT) -> Image.Image:
    """
    Returns a resized copy of PIL `img` such that the longest side ≤ `max_dim`
    (default 2 048 px for GPT-4o), preserving aspect ratio.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def batch_describe_images(images, content_prompt, batch_size=5):
    results = []
    # Concurrency for each batch
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {}
        for i in range(0, len(images), batch_size):
            # Build a single prompt with multiple images.
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
                    "content": []
                },
            ]
            # Add prompt text for each image in the batch
            for idx, img in enumerate(images[i:i+batch_size]):
                # Ensure RGB and resize to reduce payload
                img_small = _resize_for_prompt(img)        # longest side ≤ MAX_DIM_FOR_GPT (2048)
                if img_small.mode != "RGB":
                    img_small = img_small.convert("RGB")
                buffered = io.BytesIO()     # <- FIXED
                # Robust save – retry with default quality if first attempt fails
                try:
                    img_small.save(buffered, format="JPEG", quality=JPEG_QUALITY)
                except Exception:
                    buffered = io.BytesIO() # <- FIXED
                    img_small.save(buffered, format="JPEG", quality=JPEG_QUALITY)
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Validate base64 length; skip if suspiciously small
                if len(encoded_image) < 1000:
                    logging.warning("Skipped a frame: base64 too small – possible corrupted image.")
                    continue
                chat_prompt[1]["content"].append({
                    "type": "text",
                    "text": f"{content_prompt} (Image {i+idx+1})"
                })
                chat_prompt[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                })
            future = executor.submit(call_gpt_with_retry, {
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
    batched_results = batch_describe_images(
        frames, 
        COMMON_HEBREW_PROMPT, 
        batch_size=10
    )
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

# ---------- NEW HELPERS ----------
def _extract_frames(video_path: str, fps_target: float):
    """
    Return a list of PIL images sampled from the video at fps_target **and**
    passing the sharpness test.
    """
    import cv2, numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 1
    interval = int(round(fps_src / fps_target)) if fps_target else 1
    frames, idx = [], 0
    success, frame = cap.read()
    while success:
        if idx % interval == 0:
            if _is_frame_sharp(frame, float(st.session_state.get("sharp_th", SHARPNESS_THRESHOLD))):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        success, frame = cap.read()
        idx += 1
    cap.release()
    return frames

# --- Alignment helper ---
def _align_images(img_ref, img_to_align,
                  max_features: int | None = None,
                  lowe_ratio: float | None = None,
                  good_match: int = 50):
    """
    Align `img_to_align` (PIL.Image) to `img_ref` (PIL.Image) using ORB feature matching
    and homography (RANSAC).
    Returns a tuple: (aligned PIL.Image, number_of_inliers).
    If alignment fails, the original `img_to_align` is returned with 0 inliers.
    """
    import streamlit as st
    # Pull dynamic defaults from the UI (Advanced parameters)
    if max_features is None:
        max_features = int(st.session_state.get("orb_max_features", 1000))
    if lowe_ratio is None:
        lowe_ratio = float(st.session_state.get("lowe_ratio", 0.7))
    import cv2
    import numpy as np
    from PIL import Image
    # Convert to grayscale numpy arrays
    img1_gray = cv2.cvtColor(np.array(img_ref), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img_to_align), cv2.COLOR_RGB2GRAY)
    # Detect ORB key‑points and descriptors
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        return img_to_align, 0
    # Match descriptors using KNN + Lowe ratio test – more robust than simple cross‑check
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in knn_matches:
        # Lowe ratio test - stricter threshold to get only very confident matches
        if m.distance < lowe_ratio * n.distance:  # Stricter threshold (was 0.75)
            good_matches.append(m)
    # Require a minimum number of reliable matches
    if len(good_matches) < 10:
        return img_to_align, 0
    # Keep the best matches (lowest distance)
    good_matches = sorted(good_matches, key=lambda m: m.distance)[:good_match]
    # Build point arrays for homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Estimate homography with RANSAC
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return img_to_align, 0
    # Warp `img_to_align` so it matches `img_ref`
    inliers = int(mask.sum()) if mask is not None else 0
    h, w = img_ref.size[1], img_ref.size[0]  # PIL size is (w, h)
    aligned = cv2.warpPerspective(np.array(img_to_align), M, (w, h))
    return Image.fromarray(aligned), inliers

def _crop_to_overlap(img_ref: Image.Image, img_aligned: Image.Image, grid_size=(3, 3)):
    """
    Return the maximal overlapping area of `img_ref` and `img_aligned`
    after warp, then snap that rectangle so its width/height are divisible
    by `grid_size`. This guarantees that every tile boundary is identical
    in both crops.
    If no valid overlap exists the originals are returned.
    """
    import numpy as np
    a_ref   = np.array(img_ref)
    a_align = np.array(img_aligned)
    # valid pixels are non‑black (avoid warp padding)
    mask_ref   = np.any(a_ref   != 0, axis=2)
    mask_align = np.any(a_align != 0, axis=2)

    ys_ref, xs_ref     = np.where(mask_ref)
    ys_align, xs_align = np.where(mask_align)
    if xs_ref.size == 0 or ys_ref.size == 0 or xs_align.size == 0:
        return img_ref, img_aligned
    # bounding boxes
    x1_ref,  x2_ref  = xs_ref.min(),   xs_ref.max()
    y1_ref,  y2_ref  = ys_ref.min(),   ys_ref.max()
    x1_aln, x2_aln = xs_align.min(), xs_align.max()
    y1_aln, y2_aln = ys_align.min(), ys_align.max()
    # intersection rectangle
    left   = max(x1_ref,  x1_aln)
    top    = max(y1_ref,  y1_aln)
    right  = min(x2_ref,  x2_aln)
    bottom = min(y2_ref,  y2_aln)
    if right <= left or bottom <= top:
        return img_ref, img_aligned  # no overlap
    # snap dimensions to the grid
    cols, rows = grid_size
    width  = right - left + 1
    height = bottom - top + 1
    width  -= width  % cols
    height -= height % rows
    if width == 0 or height == 0:
        return img_ref, img_aligned  # fallback
    box = (left, top, left + width, top + height)
    return img_ref.crop(box), img_aligned.crop(box)

def _show_aligned_pairs(frames_before, frames_after, max_pairs: int = 5):
    """
    Align each pair of frames (up to max_pairs) and show the cropped overlap
    side‑by‑side in Streamlit. No OpenAI calls performed.
    """
    for idx, (f1, f2) in enumerate(zip(frames_before, frames_after), start=1):
        if idx > max_pairs:
            break
        aligned_f2, inliers = _align_images(f1, f2)
        ref_crop, aligned_crop = _crop_to_overlap(f1, aligned_f2, grid_size=(3,3))
        # Compose side‑by‑side
        comp = Image.new("RGB", (ref_crop.width + aligned_crop.width, ref_crop.height))
        comp.paste(ref_crop, (0, 0))
        comp.paste(aligned_crop, (ref_crop.width, 0))
        st.image(comp, caption=f"זוג {idx} – inliers={inliers}", use_container_width=True)

def _describe_ground_differences(frames_before, frames_after, content_prompt_he="תאר בעברית את השינויים בקרקע בין שתי התמונות."):
    """
    Detect visual changes (simple pixel difference) and ask GPT-Vision
    to describe only the frames with notable ground changes.
    """
    min_inliers = 10  # skip pairs that cannot be aligned reliably
    changed_frames = []
    for idx, (f1, f2) in enumerate(zip(frames_before, frames_after), start=1):
        # Align the "after" frame to the "before" frame
        aligned_f2, inliers = _align_images(f1, f2)
        if inliers < min_inliers:
            # alignment failed – skip this pair
            continue
        # Resize both images for fast diff
        a1 = np.array(f1.resize((320, 320)))
        a2 = np.array(aligned_f2.resize((320, 320)))
        # Compute absolute mean pixel difference
        diff = np.mean(np.abs(a1.astype(np.int16) - a2.astype(np.int16)))
        if diff > 10:  # threshold – tune as needed
            combo = Image.new("RGB", (a1.shape[1] * 2, a1.shape[0]))
            combo.paste(f1.resize((320, 320)), (0, 0))
            combo.paste(aligned_f2.resize((320, 0)))
            changed_frames.append(combo)
            st.image(combo, caption=f"זוג {idx} – diff={diff:.1f}, inliers={inliers}", use_container_width=True)
    if not changed_frames:
        return "לא נמצאו שינויים בקרקע בין הסרטונים.", 0
    # Re-use existing batched Azure helper
    batched = batch_describe_images(changed_frames, content_prompt_he)
    descriptions, tokens = [], 0
    for desc, tok in batched:
        descriptions.append(desc)
        tokens += tok
    summary = summarize_descriptions(descriptions, content_prompt_he + " ספק סיכום.")
    return summary, tokens
# ---------- END HELPERS ----------

# ---------- NEW STREAMLIT ENTRY ----------
def _run_ground_change_detection_legacy():
    st.title("השוואת שני סרטונים – זיהוי שינויי קרקע")
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("טעינת סרטון 'לפני'", type=["mp4", "avi", "mov", "mkv"], key="ground_before")
    with col2:
        after_file = st.file_uploader("טעינת סרטון 'אחרי'", type=["mp4", "avi", "mov", "mkv"], key="ground_after")
    fps_target = st.selectbox("קצב דגימת פריימים", [0.5, 1, 2], index=1)
    custom_prompt = st.text_input(
        "הנחיית תוכן (עברית)",
        value=TILE_COMPARE_PROMPT
    )
    if st.button("ניתוח השינויים") and before_file and after_file:
        # Save temp videos
        temp_dir = tempfile.mkdtemp(prefix="ground_change_")
        path_before = os.path.join(temp_dir, before_file.name)
        path_after = os.path.join(temp_dir, after_file.name)
        with open(path_before, "wb") as f: f.write(before_file.getbuffer())
        with open(path_after, "wb") as f: f.write(after_file.getbuffer())
        st.video(path_before, start_time=0)
        st.video(path_after, start_time=0)
        st.info("מחלץ פריימים...")
        frames_before = _extract_frames(path_before, fps_target)
        frames_after  = _extract_frames(path_after, fps_target)
        st.success("בחר/י זוגות לניתוח (ה-GPT יופעל רק על הזוגות המסומנים)")
        # --- build & display all pairs once ---
        pairs = []  # [(idx, ref_img, aligned_img)]
        for idx, (f1, f2) in enumerate(zip(frames_before, frames_after), start=1):
            aligned_f2, inliers = _align_images(f1, f2)
            ref_crop, aligned_crop = _crop_to_overlap(f1, aligned_f2)
            pairs.append((idx, ref_crop, aligned_crop, inliers))
        selected_ids = []
        for idx, ref_crop, aligned_crop, inliers in pairs:
            comp = Image.new("RGB", (ref_crop.width + aligned_crop.width, ref_crop.height))
            comp.paste(ref_crop, (0, 0))
            comp.paste(aligned_crop, (ref_crop.width, 0))
            st.image(comp, caption=f"זוג {idx} – inliers={inliers}", use_container_width=True)
            if st.checkbox(f"נתח זוג {idx}", key=f"chk_pair_{idx}"):
                selected_ids.append(idx)
        if st.button("Analyze selected pairs") and selected_ids:
            st.info("מריץ GPT-Vision על הזוגות שנבחרו...")
            for idx, ref_crop, aligned_crop, _ in pairs:
                if idx not in selected_ids:
                    continue
                # Convert both crops to base64
                def _b64(img):
                    buf = io.BytesIO()      # <- FIXED
                    img.save(buf, format="JPEG")
                    return base64.b64encode(buf.getvalue()).decode()
                chat_prompt = [
                    {"role": "system", "content": [{"type": "text", "text": "אתה עוזר בינה מלאכותית שמנתח שינויים באזור מוגדר בין שתי תמונות. דווח **רק** על שינויים מהותיים (הופעת/היעלמות אובייקטים, תזוזות גדולות, שינויי מבנה או קרקע) והתעלם משינויים זניחים כגון שינויים קלים בגוונים, תאורה, רעש מצלמה או תנועות עלים קטנות."}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(ref_crop)}"}},
                        {"type": "text", "text": "---"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(aligned_crop)}"}}
                    ]}
                ]
                resp = call_gpt_with_retry({
                    "model": DEPLOYMENT,
                    "messages": chat_prompt,
                    "max_tokens": 800,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "stream": False
                })
                diff_txt = resp.choices[0].message.content if resp else "שגיאה בקבלת תוצאה."
                st.markdown(f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">{diff_txt}</div>', unsafe_allow_html=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
# ---------- END STREAMLIT ENTRY ----------

# ------------------------- helpers -------------------------
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- STABLE-CHANGE FILTER ----------
def _is_stable_change(idx, r, c, cube, th=MIN_SSIM_DIFF, win: int = 1):
    """
    Return True iff tile (r,c) at pair `idx` exceeds `th` **and** the same tile
    exceeds the threshold in at least one neighbouring pair within ±`win` frames.
    Filters out momentary artefacts (e.g. leaves swaying in just one frame).
    """
    if cube[idx, r, c] < th:
        return False
    lo = max(0, idx - win)
    hi = min(cube.shape[0] - 1, idx + win)
    for j in range(lo, hi + 1):
        if j != idx and cube[j, r, c] >= th:
            return True
    return False

# ---------- DIFF‑CUBE (temporal persistence) ----------
def _build_diff_cube(frames_before, frames_after, grid_size=(4, 4)):
    """
    Returns a NumPy array of shape (num_pairs, rows, cols) where each cell holds
    1‑SSIM for that tile.  We later use this cube to ignore transient artefacts
    that appear in only a single frame.
    """
    rows, cols = grid_size
    n = len(frames_before)
    cube = np.zeros((n, rows, cols), dtype=np.float32)
    for idx, (f1, f2) in enumerate(zip(frames_before, frames_after)):
        aligned, _ = _align_images(f1, f2)
        f1c, f2c = _crop_to_overlap(f1, aligned, grid_size)
        w, h = f1c.size
        tw, th = w // cols, h // rows
        ref_gray = cv2.cvtColor(np.array(f1c), cv2.COLOR_RGB2GRAY)
        aln_gray = cv2.cvtColor(np.array(f2c), cv2.COLOR_RGB2GRAY)
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * th, (r + 1) * th
                x0, x1 = c * tw, (c + 1) * tw
                t1 = ref_gray[y0:y1, x0:x1]
                t2 = aln_gray[y0:y1, x0:x1]
                cube[idx, r, c] = 1.0 - ssim(t1, t2)
    return cube

# -------------------------------------------------------

def _build_aligned_pairs(path_before: str, path_after: str, fps_target: float):
    """
    Heavy routine – extract frames & build aligned pairs once.
    Stored in st.session_state to avoid recomputation on every rerun.
    Returns list[dict] with keys: idx, comp_img, b64_1, b64_2, inliers
    """
    frames_before = _extract_frames(path_before, fps_target)
    frames_after  = _extract_frames(path_after,  fps_target)
    pairs = []
    for idx, (f1, f2) in enumerate(zip(frames_before, frames_after), start=1):
        aligned_f2, inliers = _align_images(f1, f2)
        ref_crop, aligned_crop = _crop_to_overlap(f1, aligned_f2, grid_size=(3,3))
        # Compose side-by-side picture for display once
        comp = Image.new("RGB", (ref_crop.width + aligned_crop.width, ref_crop.height))
        comp.paste(ref_crop, (0, 0))
        comp.paste(aligned_crop, (ref_crop.width, 0))
        pairs.append({
            "idx": idx,
            "comp": comp,
            "b64_1": _pil_to_b64(ref_crop),
            "b64_2": _pil_to_b64(aligned_crop),
            "ref":  ref_crop,            # <-- NEW
            "aligned": aligned_crop,     # <-- NEW
            "inliers": inliers
        })
    # Build the diff‑cube with the same grid size chosen by the user
    if "diff_cube" not in st.session_state:
        grid_val = int(st.session_state.get("grid_size", 3))
        st.session_state.diff_cube = _build_diff_cube(
            frames_before,
            frames_after,
            grid_size=(grid_val, grid_val)
        )
    return pairs
# -----------------------------------------------------------

def run_ground_change_detection():
    st.title("השוואת שני סרטונים – זיהוי שינויי קרקע")
    # --- session‑level list of change events we discover on the fly ---
    if "change_events" not in st.session_state:
        st.session_state.change_events = []   # each item: {"idx": int, "time": float}
    # ...UI for file upload & fps select – unchanged...
    # ------------------------------------------------------

    # ---------- file-upload UI & parameters ----------
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("טעינת סרטון 'לפני'",
                                       type=["mp4", "avi", "mov", "mkv"],
                                       key="ground_before")
    with col2:
        after_file = st.file_uploader("טעינת סרטון 'אחרי'",
                                      type=["mp4", "avi", "mov", "mkv"],
                                      key="ground_after")

    # --- fallback to local demo files if user did not upload ---
    default_before_path = os.path.abspath(DEFAULT_BEFORE_VIDEO)
    default_after_path  = os.path.abspath(DEFAULT_AFTER_VIDEO)
    if before_file is None and os.path.exists(default_before_path):
        before_file = open(default_before_path, "rb")
    if after_file is None and os.path.exists(default_after_path):
        after_file = open(default_after_path, "rb")
    # -----------------------------------------------------------
    fps_target = st.selectbox("קצב דגימת פריימים", [0.5, 1, 2], index=1)
    # --- unified prompt input -------------------------------
    custom_prompt = st.text_area(
        "הנחיית תוכן (עברית)",
        value=COMMON_HEBREW_PROMPT,
        height=300
    )
    # Add UI for the new parameters
    with st.expander("Advanced parameters"):
        st.session_state["api_provider"] = st.radio(
        "GPT API Provider",
        ["Azure OpenAI", "OpenAI.com"],
        index=0
    )
        st.session_state["MIN_SSIM_DIFF"] = st.number_input(
            "MIN_SSIM_DIFF",
            0.0,
            1.0,
            0.65
        )
        st.session_state["grid_size"] = st.number_input(
            "Grid size",
            min_value=1,
            value=5
        )
        st.session_state["top_k"] = st.number_input(
            "Top K",
            min_value=1,
            value=30
        )
        st.session_state["stable_window"] = st.number_input(
            "Stable window",
            min_value=0,
            value=1
        )
        st.session_state["sharp_th"] = st.number_input(
            "Sharpness threshold (variance-of-Laplacian)",
            min_value=0.0,
            value=SHARPNESS_THRESHOLD
        )
        st.session_state["use_segmentation"] = True  # Segmentation filter ON
        st.session_state["seg_score_thr"] = st.slider(
            "Mask-R CNN score threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.10,
            step=0.05
        )
        st.session_state["seg_iou_thr"] = st.slider(
            "Segmentation IoU threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.10,
            step=0.05
        )
        # --- NEW UI parameters ---
        st.session_state["orb_max_features"] = st.number_input(
            "ORB max features",
            min_value=100,
            max_value=5000,
            value=int(st.session_state.get("orb_max_features", 1000)),
            step=100
        )
        st.session_state["lowe_ratio"] = st.slider(
            "Lowe‑ratio threshold",
            min_value=0.50,
            max_value=0.95,
            value=float(st.session_state.get("lowe_ratio", 0.70)),
            step=0.05
        )
        st.session_state["diff_mask_thr"] = st.number_input(
            "Diff‑mask threshold (% pixels changed)",
            min_value=0.5,
            max_value=50.0,
            value=40.0,
            step=0.5,
            format="%.1f"
        )
        st.session_state["show_diff_tiles"] = st.checkbox("Show tiles after Diff‑mask filter (debug)")
        # --- END NEW UI parameters ---
        st.session_state["show_before_tiles"] = st.checkbox("Show tiles before SSIM filter (debug)")
        st.session_state["show_pre_tiles"]  = st.checkbox("Show tiles after SSIM filter (debug)")
        st.session_state["show_post_tiles"] = st.checkbox("Show tiles after Mask-R CNN filter (debug)")
        st.session_state["show_tile_stats"] = st.checkbox(
            "Show tile counts at each stage (debug)"
        )
        st.session_state["show_tile_debug"] = st.checkbox(
        "Show GPT tile debug (tile + analysis text)",
        value=False
)
    if st.button("הכן זוגות") and before_file is not None and after_file is not None:
        # clean old state
        for k in list(st.session_state.keys()):
            if k.startswith("chk_pair_"):
                st.session_state.pop(k)
        # save videos to temp dir
        tmp = tempfile.mkdtemp(prefix="ground_change_")

        # Determine filenames, stripping any absolute path for local fallback objects
        before_filename = os.path.basename(getattr(before_file, "name", DEFAULT_BEFORE_VIDEO))
        after_filename  = os.path.basename(getattr(after_file,  "name", DEFAULT_AFTER_VIDEO))
        path_before = os.path.join(tmp, before_filename)
        path_after  = os.path.join(tmp,  after_filename)

        # If the file was uploaded through Streamlit it has getbuffer(); if it's a local file-handle we copy it.
        if hasattr(before_file, "getbuffer"):
            with open(path_before, "wb") as f:
                f.write(before_file.getbuffer())
        else:  # local fallback
            shutil.copyfile(default_before_path, path_before)

        if hasattr(after_file, "getbuffer"):
            with open(path_after, "wb") as f:
                f.write(after_file.getbuffer())
        else:
            shutil.copyfile(default_after_path, path_after)

        # store paths so we can replay video segments later
        st.session_state.path_before = path_before
        st.session_state.path_after  = path_after
        # heavy compute – run once and cache in session_state
        with st.spinner("מחלץ ומיישר פריימים ..."):
            st.session_state.ground_pairs = _build_aligned_pairs(path_before, path_after, fps_target)
        st.success("הזוגות מוכנים! סמן/י ונתח.")
        # keep temp dir so crops stay valid during session
        st.session_state.temp_dir_gc = tmp

    # ---------- show pairs & final report ----------
    if "ground_pairs" in st.session_state:
        # Two tabs: analysis view and final report
        tab_analysis, tab_report = st.tabs(["🔍 ניתוח", "📄 דו\"ח סופי"])

        # -------- ANALYSIS TAB --------
        with tab_analysis:
            selected_ids = []
            for pair in st.session_state.ground_pairs:
                idx = pair["idx"]
                container = st.container()
                with container:
                    # side-by-side pair (no red overlay)
                    comp_img = _compose_pair(pair["ref"], pair["aligned"], draw_seg=False)
                    st.image(comp_img,
                             caption=f"זוג {idx} – inliers={pair['inliers']}",
                             use_container_width=True)
                    # selection check-box
                    if st.checkbox(f"נתח זוג {idx}", key=f"chk_pair_{idx}"):
                        selected_ids.append(idx)
                    # placeholder for GPT result
                    txt_key = f"gpt_txt_{idx}"
                    st.markdown(st.session_state.get(txt_key, ""),
                                unsafe_allow_html=True)

            # run GPT analysis on the chosen pairs
            if st.button("Analyze selected pairs") and selected_ids:
                _run_pairs_analysis(selected_ids, custom_prompt)  # ← existing helper

        # -------- FINAL-REPORT TAB --------
        with tab_report:
            report = st.session_state.get("report_data", [])
            if not report:
                st.write("אין ממצאים להצגה עדיין.")
            else:
                for entry in report:
                    st.image(f"data:image/jpeg;base64,{entry['pair_b64']}",
                             caption=f"זוג {entry['pair_idx']}",
                             use_container_width=True)
                    st.image(f"data:image/jpeg;base64,{entry['tile_b64']}",
                             caption=entry["text"],
                             use_container_width=True)

def _extract_focused_regions(img_ref, img_aligned,
                             grid_size=(3, 3), top_k: int = 30,
                             min_ssim_diff: float = MIN_SSIM_DIFF,
                             use_segmentation: bool = True):       # ← signature updated
    """
    Split image into a grid of small pieces for detailed analysis.
    Returns a list of tuples (region1_b64, region2_b64, description).
    • Tiles עם יותר מ‑20 % פיקסלים שחורים (0,0,0) נדחים אוטומטית כדי למנוע רעש מאזורי warp.

    Args:
        img_ref: Reference image (PIL Image)
        img_aligned: Aligned image (PIL Image)
        grid_size: Tuple (cols, rows) defining the grid dimensions
        top_k: Number of top regions to return by SSIM-difference
    """
    import base64, io        # <-- moved here (was inside a nested block)
    def img_to_b64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()

    show_pre  = bool(st.session_state.get("show_pre_tiles",  False))
    show_post = bool(st.session_state.get("show_post_tiles", False))
    show_before = bool(st.session_state.get("show_before_tiles", False))

    # --- counters for debug statistics ---
    tile_total = 0          # number of tiles before any filtering
    tile_after_ssim = 0     # tiles that survived SSIM filter
    # tile_after_yolo will be computed later

    # --- NEW: force-sync with the UI value ---
    min_ssim_diff = float(st.session_state.get("MIN_SSIM_DIFF", min_ssim_diff))
    # -----------------------------------------

    diff_values = []          # NEW – collect diff-SSIM values for debug
    candidates_raw = []       # container for tiles that pass initial checks

    # ---------- geometry helpers (NEEDED for the loop) ----------
    width, height = img_ref.size
    cols, rows = grid_size
    tile_width  = width  // cols
    tile_height = height // rows
    # ------------------------------------------------------------

    # ---------- STEP-1 -------------------------------------------------
    for y in range(rows):
        for x in range(cols):
            tile_total += 1
            left = x * tile_width
            upper = y * tile_height
            right = min((x + 1) * tile_width, width)
            lower = min((y + 1) * tile_height, height)
            ref_tile = img_ref.crop((left, upper, right, lower))
            aligned_tile = img_aligned.crop((left, upper, right, lower))
            ref_arr  = np.array(ref_tile)
            aligned_arr = np.array(aligned_tile)
            if min(ref_arr.shape) == 0 or min(aligned_arr.shape) == 0:
                continue
            # --- ignore tiles that are mostly warp‑black (0,0,0) ---
            # A pixel is "black" if all RGB channels are below 5.
            black_mask   = np.all(aligned_arr < 5, axis=2)
            black_ratio  = black_mask.mean()
            # If more than 20 % of the tile is black, skip – it is an artefact of perspective warp
            if black_ratio > 0.20:
                continue
            # --- use SSIM difference (1 - similarity) ---
            import cv2
            ref_gray     = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2GRAY)
            aligned_gray = cv2.cvtColor(aligned_arr, cv2.COLOR_RGB2GRAY)
            diff_ssim = 1.0 - ssim(ref_gray, aligned_gray)
            diff_values.append(diff_ssim)      # NEW
            position_desc = f"חלק {x+1},{y+1} - שורה {y+1}, עמודה {x+1}"
            if show_before:
                dbg = Image.new("RGB", (ref_tile.width + aligned_tile.width, ref_tile.height))
                dbg.paste(ref_tile, (0, 0))
                dbg.paste(aligned_tile, (ref_tile.width, 0))
                st.image(dbg, caption=f"Before SSIM – {position_desc}", use_container_width=True)
            # דלג על אריחים עם שינוי מבני זעיר
            if diff_ssim < min_ssim_diff:
                continue
            # ---- stable‑change filter (must persist in neighbour pair) ----
            if "diff_cube" in st.session_state and "current_pair_idx" in st.session_state:
                pair_idx0 = st.session_state.current_pair_idx - 1  # cube is 0‑based
                win = int(st.session_state.get("stable_window", 1))
                if not _is_stable_change(pair_idx0, y, x,
                                         st.session_state.diff_cube,
                                         th=min_ssim_diff,
                                         win=win):
                    continue  # transient – skip
            candidates_raw.append((
                img_to_b64(ref_tile),          # 0
                img_to_b64(aligned_tile),      # 1
                position_desc,                 # 2
                diff_ssim,                     # 3
                (left, upper, right, lower)      # <-- NEW: absolute coords in ref_img
            ))
            tile_after_ssim += 1

    # Normalise → all tuples now length-4
    candidates = candidates_raw

    # ---------- DEBUG : view tiles kept after SSIM ----------
    if show_pre and candidates:                         # NEW
        st.subheader("🔍 Tiles after SSIM filter")      # NEW
        for b64_r, b64_a, desc, _, box in candidates[:top_k]:   # NEW
            ref_img = Image.open(io.BytesIO(base64.b64decode(b64_r)))  # NEW
            aln_img = Image.open(io.BytesIO(base64.b64decode(b64_a)))  # NEW
            comp = Image.new("RGB", (ref_img.width + aln_img.width, ref_img.height))  # NEW
            comp.paste(ref_img, (0, 0)); comp.paste(aln_img, (ref_img.width, 0))      # NEW
            st.image(comp, caption=desc, use_container_width=True)                    # NEW

    # ---------- STEP‑1b : Diff‑mask pixel change filter ----------
    diff_thr = float(st.session_state.get("diff_mask_thr", 3.0)) / 100.0  # convert %→fraction
    diff_filtered = []
    tile_after_diff = 0
    for b64_r, b64_a, desc, diff_val, box in candidates:
        ref_tile = Image.open(io.BytesIO(base64.b64decode(b64_r)))
        aln_tile = Image.open(io.BytesIO(base64.b64decode(b64_a)))
        # simple abs‑diff on grayscale
        g1 = cv2.cvtColor(np.array(ref_tile), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(np.array(aln_tile), cv2.COLOR_RGB2GRAY)
        absdiff = cv2.absdiff(g1, g2)
        _, mask = cv2.threshold(absdiff, 25, 255, cv2.THRESH_BINARY)
        changed_ratio = mask.mean() / 255.0
        if changed_ratio >= diff_thr:
            diff_filtered.append((b64_r, b64_a, desc, diff_val, box))
            tile_after_diff += 1
    candidates = diff_filtered

    # Debug view of diff‑mask tiles: view tiles kept after SSIM ----------
    if st.session_state.get("show_diff_tiles", False) and candidates:
        st.subheader("⚡️ Tiles after Diff‑mask filter")
        for b64_r, b64_a, desc, _, box in candidates[:top_k]:
            ref_img = Image.open(io.BytesIO(base64.b64decode(b64_r)))   # FIX
            aln_img = Image.open(io.BytesIO(base64.b64decode(b64_a)))   # FIX
            comp  = Image.new("RGB", (ref_img.width + aln_img.width, ref_img.height))
            comp.paste(ref_img, (0, 0)); comp.paste(aln_img, (ref_img.width, 0))
            st.image(comp, caption=desc, use_container_width=True)

    # ---------- STEP-2: optional segmentation filter (Mask-R CNN) ----------
    if use_segmentation:
        score_thr = float(st.session_state.get("seg_score_thr", 0.50))
        iou_thr   = float(st.session_state.get("seg_iou_thr", 0.30))
        filtered  = []
        for b64_r, b64_a, desc, diff, box in candidates:
            ref_tile  = Image.open(io.BytesIO(base64.b64decode(b64_r)))
            aln_tile  = Image.open(io.BytesIO(base64.b64decode(b64_a)))
            boxes = _maskrcnn_new_objects(ref_tile, aln_tile,
                                          score_thr=score_thr,
                                          iou_thr=iou_thr)
            if not boxes:
                continue
            # draw boxes for debugging
            ref_draw, aln_draw = ref_tile.copy(), aln_tile.copy()
            d1, d2 = ImageDraw.Draw(ref_draw), ImageDraw.Draw(aln_draw)
            for x1, y1, x2, y2 in boxes:
                d1.rectangle([x1, y1, x2, y2], outline="red", width=3)
                d2.rectangle([x1, y1, x2, y2], outline="red", width=3)
            filtered.append((img_to_b64(ref_draw),
                             img_to_b64(aln_draw),
                             desc,
                             diff,
                             box))
        candidates = filtered

        # ---------- DEBUG : view tiles that passed the YOLO test ----------
        if show_post and candidates:
            st.subheader("🎯 Tiles after Segmentation filter")
            for b64_r, b64_a, desc, _, box in candidates[:top_k]:
                ref_img = Image.open(io.BytesIO(base64.b64decode(b64_r)))  # FIX
                aln_img = Image.open(io.BytesIO(base64.b64decode(b64_a)))  # FIX
                comp  = Image.new("RGB", (ref_img.width + aln_img.width, ref_img.height))
                comp.paste(ref_img, (0, 0)); comp.paste(aln_img, (ref_img.width, 0))
                st.image(comp, caption=desc, use_container_width=True)

    # ---------- STEP-3: sort & return ----------
    tile_after_yolo = len(candidates)

    # --- draw a red border on every remaining tile (so GPT sees it) ---
    def _with_border(b64_img: str) -> str:
        """Decode → draw 3-px red rectangle → re-encode."""
        from PIL import Image, ImageDraw
        import io, base64
        img = Image.open(io.BytesIO(base64.b64decode(b64_img))).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width - 1, img.height - 1], outline="red", width=1)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    # apply the red border to every surviving candidate _before_ sorting/returning
    bordered = []
    for r_b64, a_b64, desc, diff_v, box in candidates:
        bordered.append((_with_border(r_b64),
                         _with_border(a_b64),
                         desc,
                         diff_v,
                         box))
    candidates = bordered

    # ---------- DEBUG : show counts ----------
    if st.session_state.get("show_tile_stats", False):
        # ...existing tile-count message...
        st.info(
            f"📊 Tile counts – before SSIM: {tile_total} | "
            f"after SSIM: {tile_after_ssim} | "
            f"after Diff-mask: {tile_after_diff} | "
            f"after YOLO: {tile_after_yolo}"
        )
        # NEW – SSIM statistics & quick histogram
        if diff_values:
            dv = np.array(diff_values)
            st.info(
                f"🔍 SSIM-diff stats – min={dv.min():.3f}, "
                f"mean={dv.mean():.3f}, max={dv.max():.3f}, "
                f"threshold={min_ssim_diff}"
            )
            st.bar_chart(dv, use_container_width=True)
        # NEW – show parameter set (kept unchanged)
        st.info(
            "🔧 Parameters in use – "
            f"grid={grid_size[0]}×{grid_size[1]}, "
            f"MIN_SSIM_DIFF={min_ssim_diff:.3f}, "
            f"diff_mask_thr={st.session_state.get('diff_mask_thr')} %, "
            f"seg_score_thr={st.session_state.get('seg_score_thr')}, "
            f"seg_iou_thr={st.session_state.get('seg_iou_thr')}, "
            f"stable_window={st.session_state.get('stable_window')}, "
            f"top_k={top_k}, "
            f"sharp_th={st.session_state.get('sharp_th')}"
        )
    candidates.sort(key=lambda t: t[3], reverse=True)
    return [(r, a, d, box) for r, a, d, _, box in candidates[:top_k]]

# ---------- NEW UTILITY ----------
def _compose_b64_side_by_side(b64_left:str, b64_right:str) -> str:
    """
    Build a small side-by-side composite from two base64-encoded JPEGs
    and return the resulting image encoded back to base64.
    """
    from PIL import Image
    import base64, io
    img_l = Image.open(io.BytesIO(base64.b64decode(b64_left)))
    img_r = Image.open(io.BytesIO(base64.b64decode(b64_right)))
    comp   = Image.new("RGB", (img_l.width + img_r.width, max(img_l.height, img_r.height)))
    comp.paste(img_l, (0, 0)); comp.paste(img_r, (img_l.width, 0))
    buf = io.BytesIO()
    comp.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()
# ----------------------------------

# ...existing imports...
import time      # NEW
# ...existing code...
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
# ---------------------------------------------------------

# ---------- NEW UTILITY ----------
def _timed_gpt_call(payload: dict, label: str = ""):
    """
    Wrapper around `call_gpt_with_retry` that measures the call
    duration and logs / prints it.
    """
    start = time.time()
    logging.info(f"↗️  GPT call started  {label}")
    resp = call_gpt_with_retry(payload)
    elapsed = time.time() - start
    logging.info(f"✅ GPT call finished {label}  – {elapsed:.1f}s")
    st.write(f"⏱️ {label} took {elapsed:.1f}s")  # visible in Streamlit
    return resp, elapsed

def _is_frame_sharp(bgr_frame, thresh: float = SHARPNESS_THRESHOLD) -> bool:
    """
    Simple blur detector: returns True when the variance of the Laplacian
    of the gray image exceeds `thresh`.
    """
    import cv2
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= thresh
# -------------------------------------------

# ---------- MASK-R CNN (lazy-loaded) ----------
_MASKRCNN_MODEL = None
def _get_maskrcnn_model():
    """
    Load a pre-trained Mask-R CNN (ResNet-50 + FPN) once and reuse.
    """
    global _MASKRCNN_MODEL
    if _MASKRCNN_MODEL is None:
        _MASKRCNN_MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        _MASKRCNN_MODEL.eval()
    return _MASKRCNN_MODEL

def _maskrcnn_new_objects(img_before, img_after,
                          score_thr: float = 0.50,
                          iou_thr: float = 0.30):
    """
    Bounding-boxes of objects that appear in `img_after` but not in
    `img_before` (Mask-R CNN detections).
    """
    import torchvision.ops as ops
    from torchvision.transforms.functional import to_tensor
    model = _get_maskrcnn_model()
    with torch.no_grad():
        pred_b = model([to_tensor(img_before)])[0]
        pred_a = model([to_tensor(img_after)])[0]

    # keep high-score detections
    keep_b = pred_b["scores"] >= score_thr
    keep_a = pred_a["scores"] >= score_thr
    boxes_b, labels_b = pred_b["boxes"][keep_b], pred_b["labels"][keep_b]
    boxes_a, labels_a = pred_a["boxes"][keep_a], pred_a["labels"][keep_a]

    new_boxes = []
    for box_a, lbl_a in zip(boxes_a, labels_a):
        same_cls = (labels_b == lbl_a).nonzero(as_tuple=False).squeeze(1)
        if len(same_cls) == 0:
            new_boxes.append(box_a.int().tolist())
            continue
        if ops.box_iou(box_a.unsqueeze(0), boxes_b[same_cls]).max().item() < iou_thr:
            new_boxes.append(box_a.int().tolist())
    return new_boxes
# ---------------------------------------------

# ---------- SIDE-BY-SIDE COMPOSITE ----------
def _compose_pair(ref_img: Image.Image,
                  aligned_img: Image.Image,
                  draw_seg: bool = False):      # RENAMED
    comp = Image.new("RGB", (ref_img.width + aligned_img.width, ref_img.height))
    comp.paste(ref_img, (0, 0))
    comp.paste(aligned_img, (ref_img.width, 0))
    if draw_seg:
        try:
            bboxes = _maskrcnn_new_objects(
                ref_img,
                aligned_img,
                score_thr=float(st.session_state.get("seg_score_thr", 0.50)),
                iou_thr=float(st.session_state.get("seg_iou_thr", 0.30))
            )
            if bboxes:
                draw = ImageDraw.Draw(comp)
                x_off = ref_img.width
                for x1, y1, x2, y2 in bboxes:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    draw.rectangle([x1 + x_off, y1, x2 + x_off, y2],
                                   outline="red", width=4)
        except Exception as e:
            logging.warning(f"Segmentation overlay failed: {e}")
    return comp
# --------------------------------------------------------------------

# ...existing code...

# ---------- unified prompt helpers ----------
DEFAULT_PROMPT = (
    "אתה סוכן בינה מלאכותית להשוואת תמונות, עליך להשוות בין שתי התמונות ולהתייחס לקטע אשר מוקף באדם"
    "המטרה שלך לתאר בעברית האם יש שינוי בין שתי התמונות בריבוע אשר מסומן באדום, במידה ויש שינוי תאר אותו"
)

def get_user_prompt() -> str:
    """
    Return the prompt stored in session (or the default).
    Call `prompt_text_area()` once per page to render the UI.
    """
    return st.session_state.get("user_prompt", DEFAULT_PROMPT)

def prompt_text_area():
    """Render the prompt editor once per page."""
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = DEFAULT_PROMPT
    st.text_area("GPT prompt (applies everywhere):",
                 key="user_prompt",
                 height=100)
# -----------------------------------------------------------

# ---------- REPORT HELPERS ---------------------------------
def _compose_pair_b64(img_l: Image.Image, img_r: Image.Image) -> str:
    """Return side-by-side composite (JPEG→b64) with NO red overlay."""
    import base64, io
    comp = Image.new("RGB", (img_l.width + img_r.width, img_l.height))
    comp.paste(img_l, (0, 0)); comp.paste(img_r, (img_l.width, 0))
    buf = io.BytesIO()
    comp.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def _compose_pair_b64_with_box(img_l: Image.Image,
                               img_r: Image.Image,
                               box: tuple) -> str:
    """
    Same as _compose_pair_b64 but also paints a red rectangle (3 px)
    on both halves at `box` (coords refer to the *left* image).
    """
    import base64, io
    comp = Image.new("RGB", (img_l.width + img_r.width, img_l.height))
    comp.paste(img_l, (0, 0)); comp.paste(img_r, (img_l.width, 0))
    if box:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comp)
        draw.rectangle(box, outline="red", width=3)            # left half
        x1, y1, x2, y2 = box
        draw.rectangle([x1 + img_l.width, y1, x2 + img_l.width, y2],
                       outline="red", width=3)                 # right half
    buf = io.BytesIO()
    comp.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def _add_report_entry(pair_idx: int,
                      pair_b64: str,
                      tile_b64: str,
                      gpt_text: str,
                      box: tuple = None):      # <-- NEW param
    """Append a single tile-level finding to the session report list."""
    if "report_data" not in st.session_state:
        st.session_state.report_data = []
    st.session_state.report_data.append({
        "pair_idx": pair_idx,
        "pair_b64": pair_b64,
        "tile_b64": tile_b64,
        "text": gpt_text,
        "box": box                               # keep for reuse
    })
# -----------------------------------------------------------

# -----------------------------------------------------------
# 🆕  TEMP-FIX : run analysis for the selected pairs
def _run_pairs_analysis(selected_ids, custom_prompt: str) -> None:
    """
    Quick replacement for the previously inlined analysis block.
    Generates a single GPT-overview answer for each selected pair and
    updates both the on-screen placeholder and the report tab.
    """
    if not selected_ids:
        return

    progress = st.progress(0.0)
    status   = st.empty()
    total    = len(selected_ids)
    done     = 0

    # helper to locate pair object once
    def _get_pair(idx: int):
        return next((p for p in st.session_state.ground_pairs if p["idx"] == idx), None)

    for pair_idx in selected_ids:
        pair = _get_pair(pair_idx)
        if pair is None:
            continue

        status.text(f"מנתח זוג {pair_idx} ...")
        # --- analyse only the tiles that survived all filters ---
        tiles = _extract_focused_regions(
            pair["ref"],
            pair["aligned"],
            grid_size=(int(st.session_state.get("grid_size", 3)),
                       int(st.session_state.get("grid_size", 3))),
            top_k=int(st.session_state.get("top_k", 30)),
            min_ssim_diff=float(st.session_state.get("MIN_SSIM_DIFF", MIN_SSIM_DIFF)),
            use_segmentation=bool(st.session_state.get("use_segmentation", True))
        )

        pair_changes_html = []   # accumulate change texts for this pair

        for t_idx, (b64_r, b64_a, position_desc, box) in enumerate(tiles, start=1):
            # Build a GPT prompt for this single tile
            tile_prompt = [
                {"role": "system", "content": [
                    {"type": "text",
                     "text": "אתה עוזר AI שתפקידו לעזור לאנשים למצוא מידע."}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": TILE_COMPARE_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64_r}"}},
                    {"type": "text", "text": "---"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}}
                ]}
            ]
            # Call GPT for this tile
            resp, _ = _timed_gpt_call({
                "model": DEPLOYMENT,
                "messages": tile_prompt,
                "max_tokens": 1024,
                "temperature": 0,
            }, label=f"pair {pair_idx} tile {t_idx}")

            import json
            raw_txt = resp.choices[0].message.content if resp and hasattr(resp, "choices") else ""
            stripped = raw_txt.strip()
            if stripped.startswith("```"):
                stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
                stripped = stripped.rstrip("`").strip()

            try:
                data = json.loads(stripped)
            except Exception:
                data = None

            # Skip tile if no material change

            # Skip tile if no material change – אבל צריך קודם לבנות תיאור
            
            if data:
                description = data.get("description", "").strip()
                confidence  = data.get("confidence", 0)
            else:
                description = ""
                confidence  = 0
                # --- Debug output: tile + GPT text -----------------
            txt_full = f"**{position_desc}** – {description} (בטחון {confidence}%)"
            if st.session_state.get("show_tile_debug", False):
                tile_comp_b64 = _compose_b64_side_by_side(b64_r, b64_a)
                st.image(
                    f"data:image/jpeg;base64,{tile_comp_b64}",
                    caption=txt_full,
                    use_container_width=True
                )
            if not data or not data.get("change_detected"):
                continue
    # ---------------------------------------------------
            
            pair_changes_html.append(txt_full)

            # Visuals for report
            comp_b64  = _compose_pair_b64_with_box(pair["ref"], pair["aligned"], box)
            tile_b64  = _compose_b64_side_by_side(b64_r, b64_a)
            _add_report_entry(pair_idx, comp_b64, tile_b64, txt_full, box)

        # Render pair‑level results in the analysis tab
        if pair_changes_html:
            md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                  f'### ממצאים לזוג {pair_idx}<br><br>' +
                  "<br><br>".join(pair_changes_html) + "</div>")
        else:
            md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                  f'### ממצאים לזוג {pair_idx}<br><br>לא נמצאו שינויים.</div>')
        st.session_state[f"gpt_txt_{pair_idx}"] = md
        st.markdown(md, unsafe_allow_html=True)

        # progress
        done += 1
        progress.progress(done / total)

    status.text("הניתוח הסתיים!")
    progress.empty()
# -----------------------------------------------------------
#
# ...existing code...