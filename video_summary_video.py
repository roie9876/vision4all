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
import numpy as np  # <-- NEW
from skimage.metrics import structural_similarity as ssim  # NEW

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

# ---------- high-quality image parameters ----------
MAX_DIM_FOR_GPT = 1024        # longest side sent to GPT (was 640/800)
JPEG_QUALITY    = 95          # better quality for base64 encoding
# ---------------------------------------------------
# Minimum structural‑difference (1‑SSIM) לסינון רעש
MIN_SSIM_DIFF = 0.2   # 8 %

COMMON_HEBREW_PROMPT = (
    "נתח את התמונה וספק תיאור בעברית. התמקד רק בשינויים משמעותיים כמו הופעה של אובייקט חדש "
    "או החסרה של אובייקט, אל תציין שינויים זניחים כמו כתמים לא ברורים."
)

def resize_and_compress_image(image, max_dim: int = MAX_DIM_FOR_GPT):
    """
    Return the same PIL image if it already fits within `max_dim`,
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
        buffered = io.BytesIO()     # <- FIXED
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

    content_prompt = st.text_input(
        "Enter the content prompt:",
        value=COMMON_HEBREW_PROMPT
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

def _resize_for_prompt(img: Image.Image, max_dim: int = MAX_DIM_FOR_GPT) -> Image.Image:
    """
    Returns a resized copy of PIL `img` such that the longest side ≤ max_dim.
    Keeps aspect ratio.  Helps keep base64 payload smaller for GPT calls.
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
                # Ensure RGB and resize to reduce payload
                img_small = _resize_for_prompt(img, max_dim=640)
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
    Return a list of PIL images sampled from the video at fps_target.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 1
    interval = int(round(fps_src / fps_target)) if fps_target else 1
    frames, idx = [], 0
    success, frame = cap.read()
    while success:
        if idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        success, frame = cap.read()
        idx += 1
    cap.release()
    return frames


# --- Alignment helper ---
def _align_images(img_ref, img_to_align, max_features: int = 1000, good_match: int = 50):
    """
    Align `img_to_align` (PIL.Image) to `img_ref` (PIL.Image) using ORB feature matching
    and homography (RANSAC).  
    Returns a tuple: (aligned PIL.Image, number_of_inliers).  
    If alignment fails, the original `img_to_align` is returned with 0 inliers.
    """
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
        if m.distance < 0.7 * n.distance:  # Stricter threshold (was 0.75)
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

    if xs_ref.size == 0 or ys_ref.size == 0 or xs_align.size == 0 or ys_align.size == 0:
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
            combo.paste(aligned_f2.resize((320, 320)), (320, 0))
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
        value=COMMON_HEBREW_PROMPT
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
                resp = call_azure_openai_with_retry({
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
def _is_stable_change(idx, r, c, cube, th=MIN_SSIM_DIFF):
    """
    Return True if tile (r,c) at pair `idx` is above threshold AND
    remains above threshold in at least one neighbouring pair
    (idx‑1 or idx+1).  Filters out changes that appear in only
    a single pair.
    """
    if cube[idx, r, c] < th:
        return False
    if idx > 0 and cube[idx - 1, r, c] >= th:
        return True
    if idx < cube.shape[0] - 1 and cube[idx + 1, r, c] >= th:
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
        comp.paste(ref_crop, (0, 0)); comp.paste(aligned_crop, (ref_crop.width, 0))

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

    fps_target = st.selectbox("קצב דגימת פריימים", [0.5, 1, 2], index=1)
    custom_prompt = st.text_input(
        "הנחיית תוכן (עברית)",
              value=" תאר בעברית את השינויים מהותיים בקרקע בין שתי התמונות, כולל הופעה של אויבקט חדש או החסרה של אובייקט, תזוזות אדמה,   מבנים, אבנים,צמחיה, סימני צמיגים, גדרות. כל שינוי מהותי שאתה מזהה תאר אותה לפרטים עם ציון אחוז ודאות עד כמה אתה בטוח בשינוי")

    # Add UI for the new parameters
    with st.expander("פרמטרים מתקדמים"):
        st.session_state["MIN_SSIM_DIFF"] = st.number_input(
            "MIN_SSIM_DIFF",
            0.0,
            1.0,
            0.2
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

    if st.button("הכן זוגות") and before_file and after_file:
        # clean old state
        for k in list(st.session_state.keys()):
            if k.startswith("chk_pair_"):
                st.session_state.pop(k)
        # save videos to temp dir
        tmp = tempfile.mkdtemp(prefix="ground_change_")
        path_before = os.path.join(tmp, before_file.name)
        path_after  = os.path.join(tmp,  after_file.name)
        with open(path_before, "wb") as f: f.write(before_file.getbuffer())
        with open(path_after,  "wb") as f: f.write(after_file.getbuffer())
        # store paths so we can replay video segments later
        st.session_state.path_before = path_before
        st.session_state.path_after  = path_after

        # heavy compute – run once and cache in session_state
        with st.spinner("מחלץ ומיישר פריימים ..."):
            st.session_state.ground_pairs = _build_aligned_pairs(path_before, path_after, fps_target)
        st.success("הזוגות מוכנים! סמן/י ונתח.")

        # keep temp dir so crops stay valid during session
        st.session_state.temp_dir_gc = tmp

    # ---------- show pairs if we already have them ----------
    if "ground_pairs" in st.session_state:
        selected_ids = []
        for pair in st.session_state.ground_pairs:
            idx = pair["idx"]

            # create a dedicated container so answer is shown right below the pair
            pair_container = st.container()
            with pair_container:
                st.image(pair["comp"],
                         caption=f"זוג {idx} – inliers={pair['inliers']}",
                         use_container_width=True)

                # checkbox for selection
                if st.checkbox(f"נתח זוג {idx}", key=f"chk_pair_{idx}"):
                    selected_ids.append(idx)

                # --- GPT output placeholder handling ---
                txt_key = f"gpt_txt_{idx}"        # stores rendered markdown string
                plh_key = f"gpt_plh_{idx}"        # stores the placeholder object

                # Always create a fresh placeholder *inside this container*
                if plh_key not in st.session_state:
                    st.session_state[plh_key] = st.empty()
                placeholder = st.session_state[plh_key]

                # If we already have generated text, show it right away
                if txt_key in st.session_state:
                    placeholder.markdown(
                        st.session_state[txt_key],
                        unsafe_allow_html=True
                    )

        if st.button("Analyze selected pairs") and selected_ids:
            st.info("מריץ ניתוח מעמיק לפי חלוקה לאזורים...")

            # Create a progress bar to show overall progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("מתחיל ניתוח...")
            
            # Track the total number of analyses and completed ones
            total_analyses = len(selected_ids)
            completed_analyses = 0

            import concurrent.futures
            def _process_pair(pair, custom_prompt):
                idx = pair["idx"]
                status_text.text(f"מנתח זוג {idx}...")
                
                # Use session‑selected parameters so the user’s grid/top‑k/threshold are honoured
                regions = _extract_focused_regions(
                    pair["ref"],
                    pair["aligned"],
                    grid_size=(int(st.session_state.get("grid_size", 3)),
                               int(st.session_state.get("grid_size", 3))),
                    top_k=int(st.session_state.get("top_k", 30)),
                    min_ssim_diff=float(st.session_state.get("MIN_SSIM_DIFF", MIN_SSIM_DIFF))
                )
                
                # First analyze the whole image pair
                full_prompt = [
                    {"role": "system", "content": [
                        {"type": "text", "text": " אתה עוזר בינה מלאכותית שמנתח שינויים באזור מוגדר בין שתי תמונות. דווח **רק** על שינויים מהותיים (הופעת/היעלמות אובייקטים, תזוזות גדולות, שינויי מבנה או קרקע) והתעלם משינויים זניחים כגון שינויים קלים בגוונים, תאורה, רעש מצלמה או תנועות עלים קטנות אוכתמים לא ברורים"}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{custom_prompt}\nתן ניתוח כללי של ההבדלים העיקריים בין התמונות. התעלם משינויים זניחים ואל תדווח עליהם."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pair['b64_1']}"}},
                        {"type": "text", "text": "---"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pair['b64_2']}"}},
                    ]}
                ]
                
                gpt_resp, _ = _timed_gpt_call({   # ← uses the wrapper
                    "model": DEPLOYMENT,
                    "messages": full_prompt,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "stream": False
                }, label=f"pair-overview #{idx}")
                main_analysis = gpt_resp.choices[0].message.content if gpt_resp and hasattr(gpt_resp, 'choices') else "שגיאה בקבלת תוצאה."
                
                # Now analyze each detailed region with significant differences
                region_analyses = []
                def _one_region(args):
                    i, (r1_b64, r2_b64, region_desc) = args
                    region_prompt = [
                        {"role": "system", "content": [
                            {"type": "text", "text": "אתה עוזר בינה מלאכותית שמנתח שינויים באזור מוגדר בין שתי תמונות. דווח **רק** על שינויים מהותיים (הופעת/היעלמות אובייקטים, תזוזות גדולות, שינויי מבנה או קרקע) והתעלם משינויים זניחים כגון שינויים קלים בגוונים, תאורה, רעש מצלמה או תנועות עלים קטנות. או כתמים לא ברורים"}
                        ]},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"אנא תאר אך ורק שינויים מהותיים באזור זה ({region_desc}) – \
כגון הופעת או היעלמות אובייקטים בולטים, תזוזות משמעותיות, \
או שינויי קרקע/מבנה ניכרים. התעלם משינויים מינוריים בגוונים, \
תאורה או תנועות עלים זעירות. עבור כל שינוי מהותי ציין אחוז ודאות \
(לדוגמה: ודאות: 90%). אם אין שינוי מהותי, כתוב במפורש: 'לא זוהו שינויים מהותיים'."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r1_b64}"}},
                            {"type": "text", "text": "---"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r2_b64}"}}
                        ]}
                    ]
                    resp, _ = _timed_gpt_call({
                        "model": DEPLOYMENT,
                        "messages": region_prompt,
                        "max_tokens": 4096,
                        "temperature": 0,
                        "top_p": 0.95,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "stop": None,
                        "stream": False
                    }, label=f"region {i} (pair {idx})")
                    if resp and hasattr(resp, "choices"):
                        txt = resp.choices[0].message.content
                        if ("אין שינויים" not in txt) and ("לא נמצאו שינויים" not in txt):
                            comp_b64 = _compose_b64_side_by_side(r1_b64, r2_b64)
                            img_tag  = f'<img src="data:image/jpeg;base64,{comp_b64}" style="max-width:100%;height:auto;border:1px solid #ddd;margin-bottom:6px"/>'
                            return f"{img_tag}<br><b>{region_desc}</b><br>{txt}"
                    return None

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    for ret in pool.map(_one_region, enumerate(regions, 1)):
                        if ret:
                            region_analyses.append(ret)
                
                # Combine the analyses
                if region_analyses:
                    combined_analysis = (
                        "<b>## ניתוח כללי</b><br>"
                        f"{main_analysis}<br><br>"
                        "<b>## ניתוח מפורט לפי אזורים</b><br>" +
                        "<br><br>".join(region_analyses)
                    )
                else:
                    combined_analysis = main_analysis
                
                return idx, combined_analysis

            # Process each pair one at a time to avoid resource contention
            for idx in selected_ids:
                # Find the pair with the matching idx
                pair = next((p for p in st.session_state.ground_pairs if p["idx"] == idx), None)
                if pair:
                    try:
                        result_idx, analysis_text = _process_pair(pair, custom_prompt)
                        
                        # Prepare markdown
                        md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                              f'### תוצאה לזוג {result_idx}<br><br>{analysis_text}</div>')
                        
                        # Save & render
                        st.session_state[f"gpt_txt_{result_idx}"] = md
                        st.session_state[f"gpt_plh_{result_idx}"].markdown(md, unsafe_allow_html=True)
                        
                        # Update progress
                        completed_analyses += 1
                        progress_bar.progress(completed_analyses / total_analyses)
                        status_text.text(f"הושלם ניתוח {completed_analyses} מתוך {total_analyses}")
                    except Exception as e:
                        st.error(f"שגיאה בניתוח זוג {idx}: {str(e)}")
                        logging.error(f"Error analyzing pair {idx}: {str(e)}")
                        
            # Final update
            progress_bar.progress(1.0)
            status_text.text(f"הניתוח הושלם! נותחו {completed_analyses} מתוך {total_analyses} זוגות.")

def _extract_focused_regions(img_ref, img_aligned,
                             grid_size=(3, 3), top_k: int = 30,
                             min_ssim_diff: float = MIN_SSIM_DIFF):
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
    width, height = img_ref.size
    cols, rows = grid_size
    tile_width = width // cols
    tile_height = height // rows

    def img_to_b64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()

    regions = []
    for y in range(rows):
        for x in range(cols):
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
            position_desc = f"חלק {x+1},{y+1} - שורה {y+1}, עמודה {x+1}"
            # דלג על אריחים עם שינוי מבני זעיר
            if diff_ssim < min_ssim_diff:
                continue
            # ---- stable‑change filter (must persist in neighbour pair) ----
            if "diff_cube" in st.session_state and "current_pair_idx" in st.session_state:
                pair_idx0 = st.session_state.current_pair_idx - 1  # cube is 0‑based
                if not _is_stable_change(pair_idx0, y, x,
                                         st.session_state.diff_cube,
                                         th=min_ssim_diff):
                    continue  # transient – skip
            regions.append((img_to_b64(ref_tile), img_to_b64(aligned_tile), position_desc, diff_ssim))

    # Sort by descending SSIM‑difference and return top_k regions
    sorted_regions = sorted(regions, key=lambda r: r[3], reverse=True)
    return [(r[0], r[1], r[2]) for r in sorted_regions[:top_k]]

# Helper to test temporal persistence
# def _is_persistent(idx, r, c, cube, ssim_th=MIN_SSIM_DIFF, win=1):
#     """
#     Return True if tile (r,c) at pair `idx` exceeds threshold AND the same tile
#     exceeds threshold in at least one neighbouring pair within ±win.
#     """
#     if cube[idx, r, c] < ssim_th:
#         return False
#     lo = max(0, idx - win)
#     hi = min(cube.shape[0] - 1, idx + win)
#     for j in range(lo, hi + 1):
#         if j != idx and cube[j, r, c] >= ssim_th:
#             return True
#     return False

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
    Wrapper around `call_azure_openai_with_retry` that measures the call
    duration and logs / prints it.
    """
    start = time.time()
    logging.info(f"↗️  GPT call started  {label}")
    resp = call_azure_openai_with_retry(payload)
    elapsed = time.time() - start
    logging.info(f"✅ GPT call finished {label}  – {elapsed:.1f}s")
    st.write(f"⏱️ {label} took {elapsed:.1f}s")  # visible in Streamlit
    return resp, elapsed
# ----------------------------------

# ...existing code...

def run_ground_change_detection():
    st.title("השוואת שני סרטונים – זיהוי שינויי קרקע")
    # Ensure change_events list exists each rerun
    if "change_events" not in st.session_state:
        st.session_state.change_events = []
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

    fps_target = st.selectbox("קצב דגימת פריימים", [0.5, 1, 2], index=1)
    custom_prompt = st.text_input(
        "הנחיית תוכן (עברית)",
              value=" תאר בעברית את השינויים מהותיים בקרקע בין שתי התמונות, כולל הופעה של אויבקט חדש או החסרה של אובייקט, תזוזות אדמה,   מבנים, אבנים,צמחיה, סימני צמיגים, גדרות. כל שינוי מהותי שאתה מזהה תאר אותה לפרטים עם ציון אחוז ודאות עד כמה אתה בטוח בשינוי")

    # Add UI for the new parameters
    with st.expander("Advanced parameters"):
        st.session_state["MIN_SSIM_DIFF"] = st.number_input(
            "MIN_SSIM_DIFF",
            0.0,
            1.0,
            0.2
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

    if st.button("הכן זוגות") and before_file and after_file:
        # clean old state
        for k in list(st.session_state.keys()):
            if k.startswith("chk_pair_"):
                st.session_state.pop(k)
        # save videos to temp dir
        tmp = tempfile.mkdtemp(prefix="ground_change_")
        path_before = os.path.join(tmp, before_file.name)
        path_after  = os.path.join(tmp,  after_file.name)
        with open(path_before, "wb") as f: f.write(before_file.getbuffer())
        with open(path_after,  "wb") as f: f.write(after_file.getbuffer())
        # Save paths so timeline can embed video players
        st.session_state.path_before = path_before
        st.session_state.path_after  = path_after

        # heavy compute – run once and cache in session_state
        with st.spinner("מחלץ ומיישר פריימים ..."):
            st.session_state.ground_pairs = _build_aligned_pairs(path_before, path_after, fps_target)
        st.success("הזוגות מוכנים! סמן/י ונתח.")

        # keep temp dir so crops stay valid during session
        st.session_state.temp_dir_gc = tmp

    # ---------- show pairs if we already have them ----------
    if "ground_pairs" in st.session_state:
        selected_ids = []
        for pair in st.session_state.ground_pairs:
            idx = pair["idx"]
            # create a dedicated container so answer is shown right below the pair
            pair_container = st.container()
            with pair_container:
                st.image(pair["comp"],
                         caption=f"זוג {idx} – inliers={pair['inliers']}",
                         use_container_width=True)

                # checkbox for selection
                if st.checkbox(f"נתח זוג {idx}", key=f"chk_pair_{idx}"):
                    selected_ids.append(idx)

                # --- GPT output placeholder handling ---
                txt_key = f"gpt_txt_{idx}"        # stores rendered markdown string
                plh_key = f"gpt_plh_{idx}"        # stores the placeholder object

                # Always create a fresh placeholder *inside this container*
                if plh_key not in st.session_state:
                    st.session_state[plh_key] = st.empty()
                placeholder = st.session_state[plh_key]

                # If we already have generated text, show it right away
                if txt_key in st.session_state:
                    placeholder.markdown(
                        st.session_state[txt_key],
                        unsafe_allow_html=True
                    )

        # Decide which pairs to analyze
        analyze_all_clicked   = st.button("נתח את כל הזוגות")
        analyze_selected_clicked = st.button("נתח זוגות נבחרים")

        if analyze_all_clicked:
            selected_ids = [p["idx"] for p in st.session_state.ground_pairs]

        # Run analysis if either button was pressed and we have pairs
        if (analyze_all_clicked or analyze_selected_clicked) and selected_ids:
            st.info("מריץ ניתוח מעמיק לפי חלוקה לאזורים...")

            # Create a progress bar to show overall progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("מתחיל ניתוח...")
            # ----- timeline placeholder (shows after progress bar) -----
            timeline_plh = st.empty()

            def _render_timeline():
                """Render/refresh the timeline of all pairs, right below the progress bar."""
                timeline_plh.empty()
                with timeline_plh.container():
                    st.subheader("Timeline – צפייה בכל הזוגות")
                    idxs_with_change = {ev["idx"] for ev in st.session_state.change_events}
                    fps_val = fps_target
                    for pair in st.session_state.ground_pairs:
                        idx = pair["idx"]
                        ts = (idx - 1) / fps_val if fps_val else 0
                        marker = "🟥" if idx in idxs_with_change else "⬜️"
                        label = f"{marker} זוג {idx} — {format_time(ts)}"
                        with st.expander(label, expanded=False):
                            if "path_before" in st.session_state and "path_after" in st.session_state:
                                st.video(st.session_state["path_before"], start_time=int(ts))
                                st.video(st.session_state["path_after"],  start_time=int(ts))
                            st.markdown(
                                st.session_state.get(f"gpt_txt_{idx}", "*ה‑GPT טרם ניתח זוג זה.*"),
                                unsafe_allow_html=True
                            )

            # initial empty render so the placeholder occupies space
            _render_timeline()

            # Track the total number of analyses and completed ones
            total_analyses = len(selected_ids)
            completed_analyses = 0

            import concurrent.futures
            def _process_pair(pair, custom_prompt):
                idx = pair["idx"]
                # Skip GPT work if we already have a cached result
                cache_key = f"gpt_txt_{idx}"
                if cache_key in st.session_state:
                    return idx, st.session_state[cache_key]
                status_text.text(f"מנתח זוג {idx}...")

                st.session_state.current_pair_idx = idx   # used by persistence filter
                # Use session‑selected parameters so the user’s grid/top‑k/threshold are honoured
                regions = _extract_focused_regions(
                    pair["ref"],
                    pair["aligned"],
                    grid_size=(int(st.session_state.get("grid_size", 3)),
                               int(st.session_state.get("grid_size", 3))),
                    top_k=int(st.session_state.get("top_k", 30)),
                    min_ssim_diff=float(st.session_state.get("MIN_SSIM_DIFF", MIN_SSIM_DIFF))
                )

                # First analyze the whole image pair
                full_prompt = [
                    {"role": "system", "content": [
                        {"type": "text", "text": "אתה עוזר בינה מלאכותית שמנתח שינויים באזור מוגדר בין שתי תמונות. דווח **רק** על שינויים מהותיים (הופעת/היעלמות אובייקטים, תזוזות גדולות, שינויי מבנה או קרקע) והתעלם משינויים זניחים כגון שינויים קלים בגוונים, תאורה, רעש מצלמה או תנועות עלים קטנות. או הופעה של כתמים לא ברורים."}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{custom_prompt}\nתן ניתוח כללי של ההבדלים העיקריים בין התמונות."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pair['b64_1']}"}},
                        {"type": "text", "text": "---"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pair['b64_2']}"}},
                    ]}
                ]

                gpt_resp, _ = _timed_gpt_call({   # ← uses the wrapper
                    "model": DEPLOYMENT,
                    "messages": full_prompt,
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "stream": False
                }, label=f"pair-overview #{idx}")
                main_analysis = gpt_resp.choices[0].message.content if gpt_resp and hasattr(gpt_resp, 'choices') else "שגיאה בקבלת תוצאה."

                # Now analyze each detailed region with significant differences
                region_analyses = []
                def _one_region(args):
                    i, (r1_b64, r2_b64, region_desc) = args
                    region_prompt = [
                        {"role": "system", "content": [
                            {"type": "text",
                             "text": "אתה עוזר בינה מלאכותית שמנתח שינויים בין שתי תמונות באזור ממוקד. דווח *רק* על שינויים מהותיים (הופעת/היעלמות אובייקטים, כתם בקרקע, תזוזה גדולה) והתעלם משינויים זניחים כמו גוונים, תאורה או תנועות עלים קטנות או הופעה של כתמים לא ברורים"}
                        ]},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"התמקד באזור זה ({region_desc}) והתייחס רק לשינויים הנראים בו. "
                                                      f"הבלט פרטים קטנים ותזוזות עדינות. עבור כל שינוי ציין אחוז ודאות (למשל: אחוז ודאות: 90%)."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r1_b64}"}},
                            {"type": "text", "text": "---"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r2_b64}"}}
                        ]}
                    ]
                    resp, _ = _timed_gpt_call({
                        "model": DEPLOYMENT,
                        "messages": region_prompt,
                        "max_tokens": 300,
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "stop": None,
                        "stream": False
                    }, label=f"region {i} (pair {idx})")
                    if resp and hasattr(resp, "choices"):
                        txt = resp.choices[0].message.content
                        if ("אין שינויים" not in txt) and ("לא נמצאו שינויים" not in txt):
                            comp_b64 = _compose_b64_side_by_side(r1_b64, r2_b64)
                            img_tag  = f'<img src="data:image/jpeg;base64,{comp_b64}" style="max-width:100%;height:auto;border:1px solid #ddd;margin-bottom:6px"/>'
                            text_only  = f"{region_desc}: {txt}"
                            html_chunk = f"{img_tag}<br><b>{region_desc}</b><br>{txt}"
                            return text_only, html_chunk
                        return None

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    for ret in pool.map(_one_region, enumerate(regions, 1)):
                        if ret:
                            region_analyses.append(ret)

                # --- split text / HTML and build high‑level summary ---
                text_summaries = [t[0] for t in region_analyses]
                html_chunks    = [t[1] for t in region_analyses]

                # Summarise the detailed findings, ignoring minor changes
                if text_summaries:
                    summariser_prompt = [
                        {"role": "system", "content": [
                            {"type": "text",
                             "text": "סכם עבור המשתמש רק את השינויים המהותיים שמתוארים, "
                                     "והתעלם מממצאים זניחים (גוונים, תאורה, תנועות צמחייה קטנות)."}
                        ]},
                        {"role": "user", "content": [
                            {"type": "text",
                             "text": "להלן ממצאים מפורטים:\n\n" + "\n\n".join(text_summaries) +
                                     "\n\nהצג סיכום קצר בעברית, תן ודאות כשיש, "
                                     "והתעלם משינויים שאינם מהותיים."}
                        ]}
                    ]
                    sum_resp, _ = _timed_gpt_call({
                        "model": DEPLOYMENT,
                        "messages": summariser_prompt,
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "stop": None,
                        "stream": False
                    }, label=f"pair-summary #{idx}")
                    overall_summary = sum_resp.choices[0].message.content \
                        if sum_resp and hasattr(sum_resp, "choices") \
                        else main_analysis
                else:
                    overall_summary = main_analysis

                # --- compose final HTML ---
                if html_chunks:
                    combined_analysis = (
                        "<b>## ניתוח כללי</b><br>" +
                        overall_summary + "<br><br>" +
                        "<b>## ניתוח מפורט לפי אזורים</b><br>" +
                        "<br><br>".join(html_chunks)
                    )
                else:
                    combined_analysis = ("<b>## ניתוח כללי</b><br>" + overall_summary)

                return idx, combined_analysis

            # Process each pair one at a time to avoid resource contention
            for idx in selected_ids:
                # Find the pair with the matching idx
                pair = next((p for p in st.session_state.ground_pairs if p["idx"] == idx), None)
                if pair:
                    try:
                        result_idx, analysis_text = _process_pair(pair, custom_prompt)

                        # Prepare markdown
                        md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                              f'### תוצאה לזוג {result_idx}<br><br>{analysis_text}</div>')

                        # Save & render
                        st.session_state[f"gpt_txt_{result_idx}"] = md
                        st.session_state[f"gpt_plh_{result_idx}"].markdown(md, unsafe_allow_html=True)

                        # Refresh the timeline so GPT text becomes visible even
                        # when no "change event" was recorded for this pair
                        _render_timeline()

                        # record this change (only if there were region‑level findings)
                        if "לא זוהו שינויים" not in analysis_text:
                            time_sec = (result_idx - 1) / fps_target if fps_target else 0
                            st.session_state.change_events.append({"idx": result_idx,
                                                                   "time": time_sec})
                            # _render_timeline()   # update live -- now handled above

                        # Update progress
                        completed_analyses += 1
                        progress_bar.progress(completed_analyses / total_analyses)
                        status_text.text(f"הושלם ניתוח {completed_analyses} מתוך {total_analyses}")
                    except Exception as e:
                        st.error(f"שגיאה בניתוח זוג {idx}: {str(e)}")
                        logging.error(f"Error analyzing pair {idx}: {str(e)}")

            # Final update
            progress_bar.progress(1.0)
            status_text.text(f"הניתוח הושלם! נותחו {completed_analyses} מתוך {total_analyses} זוגות.")