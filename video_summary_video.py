#working ver#
#working ver#
import streamlit as st
# Use the full browser width – avoids the "single narrow column" effect
st.set_page_config(page_title="Ground‑Change Detector", layout="wide")
# allow markdown containers to use the full width
st.markdown(
    """<style>
    .stMarkdown{max-width:100% !important;}
    /* Full-width images in expanders */
    .streamlit-expanderContent img {
        width: 100%;
        height: auto;
    }
    </style>""",
    unsafe_allow_html=True,
)
# --- helper to support both old `st.experimental_rerun` and new `st.rerun` ---
def _safe_rerun():
    """
    Call Streamlit's rerun function regardless of version:
    • st.rerun (new)
    • st.experimental_rerun (old)
    """
    func = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(func):
        func()
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
from PIL import Image, ImageDraw, ImageOps
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

# ---------- Deterministic behaviour ----------
import random as _rnd
from threading import Lock

torch.manual_seed(42)
np.random.seed(42)
_rnd.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
_PAIR_LOCK = Lock()
# --------------------------------------------
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
    # Always use the `gpt-4o` model on api.openai.com
    payload["model"] = "gpt-4o"
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

    For Azure “o*” reasoning models we must translate legacy
    parameters to the new preview API names.
    """
    """
    Dispatch to Azure OpenAI or api.openai.com according to
    st.session_state['api_provider'] (default 'Azure OpenAI').

    Adds a retry loop **also for Azure** so that transient
    400 Bad Request (and 5xx) responses are retried exactly the
    same way we already do for api.openai.com.

    For Azure “o*” reasoning models we must translate legacy
    parameters to the new preview API names.
    """
    import time, logging, copy
    import openai
    from openai import OpenAIError

    def _adapt_for_o3(p: dict):
        """
        Normalise payload `p` for Azure o‑series reasoning models.

        All o‑series deployments currently expect the **new** compact
        contract (preview 2025‑03‑01), even if the workspace is still on
        an older global API‑version.  Therefore we always:

        • Strip parameters that are not on the whitelist  
        • Rename `max_tokens` → `max_completion_tokens`  
        • Add `reasoning_effort` if the caller omitted it
        """
        import copy, streamlit as st

        # ---------- STEP‑1 : start with a clean copy ----------
        q = copy.deepcopy(p)

        # Parameters never accepted by preview endpoints
        q.pop("stop", None)
        if q.get("frequency_penalty", 0) == 0:
            q.pop("frequency_penalty", None)
        if q.get("presence_penalty", 0) == 0:
            q.pop("presence_penalty", None)
        if not q.get("stream"):
            q.pop("stream", None)

        # ---------- STEP‑2 : rename max_tokens ----------
        if "max_tokens" in q:
            q["max_completion_tokens"] = q.pop("max_tokens")
        q.setdefault("max_completion_tokens", 2048)

        # ---------- STEP‑3 : always supply reasoning_effort ----------
        q.setdefault(
            "reasoning_effort",
            st.session_state.get("reasoning_effort", "medium")
        )

        # ---------- STEP‑4 : keep only allowed keys ----------
        allowed = {
            "model", "messages", "temperature", "top_p",
            "reasoning_effort", "max_completion_tokens",
            "response_format"
        }
        q = {k: v for k, v in q.items() if k in allowed}

        return q

    provider = st.session_state.get("api_provider", "Azure OpenAI")
    is_o3 = provider == "Azure OpenAI" and re.match(r"o\d", str(payload.get("model", "")))

    if is_o3:
        payload = _adapt_for_o3(payload)

    # -------------------------------------------------
    # Uniform retry policy – 400 & 5xx are considered transient
    # -------------------------------------------------
    max_retries = 5
    backoff     = 2.0
    attempt     = 1
    while True:
        try:
            if provider == "OpenAI.com":
                # call_openai_com_with_retry already implements its own retry loop,
                # so a single call is enough here.
                return call_openai_com_with_retry(payload, max_retries=max_retries, backoff=backoff)
            else:  # Azure OpenAI
                return call_azure_openai_with_retry(payload)
        except Exception as e:
            # ---------------- robust retry logic ----------------
            msg = str(e)

            # Heuristic defaults
            status_code: int | None = None
            retriable = False

            # 1) Azure sometimes embeds "(BadRequest)" with no numeric code
            if "BadRequest" in msg or "Bad Request" in msg:
                status_code = 400
                retriable = True

            # 2) openai‑python 1.x TypeError on _notify_fail(...)
            if "_notify_fail() got an unexpected keyword argument" in msg:
                status_code = 400
                retriable = True

            # 3) Try native attribute
            if status_code is None:
                status_code = getattr(e, "status_code", None)

            # 4) Parse explicit HTTP code from message
            if status_code is None:
                m = re.search(r"\b(\d{3})\b", msg)
                if m:
                    status_code = int(m.group(1))

            # 5) Generic retry set
            if status_code is not None:
                retriable = retriable or status_code in (400, 429, 500, 502, 503, 504)

            if not retriable or attempt >= max_retries:
                logging.warning(
                    f"GPT call failed (attempt {attempt}/{max_retries}) – giving up. Error: {e}"
                )
                raise

            logging.warning(f"GPT call error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(backoff * attempt)
            attempt += 1
# ------------------------------------------

# Local imports
from utils import (
    summarize_descriptions,
    extract_frames,             # we still use for sub‑video frames
    call_azure_openai_with_retry  # Azure helper used by the dispatcher
)

# Add a wrapper function to redirect to the imported extract_frames
def _extract_frames(video_path, fps_target):
    """Wrapper around the imported extract_frames function"""
    return extract_frames(video_path, fps_target)

def _align_images(img_ref, img_to_align, max_features: int = 1000, good_match: int = 50):
    """
    Aligns img_to_align to img_ref using ORB feature detection.
    Returns: (aligned_image, inliers_count)
    """
    import cv2
    import numpy as np
    import base64  # NEW

    # ---------- robust conversion to cv2 image ----------  NEW
    def _as_cv2(obj):
        """
        Return a BGR `np.ndarray` for any reasonable image container:
        PIL.Image, np.ndarray, bytes, base64-str, or (nested) tuple/list.
        """
        # Debug the input type
        obj_type = type(obj).__name__
        obj_repr = str(obj)[:100] if isinstance(obj, str) else str(type(obj))
        
        # For debugging
        logging.info(f"Converting object type: {obj_type}")
        
        if isinstance(obj, (tuple, list)):
            for item in obj:
                try:
                    return _as_cv2(item)
                except Exception:
                    continue
            raise ValueError(f"Unsupported image container (type={obj_type})")
        
        if isinstance(obj, np.ndarray):
            # Handle string arrays explicitly
            if obj.dtype.kind in ['U', 'S']:  # Unicode or byte string arrays
                logging.warning(f"Got string array with dtype={obj.dtype}")
                if obj.size > 0:
                    # Try first element if it's a string array
                    try:
                        return _as_cv2(obj.item(0))
                    except Exception as e:
                        logging.warning(f"Failed to convert first item from array: {e}")
            
            if obj.ndim == 3 and obj.shape[2] == 3:  # Already RGB/BGR
                return obj
            elif obj.ndim == 2:  # Grayscale
                return cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
            elif obj.ndim == 3 and obj.shape[2] == 4:  # RGBA
                return cv2.cvtColor(obj, cv2.COLOR_RGBA2BGR)
            return obj  # Let CV2 handle other array formats
        
        # Check for PIL Image more robustly
        if hasattr(obj, 'convert') and hasattr(obj, 'size'):
            try:
                return cv2.cvtColor(np.array(obj.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Failed to convert PIL Image: {e} (type={obj_type})")
                
        if isinstance(obj, (bytes, bytearray)):
            try:
                arr = np.frombuffer(obj, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception as e:
                raise ValueError(f"Failed to decode bytes: {e} (type={obj_type})")
                
        if isinstance(obj, str):
            # First try as file path
            if os.path.exists(obj):
                try:
                    return cv2.imread(obj)
                except Exception as e:
                    logging.warning(f"Failed to read as file path: {e}")
            
            # Try as base64, fixing padding if needed
            try:
                # Ensure proper padding for base64
                padding = 4 - (len(obj) % 4) if len(obj) % 4 else 0
                padded_b64 = obj + ('=' * padding) 
                arr = np.frombuffer(base64.b64decode(padded_b64), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception as e:
                logging.warning(f"Failed to decode as base64: {e}")
        
        # FINAL FALLBACK - anything convertible to numpy array
        try:
            arr = np.array(obj)
            if arr.dtype.kind in ['U', 'S']:  # Catch numpy string arrays
                raise ValueError(f"Cannot convert string array with dtype={arr.dtype}")
                
            if arr.ndim == 3:
                if arr.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif arr.shape[2] == 3:  # RGB
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif arr.shape[2] == 1:  # Grayscale with channel
                    return cv2.cvtColor(arr.squeeze(2), cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 2:  # Grayscale
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logging.warning(f"Final fallback failed: {e}")
            
        # If all else fails, provide detailed error
        raise ValueError(f"Unsupported image type: {obj_type}, representation: {obj_repr}")

    # Use the new helper with comprehensive error handling
    try:
        img1 = _as_cv2(img_ref)
        img2 = _as_cv2(img_to_align)
    except Exception as e:
        # Last resort diagnosis and fallback
        logging.error(f"Advanced conversion failed: {e}, input types: {type(img_ref).__name__}, {type(img_to_align).__name__}")
        
        # Try to log more details about the object
        if isinstance(img_ref, np.ndarray):
            logging.info(f"img_ref is ndarray with shape={img_ref.shape}, dtype={img_ref.dtype}")
        
        # If input is a PIL image, extract the array directly 
        if hasattr(img_ref, 'convert'):
            img1 = cv2.cvtColor(np.array(img_ref.convert("RGB")), cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(np.array(img_to_align.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            # Give a more informative error
            raise ValueError(f"Cannot convert image of type {type(img_ref).__name__}") from e
    # ---------- end robust conversion ----------

    # Use ORB features with adjusted parameters from UI
    max_features = int(st.session_state.get("orb_max_features", 1000))
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Too few keypoints to match
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), 0
    
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test (Lowe)
    lowe_ratio = float(st.session_state.get("lowe_ratio", 0.70))
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)
    
    # Not enough good matches
    if len(good_matches) < good_match:
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), 0
    
    # Get points for homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography matrix
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    inliers = int(sum(mask))
    
    # Warp img2 to img1's perspective
    h, w = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, H, (w, h))
    
    # Convert back to PIL
    from PIL import Image
    pil_aligned = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    
    return pil_aligned, inliers

def _crop_to_overlap(img_ref, img_aligned, grid_size=(3, 3)):
    """
    Crop both images to the maximum common area.
    Ensures proper grid size (multiple of grid_size).
    Returns: (ref_cropped, aligned_cropped)
    """
    # Convert to PIL Image if we received strings or paths
    if isinstance(img_ref, str):
        if os.path.exists(img_ref):
            # It's a file path
            img_ref = Image.open(img_ref)
        else:
            # Try to decode as base64
            try:
                padding = 4 - (len(img_ref) % 4) if len(img_ref) % 4 else 0
                padded_b64 = img_ref + ('=' * padding)
                img_ref = Image.open(io.BytesIO(base64.b64decode(padded_b64)))
            except Exception as e:
                logging.error(f"Failed to convert str to Image: {e}")
                raise ValueError(f"Cannot process image string. Use a PIL Image object, path, or base64.")
    
    if isinstance(img_aligned, str):
        if os.path.exists(img_aligned):
            # It's a file path
            img_aligned = Image.open(img_aligned)
        else:
            # Try to decode as base64
            try:
                padding = 4 - (len(img_aligned) % 4) if len(img_aligned) % 4 else 0
                padded_b64 = img_aligned + ('=' * padding)
                img_aligned = Image.open(io.BytesIO(base64.b64decode(padded_b64)))
            except Exception as e:
                logging.error(f"Failed to convert str to Image: {e}")
                raise ValueError(f"Cannot process image string. Use a PIL Image object, path, or base64.")
    
    # Ensure we have valid PIL Images
    for img, name in [(img_ref, "img_ref"), (img_aligned, "img_aligned")]:
        if not hasattr(img, 'size'):
            logging.error(f"{name} is type {type(img).__name__}, not a PIL Image")
            raise TypeError(f"{name} must be a PIL Image object or convertible to one")
    
    width, height = img_ref.size
    cols, rows = grid_size
    
    # Ensure dimensions are divisible by grid
    new_width = (width // cols) * cols
    new_height = (height // rows) * rows
    
    # Calculate crop coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop both images the same way
    ref_crop = img_ref.crop((left, top, right, bottom))
    aligned_crop = img_aligned.crop((left, top, right, bottom))
    
    return ref_crop, aligned_crop

load_dotenv()
#st.write("API key starts with:", os.getenv("OPENAI_API_KEY")[:10])

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

#  שגודלו ≥ 0.1 % מהמסגרת **או** כל שינוי ברור מירוק → חום/אפור/בז, גם אם < 0.1 %.  

TILE_COMPARE_PROMPT = """
🟢   התעלם לחלוטין מ-
• שינויי תאורה / צל / איזון-לבן / רעש-ISO / הבהובים.
• תנועות קלות של עלים, עשב, שיחים וענפים.
• פסי Letter-box או קרופ (מסגרת בצבע אחיד).
• גושים אחידים בצבע שנוגעים בגבול התמונה, גובה/רוחב ≤ 10 % מהציר.
• Patch בצבע אחיד (σGray < ±2 או entropy < 1.0) וגודלו < 500 px.
• מסנן קווים דקים: יחס-צירים ≥ 5 : 1 ו-רוחב-או-גובה ≤ 1 % מהציר או ≤ 15 px.

• פילטר צמחייה v3.2  
  Patch שבו ≥ 70 % פיקסלים בטווח ירוק-חום טבעי*  
  ◦ שטח < 3 %   ◦ min-dim < 120 px   ◦ changed_pixels_percent < 6 %  
  ◦ ΔSat < 0.20   → אל תדווח.  
  ⬆︎ חריג: אם edge-density > 0.15 או ΔBrightness ≥ 20 Gray → אל תפסול (ייתכן סלע/אבן גלויה).

• Edge-Strip Filter:  
  bbox נוגע בדופן (≤ 20 % × ≥ 50 %) + SSIM > 0.65 → התעלם.

• Letter-box Bar Filter:  
  bbox נוגע בדופן; Gray ≤ 30 או ≥ 225; σGray < ±3 → התעלם.

• פילטר חיתוך-משתנה (Trans-Shift משופר):  
  – מצא הזחה גלובלית (ECC / median-flow).  
  – אם |dx|,|dy| ≤ 25 px ו-≥ 60 % מהפיקסלים חווים אותה הזחה בשני הצירים → הזחה.  
  – אחרת, לכל bbox: overlap ≥ 70 % & SSIM ≥ 0.75 → חוסר-יישור; התעלם.

• Interior-Patch Filter:  
  הפעל רק אם ‎> 60 % משטח ה-bbox במרחק < 5 % מהקצה ו-הזחה ≤ 25 px.

• ⚠️  High-Contrast Small-Object (v2):  
  אם מופיע/נעלם Patch ש-  
  ◦ changed_pixels_percent ≥ 0.3 %  
  ◦ min-dim ≥ 15 px או שטח ≥ 0.3 %  
  ◦ (ΔBrightness ≥ 15 Gray **או** ΔHue ≥ 15°)  
  ◦ אם Saturation-before < 0.25 → אין דרישת ΔSat ≥ 0.25  
  → כן לדווח (“אובייקט קטן אך בולט”).

• נקודת צמחייה/קרקע קטנה (< 2 % או < 50×50 px) המופיעה/נעלמת, אם אין תזוזה ≥ 3 % מציר התמונה.

טווח ירוק-חום טבעי = Hue ≈ 60-140°, Saturation < 0.5, |R-G| < 15.

🔵  סנן מראש אם כל הבאים:  
• changed_pixels_percent < 0.3 **וכן** < 300 px.  
• bbox במרחק < 2 % מכל גבול.  
• ΔHue < 20° ו-ΔBrightness < 15.

🔴  החזר `"change_detected": true` רק אם אחד:  
1. אובייקט בולט (≥ 20 % מהמסגרת) הופיע/נעלם.  
2. אובייקט זהה זז ≥ max(3 % מהמסגרת, 25 % מגודלו, 50 px).  
3. רצועה רציפה (כביש, תעלה, פס) ≥ 50 % אורך → Δ-מיקום ≥ 30 px ו-Δ-שטח ≥ 5 %.  
4. Patch חדש/נעלם (סלע, אדמה, אספלט, שלולית, ערימה)  
   • אם min-dim ≥ 30 px → די בשטח ≥ 0.3 %.  
   • אם 15 px ≤ min-dim < 30 px → חייב לעמוד בכלל High-Contrast v2  
     (ΔBrightness ≥ 15 Gray **או** ΔHue ≥ 15°) **וגם** changed_pixels_percent ≥ 0.3 %.  
   • ואינו נפסל ע״י הכללים הירוקים/כחולים.

↩︎  החזר **רק** JSON בעברית, בלי ```:

{
"change_detected": true/false,
"reason": "תיאור קצר",
"bbox_before": [x1, y1, x2, y2],
"bbox_after":  [x1’, y1’, x2’, y2’],
"movement_px": 0-999,
"changed_pixels_percent": 0-100,
"confidence": 0-100
}
"""


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
    # --- BYPASS when no temporal filtering requested ---
    if win == 0 or cube.shape[0] <= 1:
        return True
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
                x0, x1 = c * tw, (c + 1) * tw   # BUGFIX – correct X‑axis upper bound
                t1 = ref_gray[y0:y1, x0:x1]
                t2 = aln_gray[y0:y1, x0:x1]
                # --- robust adaptive SSIM window ---
                min_dim = min(t1.shape[0], t1.shape[1])
                # choose the largest odd window ≤ min(min_dim, 7)
                candidate = min(min_dim, 7)
                if candidate % 2 == 0:
                    candidate -= 1
                win = candidate
                if win < 3:                       # tile too small – treat as identical
                    cube[idx, r, c] = 0.0
                else:
                    cube[idx, r, c] = 1.0 - ssim(t1, t2, win_size=win)
    return cube

# -------------------------------------------------------

def _build_aligned_pairs(path_before: str, path_after: str, fps_target: float):
    """
    Heavy routine – extract frames & build aligned pairs once.
    Stored in st.session_state to avoid recomputation on every rerun.
    Returns list[dict] with keys: idx, comp_img, b64_1, b64_2, inliers
    """
    frames_before = _extract_frames(path_before, fps_target)
    frames_after  = _extract_frames(path_after, fps_target)
    pairs = []
    grid_val = int(st.session_state.get("grid_size", 3))
    # calculate src-frame interval so we can map sampled idx ↔ original frame #
    import cv2
    fps_src = cv2.VideoCapture(path_before).get(cv2.CAP_PROP_FPS) or 1
    interval = int(round(fps_src / fps_target)) if fps_target else 1
    for idx, (f1, f2) in enumerate(zip(frames_before, frames_after), start=1):
        aligned_f2, inliers = _align_images(f1, f2)
        ref_crop, aligned_crop = _crop_to_overlap(f1, aligned_f2, grid_size=(grid_val, grid_val))
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
            "inliers": inliers,
            "frame_idx": (idx - 1) * interval,
            "path_before": path_before,
            "path_after":  path_after
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
            0.40
        )
        st.session_state["grid_size"] = st.number_input(
            "Grid size",
            min_value=1,
            value=4
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

        # ----- חלון אימות טמפורלי -----
        st.session_state["ctx_window"] = st.slider(
            "חקירה מתקדמת (מספר זוגות לכל כיוון)",
            1, 4, 2)
        
        # ----- חלון אימות טמפורלי במילישניות -----
        st.session_state["ctx_ms"] = st.slider(
            "חלון חקירה מתקדמת (מילישניות)",
            100, 2000, 200, step=100)

        st.session_state["diff_mask_thr"] = st.number_input(
            "Diff‑mask threshold (% pixels changed)",
            min_value=0.5,
            max_value=50.0,
            value=10.0,
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
            # gather currently‑checked pairs (before we render the buttons)
            selected_ids = [
                p["idx"] for p in st.session_state.ground_pairs
                if st.session_state.get(f"chk_pair_{p['idx']}", False)
            ]
            # --- Select / Deselect all helpers ---
            col_sel_all, col_clear_all, col_run = st.columns(3)
            with col_sel_all:
                if st.button("סמן את כל הזוגות", key="btn_select_all"):
                    for p in st.session_state.ground_pairs:
                        st.session_state[f"chk_pair_{p['idx']}"] = True
                    _safe_rerun()  # refresh UI
            with col_clear_all:
                if st.button("נקה בחירה", key="btn_clear_all"):
                    for p in st.session_state.ground_pairs:
                        st.session_state[f"chk_pair_{p['idx']}"] = False
                    _safe_rerun()
            with col_run:
                if st.button("נתח זוגות", key="btn_run_selected", type="primary") and selected_ids:
                    _run_pairs_analysis(selected_ids, custom_prompt)
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

        # -------- FINAL-REPORT TAB --------
        with tab_report:
            report = st.session_state.get("report_data", [])
            if not report:
                st.write("אין ממצאים להצגה עדיין.")
            else:
                total_changes = len(report)
                import time
                run_ts  = st.session_state.get("report_start_ts", "—")
                runtime = st.session_state.get("report_runtime", 0.0)
                elapsed = time.strftime("%H:%M:%S", time.gmtime(runtime)) if runtime else "—"
                st.markdown(
                    f"**תאריך הרצה:** {run_ts} &nbsp;&nbsp; "
                    f"**משך:** {elapsed} &nbsp;&nbsp; "
                    f"**סה\"כ שינויים:** **{total_changes}**"
                )
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
    # --- select where debug images go ---
    dbg_target = st.session_state.get("tile_debug_container") or st
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
            min_dim = min(ref_gray.shape[0], ref_gray.shape[1])
            candidate = min(min_dim, 7)
            if candidate % 2 == 0:
                candidate -= 1
            win = candidate
            if win < 3:
                diff_ssim = 0.0   # skip tiny tiles
            else:
                diff_ssim = 1.0 - ssim(ref_gray, aligned_gray, win_size=win)
            diff_values.append(diff_ssim)      # NEW
            position_desc = f"חלק {x+1},{y+1} - שורה {y+1}, עמודה {x+1}"
            if show_before:
                dbg = Image.new("RGB", (ref_tile.width + aligned_tile.width, ref_tile.height))
                dbg.paste(ref_tile, (0, 0))
                dbg.paste(aligned_tile, (ref_tile.width, 0))
                dbg_target.image(dbg, caption=f"Before SSIM – {position_desc}", use_container_width=True)
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
        for b64_r, b64_a, desc, _, box in candidates[:top_k]:   # NEW
            ref_img = Image.open(io.BytesIO(base64.b64decode(b64_r)))  # FIXED typo
            aln_img = Image.open(io.BytesIO(base64.b64decode(b64_a)))  # FIXED typo
            comp = Image.new("RGB", (ref_img.width + aln_img.width, ref_img.height))  # NEW
            comp.paste(ref_img, (0, 0)); comp.paste(aln_img, (ref_img.width, 0))      # NEW
            dbg_target.image(comp, caption=desc, use_container_width=True)            # NEW                    # NEW

    # ---------- STEP-2: optional segmentation filter (Mask-R CNN) ----------
    if use_segmentation:
        score_thr = float(st.session_state.get("seg_score_thr", 0.50))
        iou_thr   = float(st.session_state.get("seg_iou_thr", 0.30))
        filtered = []
        for b64_r, b64_a, desc, diff, box in candidates:
            ref_tile  = Image.open(io.BytesIO(base64.b64decode(b64_r)))  # FIXED typo
            aln_tile  = Image.open(io.BytesIO(base64.b64decode(b64_a)))  # FIXED typo
            # keep the tile only if Mask‑RCNN detects NEW objects
            boxes = _maskrcnn_new_objects(
                ref_tile, aln_tile,
                score_thr=score_thr,
                iou_thr=iou_thr
            )
            if not boxes:
                continue
            # 👉 keep ORIGINAL tiles (no red overlay) for GPT
            filtered.append((b64_r, b64_a, desc, diff, box))
        candidates = filtered

        # ---------- DEBUG : view tiles that passed the YOLO test ----------
        if show_post and candidates:
            st.subheader("🎯 Tiles after Segmentation filter")
            for b64_r, b64_a, desc, _, box in candidates[:top_k]:
                ref_img = Image.open(io.BytesIO(base64.b64decode(b64_r)))  # FIXED typo
                aln_img = Image.open(io.BytesIO(base64.b64decode(b64_a)))  # FIXED typo
                comp  = Image.new("RGB", (ref_img.width + aln_img.width, ref_img.height))
                comp.paste(ref_img, (0, 0)); comp.paste(aln_img, (ref_img.width, 0))
                dbg_target.image(comp, caption=desc, use_container_width=True)

    # ---------- STEP-3: sort & return ----------
    tile_after_yolo = len(candidates)

    # keep tiles as‑is (no red border)
    candidates = [(r_b64, a_b64, desc, box) for r_b64, a_b64, desc, diff_v, box in candidates]

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
    return [(r, a, d, box) for r, a, d, box in candidates[:top_k]]

# ---------- NEW: helper – temporal window extraction ----------
def _extract_frames_window(video_path: str,
                           center_idx: int,
                           win_frames: int,
                           sharp_th: float = SHARPNESS_THRESHOLD):
    """
    מחזיר [(idx, PIL)] עבור שני פריימים בלבד: אחד ב-offset ‎-win_frames ואחד ב-+win_frames, לאחר סינון חדות.
    """
    import cv2
    from PIL import Image
    out = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center_idx = max(0, min(center_idx, total - 1))
    # נבחר רק שני אינדקסים: אחד לפני ואחד אחרי
    target_idxs = {center_idx - win_frames, center_idx + win_frames}
    # גבולות תקינים
    target_idxs = {idx for idx in target_idxs if 0 <= idx < total}

    for idx in sorted(target_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if not _is_frame_sharp(frame, sharp_th):
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append((idx, Image.fromarray(rgb)))

    cap.release()
    return out

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

# ---------- אימות שינוי בהקשר טמפורלי ----------
# מקבל גם `pairs_shared` – רשימת dict קטנים ממנה כל Thread יכול לקרוא.
def _confirm_change_with_temporal_context(pair: dict,
                                          box: tuple,
                                          window_ms: int = 500,
                                          prompt: str = COMMON_HEBREW_PROMPT):
    """
    מאמת אם שינוי באריח עקבי לאורך כמה פריימים (±window_ms מילישניות).
    מחזיר (ok, reason)
    """
    if window_ms is None:
        window_ms = int(st.session_state.get("ctx_ms", 500))
    
    # Extract frames directly from videos based on time
    path_before = pair.get("path_before")
    path_after = pair.get("path_after")
    frame_idx = pair.get("frame_idx", 0)
    
    if not path_before or not path_after:
        return False, "אין מידע על מקור הוידאו"
    
    # Convert ms to frames (assuming 30fps as default)
    cap = cv2.VideoCapture(path_before)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    
    win_frames = int(window_ms / 1000.0 * fps)
    
    # Extract frames around the current frame
    sharp_th = float(st.session_state.get("sharp_th", SHARPNESS_THRESHOLD))
    frames_before = _extract_frames_window(path_before, frame_idx, win_frames, sharp_th)
    frames_after = _extract_frames_window(path_after, frame_idx, win_frames, sharp_th)
    
    # ----- build (frame_idx, before_img, after_img) tuples -----
    tiles: list[tuple[int, Image.Image | None, Image.Image | None]] = []
    for f_idx, pil_img in frames_before:
        tiles.append((f_idx, pil_img.crop(box), None))          # before only

    for f_idx, pil_img in frames_after:
        # try to match an existing BEFORE frame (within ±2 idx)
        matched = False
        for i, (idx_b, before_img, after_img) in enumerate(tiles):
            if abs(idx_b - f_idx) <= 2:
                tiles[i] = (idx_b, before_img, pil_img.crop(box))
                matched = True
                break
        if not matched:
            tiles.append((f_idx, None, pil_img.crop(box)))      # after only

    # keep only complete pairs (both before & after present)
    complete_tiles = [(b, a) for _, b, a in tiles if b is not None and a is not None]
    if len(complete_tiles) < 2:
        return False, "אין מספיק פריימים לאימות טמפורלי"

    # replace old variable name
    tiles = complete_tiles
    chat = [{
        "role": "system",
        "content": [{"type":"text",
                     "text":"אתה מודל בינה-מלאכותית שמאשר או פוסל שינוי מהותי. החזר JSON עם change_detected ו-reason."}]
    },{
        "role": "user",
        "content": [{"type":"text","text":prompt}]
    }]
    for i,(bef,aft) in enumerate(tiles,1):
        chat[1]["content"].extend([
            {"type":"text","text":f"Frame {i} – BEFORE"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_pil_to_b64(bef)}"}},
            {"type":"text","text":f"Frame {i} – AFTER"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{_pil_to_b64(aft)}"}},
        ])
    try:
        resp = call_gpt_with_retry({
            "model": DEPLOYMENT,
            "messages": chat,
            "response_format": {"type": "json_object"},  # force JSON reply
            "max_tokens": 5000,
            "temperature": 0,
            "reasoning_effort": "high",
        })
        # -------- extract text or tool‑call JSON --------
        txt = ""
        if resp and resp.choices:
            msg = resp.choices[0].message
            txt = (msg.content or "").strip()

            # If content is empty, try tool_calls (JSON arrives here for o‑series)
            if not txt and hasattr(msg, "tool_calls") and msg.tool_calls:
                args = msg.tool_calls[0].get("arguments", "")
                if isinstance(args, str):
                    txt = args.strip()
                else:
                    import json as _json
                    txt = _json.dumps(args, ensure_ascii=False).strip()

        import logging, textwrap
        logging.info("[DEBUG] temporal‑confirm txt (first 200): "
                     + textwrap.shorten(txt, width=200, placeholder=" …"))

        if not txt:
            return False, "GPT חזר בתשובה ריקה"

        # Remove ```json fences if present
        if txt.startswith("```"):
            import re
            txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt).rstrip("`").strip()

        # Try strict JSON first
        import json, re
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            # Fallback: extract first {...} block
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                raise
            data = json.loads(m.group(0))

        return bool(data.get("change_detected")), data.get("reason", "ללא הסבר")

    except Exception as e:
        # Show first 120 chars of GPT reply to aid debugging
        short_txt = (txt[:120] + "…") if txt else "—"
        return False, f"שגיאה באימות: {e} | תשובת GPT חלקית: {short_txt}"

# ...existing imports...
import time      # NEW
# ...existing code...
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    force=True      # ensure our config overrides default Streamlit handler
)
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

# ---------- thread‑safety lock for Mask‑RCNN inference ----------
_MASK_LOCK = Lock()

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
    # Guard inference with the global lock for thread safety
    with _MASK_LOCK:              # 🔒 thread‑safe
        with torch.no_grad():
            pred_b = model([to_tensor(img_before)])[0]
            pred_a = model([to_tensor(img_after)])[0]

    # keep high-score detections
    keep_b = pred_b["scores"] >= score_thr
    keep_a = pred_a["scores"] >= score_thr
    boxes_b, labels_b = pred_b["boxes"][keep_b], pred_b["labels"][keep_b]   # FIX
    boxes_a, labels_a = pred_a["boxes"][keep_a], pred_a["labels"][keep_a]

    new_boxes = []
    for box_a, lbl_a in zip(boxes_a, labels_a):
        same_cls = (labels_b == lbl_a).nonzero(as_tuple=False).squeeze(1)
        if len(same_cls) == 0:
            new_boxes.append(box_a.int().tolist())
            continue
        if ops.box_iou(box_a.unsqueeze(0), boxes_b[same_cls]).max().item() < iou_thr:
            new_boxes.append(box_a.int().tolist())
    logging.info(f"Mask-RCNN kept {len(new_boxes)} new objects "
             f"(score ≥ {score_thr}, IoU ≤ {iou_thr})")
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
                    draw.rectangle([x1 + x_off, y1, x2 + x_off, y2 + x_off],
                                   outline="red", width=4)
        except Exception as e:
            logging.warning(f"Segmentation overlay failed: {e}")
    return comp
# --------------------------------------------------------------------

# ...existing code...

# ---------- GLOBAL PROMPT HANDLING ----------
DEFAULT_PROMPT = (
    "אתה סוכן בינה מלאכותית להשוואת תמונות; התייחס רק לשינויים מהותיים באזור המסומן."
)

def prompt_text_area(label: str = "GPT prompt (applies everywhere):"):
    """
    Render a single text‑area for the user‑defined GPT prompt and keep it
    in ``st.session_state['user_prompt']``.  
    Returns the current prompt string.
    """
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = DEFAULT_PROMPT
    st.session_state["user_prompt"] = st.text_area(
        label,
        value=st.session_state["user_prompt"],
        height=100
    )
    return st.session_state["user_prompt"]

def get_user_prompt() -> str:
    """Safely return the current user prompt (or the default one)."""
    return st.session_state.get("user_prompt", DEFAULT_PROMPT)
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

    # --- timing metadata ------------------------------------------------
    import datetime, time
    import logging
    start_time = time.time()
    st.session_state["report_start_ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Ensure Mask‑RCNN is fully loaded once in the UI thread
    _get_maskrcnn_model()

    progress = st.progress(0.0)
    status   = st.empty()
    total    = len(selected_ids)
    done     = 0

    # helper to locate pair object once
    def _get_pair(idx: int):
        return next((p for p in st.session_state.ground_pairs if p["idx"] == idx), None)

    # --- decide whether we need Streamlit debug output (UI thread) ----
    debug_tiles_requested = any([
        st.session_state.get("show_before_tiles"),
        st.session_state.get("show_pre_tiles"),
        st.session_state.get("show_post_tiles"),
        st.session_state.get("show_diff_tiles"),
        st.session_state.get("show_tile_debug")
    ])
    # Create a container to keep debug tiles separate from the main column
    tile_debug_container = (
        st.expander("🎞️ Tiles sent to GPT (debug)", expanded=False)
        if st.session_state.get("show_tile_debug", False) else None
    )
    st.session_state["tile_debug_container"] = tile_debug_container

    # --- build task list once, independent of debug flags ---
    # רשימת זוגות לשיתוף בין Threads (שומר רק Bytes – Picklable!)
    pairs_shared = []
    tasks = []
    for idx in selected_ids:
        pair = _get_pair(idx)
        if not pair:
            continue
        # Extract tiles in the UI thread so the list is identical
        # regardless of debug check‑boxes
        st.session_state.current_pair_idx = idx
        tiles = _extract_focused_regions(
            pair["ref"],
            pair["aligned"],
            grid_size=(
                int(st.session_state.get("grid_size", 3)),
                int(st.session_state.get("grid_size", 3))
            ),
            top_k=int(st.session_state.get("top_k", 30)),
            min_ssim_diff=float(st.session_state.get("MIN_SSIM_DIFF", 0.7)),
            use_segmentation=bool(st.session_state.get("use_segmentation", True))
        )
        # נוסיף גרסה קלה למשקל (PIL → bytes) לשיתוף בין Threads
        pairs_shared.append({
            "idx": pair["idx"],
            "ref": pair["ref"],
            "aligned": pair["aligned"]
        })
        tasks.append((idx, pair, tiles))

    # --- run tasks in parallel across pairs ---
    def _process_single_pair(pair_idx, pair, tiles, pairs_shared):
        """
        Analyze all tiles for a single pair (NO Streamlit calls here).
        Returns a list of (txt_full, comp_b64, tile_b64, box) for changed tiles.
        """
        import threading, time, logging
        import concurrent.futures
        import json
        thread_name = threading.current_thread().name
        logging.info(f"[{thread_name}]  Pair {pair_idx}: {len(tiles)} tiles ready")
        t_start = time.time()
        results = []
        debug_entries = []
        # --- run GPT calls for all tiles concurrently ---
        def _gpt_for_tile(args):
            t_idx, b64_r, b64_a, position_desc, box = args
            few_shot_example = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a placeholder example for few-shot learning."
                    }
                ]
            }
            tile_prompt = [
                few_shot_example,
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "אתה מודל בינה-מלאכותית שתפקידו להשוות שתי תמונות ולזהות שינוי מהותי באריח. החזר JSON תקני בלבד."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_r}"}},
                        {"type": "text", "text": "---"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}}
                    ]
                }
            ]
            try:
                resp = call_gpt_with_retry({
                    "model": DEPLOYMENT,
                    "messages": tile_prompt,
                    "max_tokens": 1024,
                    "temperature": 0,
                })
            except Exception as e:
                logging.warning(f"GPT call exception for pair {pair_idx} tile {t_idx}: {e}")
                return None

            raw_txt = resp.choices[0].message.content if resp and hasattr(resp, "choices") else ""
            stripped = raw_txt.strip()
            if stripped.startswith("```"):
                stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
                stripped = stripped.rstrip("`").strip()
            logging.info(f"[thread_name] GPT raw pair {pair_idx} tile {t_idx}: {stripped}")
            return t_idx, stripped, position_desc, b64_r, b64_a, box

        max_tile_workers = min(4, len(tiles))  # limit to 4 concurrent calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_tile_workers) as tile_executor:
            fut_to_args = {
                tile_executor.submit(_gpt_for_tile, (idx, *tile)): tile
                for idx, tile in enumerate(tiles, start=1)
            }
            for fut in concurrent.futures.as_completed(fut_to_args):
                res = fut.result()
                if res is None:
                    continue
                t_idx, stripped, position_desc, b64_r, b64_a, box = res
                try:
                    data = json.loads(stripped)
                except Exception:
                    data = None
                if data:
                    description = (data.get("reason") or data.get("description") or "").strip()
                    confidence  = data.get("confidence", 0)
                    moved_px    = data.get("movement_px", 0)
                    changed_pct = data.get("changed_pixels_percent", 0)
                else:
                    description = ""
                    confidence  = 0
                    moved_px    = 0
                    changed_pct = 0
                txt_full = (f"**{position_desc}** – {description or '—'} "
                            f"(px {changed_pct}\u202F%, move {moved_px}px, conf {confidence}%)")
                # תמיד שומר עותק לדיבוג
                debug_entries.append(
                    (txt_full, _compose_b64_side_by_side(b64_r, b64_a))
                )
                if not data or not data.get("change_detected"):
                    continue
                comp_b64  = _compose_pair_b64_with_box(pair["ref"], pair["aligned"], box)
                tile_b64  = _compose_b64_side_by_side(b64_r, b64_a)
                results.append((txt_full, comp_b64, tile_b64, box))
        elapsed = time.time() - t_start
        logging.info(f"[{thread_name}] ✅ Pair {pair_idx} finished – "
                     f"{len(results)} changes, {elapsed:.1f}s")
        
        # ---------- אימות בטווח טמפורלי ----------
        confirmed, reasoning_entries = [], []
        for txt_full, comp_b64, tile_b64, box in results:
            ok, reason = _confirm_change_with_temporal_context(
                pair,  # Now we pass the entire pair dict instead of just the index
                box,
                window_ms=int(st.session_state.get("ctx_ms", 500)),  # Use time window in milliseconds
                prompt=custom_prompt)
            reasoning_entries.append((comp_b64, ok, reason))
            if ok:
                confirmed.append((txt_full, comp_b64, tile_b64, box))
        results = confirmed
        return results , debug_entries , reasoning_entries

    results = []
    max_workers = min(4, len(tasks)) if tasks else 1
    logging.info(f"Submitting {len(tasks)} pairs to ThreadPoolExecutor "
                 f"(max_workers={max_workers})"
                 f"– debug flags: pre={st.session_state.get('show_pre_tiles')}, "
                 f"post={st.session_state.get('show_post_tiles')}, "
                 f"diff={st.session_state.get('show_diff_tiles')}, "
                 f"gpt_tile={st.session_state.get('show_tile_debug')}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_to_idx = {
            executor.submit(
                _process_single_pair, idx, pair, tiles, pairs_shared
            ): idx
            for idx, pair, tiles in tasks          # ← הלולאה הייתה חסרה
        }
        for fut in as_completed(fut_to_idx):
            pid = fut_to_idx[fut]
            try:
                changes, dbg, pair_reason = fut.result()
                results.append((pid, changes, dbg)) # (pair_idx, list_of_results)
                logging.info(f"↩️  Pair {pid} collected – {len(changes)} changes")
            except Exception as exc:
                logging.warning(f"Pair {pid} generated an exception: {exc}")
            done += 1
            progress.progress(done / total)
            logging.info(f"Progress: {done}/{total} pairs finished")

    # --- render results in UI thread (keep order) ---
    for pair_idx, pair_changes, pair_debug in sorted(results, key=lambda x: x[0]):
        # --- הצגת דיבוג (גם false) ---
        if st.session_state.get("show_tile_debug", False):
            target = tile_debug_container if tile_debug_container else st
            for dbg_txt, dbg_tile_b64 in pair_debug:
                target.image(
                    f"data:image/jpeg;base64,{dbg_tile_b64}",
                    caption=dbg_txt,
                    use_container_width=True
                )
        # צבירת הסברים טמפורליים
        st.session_state.setdefault("temporal_debug", []).extend(pair_reason)
        pair_changes_html = []
        for txt_full, comp_b64, tile_b64, box in pair_changes:
            if st.session_state.get("show_tile_debug", False):
                target = tile_debug_container if tile_debug_container else st
                target.image(
                    f"data:image/jpeg;base64,{tile_b64}",
                    caption=txt_full,
                    use_container_width=True
                )
            pair_changes_html.append(txt_full)
            _add_report_entry(pair_idx, comp_b64, tile_b64, txt_full, box)

        if pair_changes_html:
            md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                  f'### ממצאים לזוג {pair_idx}<br><br>' +
                  "<br><br>".join(pair_changes_html) + "</div>")
        else:
            md = (f'<div dir="rtl" style="background:#f7f7f7;padding:8px;border-radius:8px">'
                  f'### ממצאים לזוג {pair_idx}<br><br>לא נמצאו שינויים.</div>')
        st.session_state[f"gpt_txt_{pair_idx}"] = md
        st.markdown(md, unsafe_allow_html=True)

    status.text("הניתוח הסתיים!")
    progress.empty()
    st.session_state["report_runtime"] = time.time() - start_time   # ← store runtime

    # --- הצגת צעדי האימות הטמפורלי ---
    if st.session_state.get("temporal_debug"):
        with st.expander("🧠 צעדי האימות הטמפורלי", expanded=False):
            for img_b64, ok, reason in st.session_state["temporal_debug"]:
                st.image(f"data:image/jpeg;base64,{img_b64}",
                         caption=("✅ " if ok else "❌ ") + reason,
                         use_container_width=True)
# -----------------------------------------------------------
#
# ...existing code...