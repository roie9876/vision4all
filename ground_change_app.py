import streamlit as st
import tempfile, os, io, base64, logging, time, concurrent.futures
import numpy as np, cv2
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import torch, torchvision
# ------------------------------------------------------------------
# CONSTANTS  (same values as video_summary_video.py)
MIN_SSIM_DIFF       = 0.7
MAX_DIM_FOR_GPT     = 2048
SHARPNESS_THRESHOLD = 120.0
COMMON_HEBREW_PROMPT = (
    "נתח את התמונה וספק תיאור בעברית. התמקד רק בשינויים אשר מוקפים בצבע אדום  כמו הופעה של אובייקט חדש "
    "או החסרה של אובייקט."
)
# ------------------------------------------------------------------
# HELPER FUNCTIONS  (copies – bodies omitted for brevity)
def _extract_frames(video_path: str, fps_target: float):
    # ...existing code from video_summary_video._extract_frames...
    pass

def _align_images(img_ref, img_to_align, max_features: int = 1000, good_match: int = 50):
    # ...existing code...
    pass

def _crop_to_overlap(img_ref: Image.Image, img_aligned: Image.Image, grid_size=(3, 3)):
    # ...existing code...
    pass

def _is_stable_change(idx, r, c, cube, th=MIN_SSIM_DIFF, win: int = 1):
    # ...existing code...
    pass

def _build_diff_cube(frames_before, frames_after, grid_size=(4, 4)):
    # ...existing code...
    pass

def _maskrcnn_new_objects(img_before, img_after, score_thr=0.50, iou_thr=0.30):
    # ...existing code...
    pass

def _compose_pair(ref_img: Image.Image, aligned_img: Image.Image, draw_seg=False):
    # ...existing code...
    pass

def _compose_pair_b64(img_l: Image.Image, img_r: Image.Image) -> str:
    # ...existing code...
    pass

def _add_report_entry(pair_idx: int, pair_b64: str, tile_b64: str, gpt_text: str):
    # ...existing code...
    pass

def _extract_focused_regions(img_ref, img_aligned, grid_size=(3,3), top_k=30,
                             min_ssim_diff=MIN_SSIM_DIFF, use_segmentation=True):
    # ...existing code...
    pass
# ------------------------------------------------------------------
# GPT-wrapper (minimal – uses your utils.call_azure_openai_with_retry)
from utils import call_azure_openai_with_retry, summarize_descriptions, DEPLOYMENT  # type: ignore
def _timed_gpt_call(payload: dict, label=""):
    start = time.time()
    resp  = call_azure_openai_with_retry(payload)
    st.write(f"⏱️ {label}: {time.time()-start:.1f}s")
    return resp
# ------------------------------------------------------------------
def _run_pairs_analysis(selected_ids, custom_prompt):
    # ...existing code (same as temp-fix block from video_summary_video)...
    pass
# ------------------------------------------------------------------
def run_ground_change_detection():
    """
    The entire ground-change Streamlit workflow (identical UI / logic
    to the version inside video_summary_video.py)
    """
    # ...existing code...
    pass
# ------------------------------------------------------------------
def _render_sidebar():
    st.sidebar.title("📋 תפריט")
    st.sidebar.markdown(
        "יישום יעודי לזיהוי שינויי קרקע בלבד.\n\n"
        "העלה סרטון “לפני” ו-“אחרי” ובחר זוגות לניתוח."
    )

def main():
    st.set_page_config(page_title="Ground-change detection",
                       page_icon="🌍", layout="wide")
    _render_sidebar()
    run_ground_change_detection()

if __name__ == "__main__":
    import sys, subprocess
    # Allow `python ground_change_app.py` → automatically call `streamlit run`
    if st.runtime.exists():
        main()
    else:
        subprocess.run(["streamlit", "run", sys.argv[0], *sys.argv[1:]])