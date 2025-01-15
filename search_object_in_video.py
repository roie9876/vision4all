import streamlit as st
import tempfile
import os
import io
import base64
import requests
import logging
from PIL import Image
from utils import extract_frames
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from openai import AzureOpenAI
from requests.packages.urllib3.util.retry import Retry
from detect_change_in_video_and_summary import run_detect_change_in_video_and_summary

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

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

def detect_object_in_image(ref_image, target_image, description):
    logging.debug("Entered detect_object_in_image function")
    
    # Resize and compress images to reduce base64 size
    ref_image = resize_and_compress_image(ref_image)
    target_image = resize_and_compress_image(target_image)

    # Convert images to base64
    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    ref_image_base64 = image_to_base64(ref_image)
    target_image_base64 = image_to_base64(target_image)

    # Prepare the chat prompt
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                      "text": " אל תענה בכן ולא, עליך לבצע ניתוח מעמיק. ולדרג עד כמה אתה בטוח בתשובה שלך באחוזים"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{target_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"האם אתה רואה את אותו {description}  בתמונת המקור והיעד"
                }
            ]
        }
    ]

    # Generate the completion
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=chat_prompt,
        max_tokens=4096,
        temperature=0.1,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Log the full response
    logging.debug(f"Response: {response}")

    # Parse the response
    try:
        result_text = response.choices[0].message.content
        logging.debug(f"Result text: {result_text}")
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        result_text = "Error occurred while processing the images."

    return result_text

def summarize_results(results):
    logging.debug("Entered summarize_results function")

    # Prepare the chat prompt for summarization
    chat_prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people summarize information."
        },
        {
            "role": "user",
            "content": "Summarize the following results in Hebrew:\n" + "\n".join(results)
        }
    ]

    # Generate the completion
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

    # Log the full response
    logging.debug(f"Response: {response}")

    # Parse the response
    try:
        summary_text = response.choices[0].message.content
        logging.debug(f"Summary text: {summary_text}")
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        summary_text = "Error occurred while summarizing the results."

    return summary_text

def run_search_object_in_video():
    st.subheader("Search Object in Video")
    st.write("Upload a reference image, describe the object, and then upload a video to search for that object.")

    ref_image_file = st.file_uploader("Choose a reference image...", type=["jpg", "jpeg", "png"])
    if ref_image_file is not None:
        ref_image = Image.open(ref_image_file)
        st.image(ref_image, caption='Reference Image', use_container_width=True)

        object_description = st.text_input("Describe the object to detect:")
        if object_description:
            sample_rate = st.selectbox(
                "Select frame extraction rate:",
                options=[1, 2, 0.5, 4],
                format_func=lambda x: f"{x} frame{'s' if x != 1 else ''} per second",
                index=0
            )
            uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                st.video(video_path)

                frames = extract_frames(video_path, sample_rate=sample_rate)
                st.write(f"Extracted {len(frames)} frames from the video.")

                results = []
                for idx, frame_path in enumerate(frames):
                    frame_image = Image.open(frame_path)
                    result_text = detect_object_in_image(ref_image, frame_image, object_description)
                    if "yes" in result_text.lower() or "כן" in result_text.lower():
                        results.append(f"Frame {idx+1}: {result_text}")

                if results:
                    summary = summarize_results(results)
                    st.write("Summary:")
                    st.write(summary)

def main():
    st.title("Video Analysis Tool")
    menu = ["Search Object in Video", "Detect Change in Video and Summary"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Search Object in Video":
        run_search_object_in_video()
    elif choice == "Detect Change in Video and Summary":
        run_detect_change_in_video_and_summary()

if __name__ == '__main__':
    main()
