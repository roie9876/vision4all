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
import time

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

def detect_object_in_image(ref_image, target_image, description):
    # logging.debug("Entered detect_object_in_image function")
    
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
                    "text": (
                        "You are an AI trained to analyze images and compare objects across different images. "
                        "Your task is to determine whether a specific object, described by the user, is present in the target images with high certainty. "
                        "The object will be described by its characteristics, which are provided as a parameter called {description} "
                        "When given the {description} of an object, examine the target images and identify if the object or a similar-looking object appears in them. "
                        "Consider factors such as image quality, angle, and lighting conditions, and provide an assessment of your confidence in the identification. "
                        "The user will provide the following information: "
                        "- {description}: [Detailed characteristics of the object] "
                        "- Source image characteristics: [Details about the source image] "
                        "- Target image context: [Context about the target images] "
                        "Your goal is to confirm with high certainty the presence of the described object in the target images and explain your reasoning. "
                        "Example input: "
                        "- Description: Yellow reflective vest with gray stripes "
                        "- Source image characteristics: Clear image with visible reflective stripes "
                        "- Target image context: Various angles and lighting conditions "
                        "Please provide your analysis and confidence level in percentage."
                        "the answer need to be in Hebrew"
                    )
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
                    "text": f"האם אתה רואה את אותו {description} בתמונת המקור והיעד"
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

    # Try extracting usage if available
    if hasattr(response, "usage") and response.usage:
        st.session_state.total_tokens_used += response.usage.total_tokens

    # Log the full response
    # logging.debug(f"Response: {response}")

    # Parse the response
    try:
        result_text = response.choices[0].message.content
        # logging.debug(f"Result text: {result_text}")
    except Exception as e:
        # logging.error(f"Error parsing response: {e}")
        result_text = "Error occurred while processing the images."

    return result_text

def detect_objects_in_images(ref_image, target_images, description):
    # logging.debug("Entered detect_objects_in_images function")
    
    # Convert reference image to base64
    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    ref_image_base64 = image_to_base64(ref_image)

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
                }
            ]
        }
    ]

    # Add target images to the chat prompt
    for target_image in target_images:
        target_image_base64 = image_to_base64(target_image)
        chat_prompt[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{target_image_base64}"
            }
        })
        chat_prompt[1]["content"].append({
            "type": "text",
            "text": f"האם אתה רואה את אותו {description} בתמונת המקור והיעד"
        })

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

    # Try extracting usage if available
    if hasattr(response, "usage") and response.usage:
        st.session_state.total_tokens_used += response.usage.total_tokens

    # Log the full response
    # logging.debug(f"Response: {response}")

    # Parse the response
    try:
        result_text = response.choices[0].message.content
        # logging.debug(f"Result text: {result_text}")
    except Exception as e:
        # logging.error(f"Error parsing response: {e}")
        result_text = "Error occurred while processing the images."

    #st.write(result_text)
    return result_text

def summarize_results(results, description):
    # logging.debug("Entered summarize_results function")

    results_str = "\n".join(results)

    # Prepare the chat prompt for summarization
    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that specializes in summarizing image analysis results in hebrew. "
                "Your goal is to provide concise and accurate summaries of multiple analyses, focusing on key findings, "
                "confidence levels, and any notable patterns or exceptions. Ensure that important high-confidence identifications "
                "are highlighted and clearly communicated."
            )
        },
        {
            "role": "user",
            "content": (
                f"פלט ניתוח התמונות:\n{results_str}\n\n"
                f"אנא ספק סיכום סופי בעברית על סמך התוצאות הללו. "
                f"You are an AI assistant tasked with summarizing the results of image analyses in hebrew. Each analysis determines the presence "
                f"of a specific {description}, described by the user, in target images. Your summary should focus on key findings and "
                "ensure no important information is missed. The input will include multiple analysis results, each containing: "
                f"- Object description: {description} "
                #  "- Presence confirmation (Yes/No) "
                "Your summary should include: "
                f"1. requested to find {description} in the video"
                # "1. Number and percentage of images where the {description} was confirmed present. "
                "2. The highest confidence level achieved across all analyses. "
                "3. Summary of key observations or patterns, especially those with high confidence levels. "
                "4. Any exceptions or uncertainties that should be noted, with an emphasis on the reasons for lower confidence levels. "
                "Please provide a summary in Hebrew, encapsulating the key findings and confidence levels, ensuring that important "
                "high-confidence identifications are highlighted. Example input: "
                ""
            )
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

    # Try extracting usage if available
    if hasattr(response, "usage") and response.usage:
        st.session_state.total_tokens_used += response.usage.total_tokens

    # Log the full response
    # logging.debug(f"Response: {response}")

    # Parse the response
    try:
        summary_text = response.choices[0].message.content
        # logging.debug(f"Summary text: {summary_text}")
    except Exception as e:
        # logging.error(f"Error parsing response: {e}")
        summary_text = "Error occurred while summarizing the results."

    #st.write("סיכום ביניים:")
    #st.write(summary_text)
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
                # Start timer
                if "total_tokens_used" not in st.session_state:
                    st.session_state.total_tokens_used = 0

                start_time = time.time()

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                st.video(video_path)

                frames = extract_frames(video_path, sample_rate=sample_rate)
                st.write(f"Extracted {len(frames)} frames from the video.")
                
                # Display all extracted frames
                #st.write("Extracted Frames:")
                # for frame_path in frames:
                #     frame_image = Image.open(frame_path)
                #     st.image(frame_image, use_column_width=True)

                # Batch frames for fewer OpenAI calls
                batch_size = 5
                results = []
                for i in range(0, len(frames), batch_size):
                    batch_frames = [Image.open(frame_path) for frame_path in frames[i:i+batch_size]]
                    result_text = detect_objects_in_images(ref_image, batch_frames, object_description)
                    #if "yes" in result_text.lower() or "כן" in result_text.lower():
                    
                    #st.write("סיכום ביניים")
                    #st.write(result_text)
                    results.append(result_text)

                if results:
                    summary = summarize_results(results,object_description)
                    #st.write("האוביקט שחיפשנו הוא")
                    #st.write(object_description)
                    st.subheader("סיכום")
                    #st.write("סיכום:")
                    st.markdown(f'<div dir="rtl">{summary}</div>', unsafe_allow_html=True)
                    #st.write(summary)

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

                st.write(f"Total analysis time: {elapsed_str}")
                st.write(f"Total tokens used: {st.session_state.total_tokens_used}")
                st.write(f"Approximate cost: ${total_price:.4f}")

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
