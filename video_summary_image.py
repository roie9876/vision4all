import streamlit as st
import os
import io
import base64
import logging
import requests
import time
from PIL import Image
from azure_openai_client import client
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

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

# Initialize token usage counter
total_tokens_used = 0

def resize_and_compress_image(image, max_size=(800, 800), quality=95):
    image.thumbnail(max_size, Image.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return Image.open(buffered)

def describe_image(image, content_prompt):
    global total_tokens_used  # Access the global token counter

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
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
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
        # Access usage correctly
        if hasattr(completion, 'usage') and hasattr(completion.usage, 'total_tokens'):
            total_tokens = completion.usage.total_tokens
            st.write(f"Total tokens used for this completion: {total_tokens}")
        else:
            pass
    except Exception as e:
        raise SystemExit(f"Failed to generate completion. Error: {e}")
    
    return description

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

def _display_patches_with_text(patches_info):
    """
    Render each cropped patch together with the text that belongs to it.

    patches_info: Iterable[Tuple[PIL.Image.Image, str]]
    """
    for idx, (img, txt) in enumerate(patches_info, 1):
        col_img, col_txt = st.columns([1, 3])
        with col_img:
            st.image(img, caption=f"Patch #{idx}", use_column_width=True)
        with col_txt:
            st.markdown(txt)

def run_image_summary():
    global total_tokens_used  # Access the global token counter

    st.title("Image Summary")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    content_prompt = st.text_input("Enter the content prompt:", value="Describe the image in Hebrew")
    if uploaded_file is not None:
        if st.button("Process Image"):
            frame_path = handle_image_upload(uploaded_file)

            image = Image.open(frame_path)
            description, elapsed_str, total_tokens_used, total_price = summarize_image_analysis(image, content_prompt)
            
            if description:
                st.markdown(f'<div dir="rtl">{description}</div>', unsafe_allow_html=True)
                st.write(f"Total analysis time: {elapsed_str}")
                st.write(f"Total tokens used: {total_tokens_used}")
                st.write(f"Approximate cost: ${total_price:.4f}")
            else:
                st.error("Failed to generate description.")

            os.remove(frame_path)
    
    # Display the total number of tokens used
    # st.write(f"Total OpenAI tokens used: {total_tokens_used}")

    # old text-only output removed / replaced ↓
    _display_patches_with_text(chunks)

if __name__ == '__main__':
    run_image_summary()