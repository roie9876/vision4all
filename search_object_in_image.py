import streamlit as st
import tempfile
import os
import io
import base64
import requests
import time
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Initialize logging
# logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def count_tokens(text):
    # For base64 strings, count the length directly
    return len(text)

def run_search_object_in_image():
    st.title("Search Object in Images")

    # Step 1: User uploads reference image
    uploaded_image = st.file_uploader("Choose a reference image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        ref_image = Image.open(uploaded_image)
        ref_image_name = uploaded_image.name
        st.image(ref_image, caption='Reference Image', use_container_width=True)

        # Step 2: Free-text description
        object_description = st.text_input("Describe the object to detect:", value="e.g., red car, wooden chair, etc.")

        if object_description:
            # Step 4: User uploads a target image
            target_image_upload = st.file_uploader("Upload a target image...", type=["jpg", "jpeg", "png"])
            if target_image_upload is not None:
                target_image = Image.open(target_image_upload)
                target_image_name = target_image_upload.name
                st.image(target_image, caption='Target Image', use_container_width=True)

                st.write(f"Processing: {target_image_upload.name}")
                # logging.debug("Calling detect_object_in_image function")
                result_text, elapsed_str, total_tokens_used, total_price = detect_object_in_image(ref_image, target_image, object_description)
                st.markdown(f"<div style='text-align: right;'>{result_text}</div>", unsafe_allow_html=True)
                st.write(f"Total analysis time: {elapsed_str}")
                st.write(f"Total tokens used: {total_tokens_used}")
                st.write(f"Approximate cost: ${total_price:.4f}")
                # logging.debug(f"Result text displayed: {result_text}")

def resize_and_compress_image(image, max_size=(800, 800), quality=95):
    image.thumbnail(max_size, Image.LANCZOS)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return Image.open(buffered)

def detect_object_in_image(ref_image, target_image, description):
    global total_tokens_used  # Access the global token counter

    # Start timer
    start_time = time.time()

    # logging.debug("Entered detect_object_in_image function")
    
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
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Log the full response
    # logging.debug(f"Response: {response}")

    # Try extracting usage if available
    if hasattr(response, "usage") and response.usage:
        total_tokens_used += response.usage.total_tokens

    # Parse the response
    try:
        result_text = response.choices[0].message.content
        # logging.debug(f"Result text: {result_text}")
    except Exception as e:
        # logging.error(f"Error parsing response: {e}")
        result_text = "Error occurred while processing the images."

    # End timer and calculate duration
    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Approximate cost calculation (example rate):
    cost_per_1k_tokens = 0.0015
    total_price = (total_tokens_used / 1000) * cost_per_1k_tokens

    return result_text, elapsed_str, total_tokens_used, total_price
