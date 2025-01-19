import streamlit as st
import os
import io
import base64
import logging
import requests  # Add this import
from PIL import Image
from azure_openai_client import client
#from openai import AzureOpenAI
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from azure_openai_client import client

# Load environment variables
load_dotenv()

# Initialize the Azure OpenAI client
# endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# subscription_key = os.getenv("AZURE_OPENAI_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# client = AzureOpenAI(
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
#     api_version=api_version
# )

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

# logging.basicConfig(
#     level=logging.DEBUG,  # Change logging level to DEBUG to capture all logs
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("app.log"),  # Write logs to app.log
#         logging.StreamHandler()  # Also output logs to the console
#     ]
# )

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
    # chat_prompt = [
    #     {
    #         "role": "system",
    #         "content": "You are an AI assistant that helps people find information."
    #     },
    #     {
    #         "role": "user",
    #         "content": content_prompt
    #     },
    #     {
    #         "role": "user",
    #         "content": f"data:image/jpeg;base64,{encoded_image}"
    #     }
    # ]
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

    # Log the chat prompt for debugging
    # logging.debug(f"Chat prompt: {chat_prompt}")

    # Generate the completion
    try:
        completion = client.chat.completions.create(
            model=deployment,
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
            # logging.info(f"Tokens used for this completion: {total_tokens}")
        else:
            # logging.warning("Total tokens used not found in the API response.")
            pass
    except Exception as e:
        # logging.error(f"Failed to generate completion. Error: {e}")
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

def run_image_summary():
    global total_tokens_used  # Access the global token counter

    st.title("Image Summary")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    content_prompt = st.text_input("Enter the content prompt:", value="Describe the image in Hebrew")
    if uploaded_file is not None:
        if st.button("Process Image"):
            # logging.debug(f"Uploaded file: {uploaded_file.name}")
            frame_path = handle_image_upload(uploaded_file)

            image = Image.open(frame_path)
            description = describe_image(image, content_prompt)
            
            if description:
                #st.write("Description:")
                #st.write(description)
                # Display the description in the UI with right-to-left text direction using HTML
                st.markdown(f'<div dir="rtl">{description}</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to generate description.")

            os.remove(frame_path)
    
    # Display the total number of tokens used
    #st.write(f"Total OpenAI tokens used: {total_tokens_used}")

    #st.write("No file uploaded.")