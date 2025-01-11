# Vision4all

Vision-4all is a Streamlit-based application that allows users to analyze images and videos using Azure OpenAI services. The application provides functionalities to search for objects in images and videos, detect objects in videos/images, and summarize video/image descriptions in Hebrew.

## Features

- **Search Object in Image**: Upload a reference image and a target image to detect if the specified object appears in both images.
- **Detect Objects in Video/Image**: Upload a video or image to detect and list all objects present.
- **Video/Image Summary**: Upload a video or image to generate a summary description in Hebrew.

## Setup

### Prerequisites

- Python 3.7 or higher
- Azure OpenAI API key and endpoint

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/vision-4o.git
    cd vision-4o
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your Azure OpenAI credentials:
    ```properties
    AZURE_OPENAI_KEY=your_openai_api_key
    AZURE_OPENAI_ENDPOINT=your_openai_endpoint
    AZURE_OPENAI_API_VERSION=2024-08-01-preview
    AZURE_OPENAI_DEPLOYMENT=gpt-4o
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run vision.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Select an application from the sidebar:
    - **Video Summary**: Upload a video or image to generate a summary description in Hebrew.
    - **Search Object in Image**: Upload a reference image and a target image to detect if the specified object appears in both images.
    - **Detect Objects**: Upload a video or image to detect and list all objects present.

## Project Structure

- `vision.py`: Main entry point for the Streamlit application.
- `search_object_in_image.py`: Module for searching objects in images.
- `detected_objects.py`: Module for detecting objects in videos/images.
- `video_summary.py`: Module for summarizing video/image descriptions.
- `utils.py`: Utility functions for frame extraction, text summarization, and object detection.
- `.env`: Environment variables for Azure OpenAI credentials (not included in the repository).
- `requirements.txt`: List of required Python packages.

## Logging

Logs are saved to `app.log` in the project root directory. The log file is cleared at the start of each run.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
