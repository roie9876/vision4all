import os
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

# 2. Retrieve config from environment
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# 3. Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
    api_version=API_VERSION
)


