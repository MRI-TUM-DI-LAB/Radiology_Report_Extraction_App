import os
import logging
import requests
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig
from config import CONFIG
logger = logging.getLogger('root')

LLM_SERVICE = CONFIG['llm']['llm_service']
VERTEX_MODEL_NAME = CONFIG['llm']['vertexai']['vertexai_model']

OLLAMA_MODEL = CONFIG['llm']['ollama']['ollama_model']
OLLAMA_ENDPOINT = CONFIG['llm']['ollama']['ollama_endpoint']

logger.info("Initializing LLM interface...")

if LLM_SERVICE == 'vertexai':
    logger.info(f"Using Vertex model: {VERTEX_MODEL_NAME}")
    GOOGLE_CLOUD_PROJECT = CONFIG['llm']['vertexai']['google_cloud_project']
    GOOGLE_CLOUD_REGION = CONFIG['llm']['vertexai']['google_cloud_region']
    GOOGLE_APPLICATION_CREDENTIALS = CONFIG['llm']['vertexai']['google_application_credentials']
    logger.info(f"Google Cloud Project: {GOOGLE_CLOUD_PROJECT}")
    logger.info(f"Google Cloud Region: {GOOGLE_CLOUD_REGION}")
    logger.info(f"Google Application Credentials: {GOOGLE_APPLICATION_CREDENTIALS}")
    if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        raise ValueError("Google application credentials file not found. Please set the path in the config"
        " and ensure the file exists.")
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
    aiplatform.init(project=GOOGLE_CLOUD_PROJECT, 
                    location=GOOGLE_CLOUD_REGION
    )
    vertex_model = GenerativeModel(VERTEX_MODEL_NAME)
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=1.0,
        top_k=30
    )
if LLM_SERVICE == 'ollama':
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    logger.info(f"Ollama endpoint: {OLLAMA_ENDPOINT}")


def query_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[No response]")
    except requests.exceptions.RequestException as e:
        return f"[Error contacting Ollama API: {e}]"

def query_vertexai(prompt, generation_config=None) -> str:
    response = vertex_model.generate_content(prompt, generation_config=generation_config)
    return response.text.strip()

def query_llm(prompt: str, generation_config = None) -> str:
    if LLM_SERVICE == 'vertexai':
        return query_vertexai(prompt, generation_config=generation_config)
    elif LLM_SERVICE == 'ollama':
        return query_ollama(prompt)
    else:
        raise ValueError(f"Unsupported llm service: {LLM_SERVICE}. Please set llm_service to 'vertexai' or 'ollama'.")
    
