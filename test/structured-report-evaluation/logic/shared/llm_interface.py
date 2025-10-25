import os
import yaml
import logging
import requests
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig
from config import CONFIG
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger('root')

ENVIRONMENT = os.environ['ENVIRONMENT']
GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
GOOGLE_CLOUD_REGION = os.environ['GOOGLE_CLOUD_REGION']
VERTEX_MODEL_NAME = CONFIG['general']['gemini_model']

OLLAMA_MODEL = CONFIG['general']['ollama_model']
OLLAMA_ENDPOINT = CONFIG['general']['ollama_endpoint']

if ENVIRONMENT not in ['development', 'hospital']:
    raise ValueError(f"Unsupported environment: {ENVIRONMENT}. Please set ENVIRONMENT to 'development' or 'hospital'.")

logger.info("Initializing LLM interface...")
logger.info(f"Google Cloud Project: {GOOGLE_CLOUD_PROJECT}")
logger.info(f"Google Cloud Region: {GOOGLE_CLOUD_REGION}")
logger.info(f"Vertex model: {VERTEX_MODEL_NAME}")
logger.info(f"Ollama model: {OLLAMA_MODEL}")
logger.info(f"Ollama endpoint: {OLLAMA_ENDPOINT}")

if ENVIRONMENT == 'development':
    logger.info("Running in development environment, using Vertex AI model.")
    aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
    vertex_model = GenerativeModel(VERTEX_MODEL_NAME)
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=1.0,
        top_k=30,
    #     max_output_tokens=8192,
    )

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
    if ENVIRONMENT == 'development':
        return query_vertexai(prompt, generation_config=generation_config)
    elif ENVIRONMENT == 'hospital':
        return query_ollama(prompt)
    else:
        raise ValueError(f"Unsupported environment: {ENVIRONMENT}. Please set ENVIRONMENT to 'development' or 'hospital'.")


