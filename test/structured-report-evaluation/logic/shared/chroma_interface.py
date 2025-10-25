import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from datetime import datetime
import uuid
import logging
import langid
from langdetect import detect
from shared.utils import clean_html, clean_json, extract_all_tables
from shared.llm_interface import query_llm
import json
from shared.prompts import PromptKey, get_prompt
from config import CONFIG
import json
logger = logging.getLogger('root')

EMBEDDING_MODEL = CONFIG['report_structuring']['rag_embedding_model']

client = chromadb.PersistentClient(path="/Users/chenyang_yu/Desktop/TUM/radiology-app/main-app/app/template_chroma_db")


collection_reports = client.get_or_create_collection(name="reports")
logger.info(f"Using embedding model: {EMBEDDING_MODEL}")

embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
 
def upload_files(uploaded_files):

    existing_reports = collection_reports.get(include=["metadatas"])
    existing_ids = existing_reports["ids"]
    existing_metadatas = existing_reports["metadatas"]

    existing_filenames = {md["filename"]: doc_id for md, doc_id in zip(existing_metadatas, existing_ids)}

    for file in uploaded_files:
        if file.name in existing_filenames:
            logger.info(f"Skipping upload for {file.name} as it already exists with ID {existing_filenames[file.name]}")
            continue

        logger.info(f"Uploading file: {file.name}")
        content = file.read().decode("utf-8")

        # --- Language Detection ---

        langid.set_languages(['en', 'de'])


        def extract_keys(d):
            keys = []

            def recurse(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        keys.append(k)
                        recurse(v)
                elif isinstance(obj, list):
                    for item in obj:
                        recurse(item)

            recurse(d)
            return " ".join(keys)
        
        keys = extract_keys(json.loads(content))
        logger.info(f"Extracted keys from JSON: {keys}")
        keys = keys.replace("_", " ").replace("-", " ").replace(":", " ").lower()

        language, confidence = langid.classify(keys)
        logger.info(f"Detected language: {language}")

        # --- Category Classification ---

        c_prompt = get_prompt(PromptKey.CLASSIFY_TEMPLATE, language) + content
        logger.debug(f"Prompt for classification: {c_prompt}")
        
        # category = query_llm(c_prompt).strip().lower()
        category = "template"
        logger.info(f"Classified as category: {category}")


        # --- Embedding ---
        embedding_text = file.name.replace("_", " ").replace("-", " ").replace(":", " ").lower() + "\n" + content

        embedding = embedder.encode(embedding_text).tolist()
        doc_id = str(uuid.uuid4())

        # --- Extract Modality from filename and normalize ---
        raw_modality = file.name.split("_")[0].lower() if "_" in file.name else "unknown"
        if raw_modality in ['mri', 'mr', 'mrt']:
            modality = "MR"
        else:
            modality = raw_modality.lower()
            
        logger.info(f"File name: {file.name}")
        logger.info(f"File size: {file.size}")
        logger.info(f"File ID: {doc_id}")
        logger.info(f"File category: {category}")
        logger.info(f"File language: {language}")
        logger.info(f"File modality: {modality}")

        # --- Upsert into ChromaDB ---
        collection_reports.upsert(
            documents=[content],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "filename": file.name,
                "size": file.size,
                "upload_time": datetime.now().isoformat(),
                "category": category,
                "language": language,
                "modality": modality
            }],
        )

        logger.info("Uploaded %s with ID %s", file.name, doc_id)

def query(query_text, n_results=1):
    language, confidence = langid.classify(query_text)
    logger.info(f"Detected language: {language}")
    results = collection_reports.query(
        query_embeddings=[embedder.encode(query_text).tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"],
        where={
            "language": {"$eq": language}
        }
    )
    return results

def delete_report(report_id):
    try:
        collection_reports.delete(ids=[report_id])
        logger.info(f"Deleted report with ID {report_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        return False

def get_all_reports():
    return collection_reports.get(include=["documents", "metadatas", "embeddings"])