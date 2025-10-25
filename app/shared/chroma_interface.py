import chromadb
from datetime import datetime
import uuid
import logging
import langid
import json
from config import CONFIG
from shared.embedding_model import embedding_model
import json
logger = logging.getLogger('root')

JSON_PLACEHOLDER = CONFIG['report_structuring']['json_placeholder']
CHROMA_PATH = CONFIG['report_structuring']['chromadb_path']
CHROMA_COLLECTION_NAME = CONFIG['report_structuring']['chromadb_collection_name']
DETECT_LANGUAGES = CONFIG['report_structuring']['detect_languages']

logger.info(f"ChromaDB path: {CHROMA_PATH}")
logger.info(f"ChromaDB collection name: {CHROMA_COLLECTION_NAME}")

DICOM_MODALITIES = [
    "AR", "AS", "AU", "BDUS", "BI", "BMD", "CR", "CT", "CTPROT", "DG", "DX",
    "ECG", "EPS", "ES", "FID", "GM", "HC", "HD", "IO", "IOL", "IVOCT", "IVUS",
    "KER", "KO", "LEN", "LS", "MG", "MR", "MRT", "MRPROT", "NM", "OAM", "OPT",
    "OPV", "OSS", "OT", "PLAN", "PR", "PT", "PX", "REG", "RESP", "RF", "RG",
    "RTDOSE", "RTIMAGE", "RTPLAN", "RTRECORD", "RTSTRUCT", "SEG", "SM", "SMR",
    "SR", "SRF", "TG", "US", "VA", "XA", "XC"
]

langid.set_languages(DETECT_LANGUAGES)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_reports = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
 
def upload_files(uploaded_files):
    for file in uploaded_files:
        try:
            logger.debug(f"Uploading file: {file.name}")
            content = file.read().decode("utf-8")

            num_placeholders = content.count(f'{JSON_PLACEHOLDER}')
            logger.debug(f"Number of {JSON_PLACEHOLDER} placeholders: {num_placeholders}")

            # --- Language Detection ---

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
            keys = keys.replace("_", ", ").replace("-", ", ").replace(":", ", ").lower()
            logger.debug(f"Extracted keys from JSON: {keys}")

            language, confidence = langid.classify(keys)

            logger.info(f"Detected language {language} for: {file.name} with confidence {confidence:.2f}")

            # --- Embedding ---
            embedding_text = file.name + "\n" + content

            embedding = embedding_model.encode(embedding_text, normalize_embeddings=True).tolist()
            doc_id = str(uuid.uuid4())

            # --- Extract Modality from filename and normalize ---
            raw_modality = file.name.split("_")[0].upper() if "_" in file.name else "UNKNOWN"
            if raw_modality in ['MRI', 'MRT']:
                modality = "MR"
            elif raw_modality in ['XRAY']:
                modality = "CR"
            elif raw_modality in ['CTA']:
                modality = "CT"
            else:
                modality = raw_modality

            if modality not in DICOM_MODALITIES and modality != "MULTIMODAL":
                logger.warning(f"Unknown DICOM modality '{modality}' in file {file.name}. Defaulting to 'UNKNOWN'.")
                modality = "UNKNOWN"
                
            logger.debug(f"File name: {file.name}")
            logger.debug(f"File size: {file.size}")
            logger.debug(f"File ID: {doc_id}")
            logger.debug(f"File language: {language}")
            logger.debug(f"File modality: {modality}")
            logger.debug(f"Number of {JSON_PLACEHOLDER} placeholders: {num_placeholders}")

            # --- Upsert into ChromaDB ---
            collection_reports.upsert(
                documents=[content],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[{
                    "filename": file.name,
                    "size": file.size,
                    "upload_time": datetime.now().isoformat(),
                    "language": language,
                    "modality": modality,
                    "num_placeholders": num_placeholders
                }],
            )

            logger.info("Uploaded %s with ID %s", file.name, doc_id)
            return True
        
        except Exception as e:
            logger.error(f"Error uploading file {file.name}: {e}")
            return False

def query(query_text, n_results=1, where=None):
    results = collection_reports.query(
        query_embeddings=[embedding_model.encode(query_text).tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"],
        where=where
    )
    return results

def set_language(report_id, new_language):
    try:
        # Update the metadata for the report
        collection_reports.update(
            ids=[report_id],
            metadatas=[{"language": new_language}]
        )
        logger.info(f"Updated language for report ID {report_id} to {new_language}")
        return True
    except Exception as e:
        logger.error(f"Error updating language: {e}")
        return False

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