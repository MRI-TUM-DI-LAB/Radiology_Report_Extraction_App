from sentence_transformers import SentenceTransformer
import logging
from config import CONFIG
logger = logging.getLogger('root')

EMBEDDING_MODEL = CONFIG['embedding_model']['model']
LOCAL_FILES_ONLY = CONFIG['embedding_model']['local_files_only']

logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
logger.info(f"local_files_only: {LOCAL_FILES_ONLY}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL,
                                        trust_remote_code=True,
                                        local_files_only=LOCAL_FILES_ONLY)
def get_embedding_model():
    return embedding_model
