import logging
from shared.llm_interface import query_llm
from shared.embedding_model import embedding_model
logger = logging.getLogger('root')
from vertexai.generative_models import GenerativeModel, GenerationConfig
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def cosine_similarity(query_emb, db_embs):
    # Ensure both inputs are 2D arrays
    query_emb = query_emb.reshape(1, -1)  # shape: (1, D)
    return sklearn_cosine_similarity(db_embs, query_emb).flatten()  # shape: (N,)

def softmax(x):
    b = 100
    e_x = np.exp(b*x) # - np.max(x))  # numerical stability
    return e_x / e_x.sum()

def retrieve_closest_icd_code(query_text, db, id_to_icd_map, k=1):
    """
    Retrieves the k closest ICD code descriptions for a given query text (extracted entity) from the chromaDB.
    """
    query_embedding = embedding_model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    indices = db.query(query_embeddings=[query_embedding], n_results=k)['ids'][0]
    distances = db.query(query_embeddings=[query_embedding], n_results=k)['distances'][0]

    '''similarities = cosine_similarity(query_embedding, db_embs)
    probabilities = softmax(similarities)'''

    results = []
    for i in range(k):
        index_id = str(indices[i])
        if index_id in id_to_icd_map:
            icd_info = id_to_icd_map[index_id]
            # softmax_idx = db_ids.index(index_id)

            results.append({
                "icd_code": icd_info["code"],
                "icd_description": icd_info["description"],
                "similarity_score": float(distances[i]),
                # "similarity_debug": float(similarities[softmax_idx]),
                # "softmax_retrieval": float(probabilities[softmax_idx])
            })
    return results

def llm_reasoning(closest_icds, phrase):
    """
    Uses LLM reasoning to select the most appropriate ICD code from the retrieved closest codes.
    """
    if not closest_icds:
        return None

    # Prepare the prompt for LLM reasoning
    prompt = f"Given the following ICD codes and their descriptions, select the most closly matches this medical context: {phrase}\n"
    for item in closest_icds:
        prompt += f"ICD Code: {item['icd_code']}\nDescription: {item['icd_description']}\n\n"
    
    prompt += "Output the selected ICD code only, without any additional text."

    selected_code = query_llm(prompt).strip()
    logger.info(f"LLM selected ICD code: {selected_code}")
    
    # Find the corresponding ICD code in the original list
    for item in closest_icds:
        if item["icd_code"] == selected_code:
            return item

    return None

def llm_thresholding(model_name, predicted_descriptions, report_text):
    """
    Uses LLM to finalize the predicted ICD codes based on their relevance to the medical report.
    """
    if not predicted_descriptions:
        return []

    # Prepare the prompt for LLM thresholding
    prompt = f"Given the following ICD codes descriptions, determine which ones are really relevant to this medical report: {report_text}\nDescriptions:\n"
    for description in predicted_descriptions:
        prompt += f"{description}\n"

    prompt += "Output only the relevant ICD codes descriptions, separated by a newline."

    # Parse the response to extract relevant ICD codes
    relevant_codes = query_llm(prompt).strip().split("\n")
    return [code.strip() for code in relevant_codes if code.strip()]