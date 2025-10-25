import logging
from vertexai.generative_models import GenerationConfig
from shared.llm_interface import query_llm
from shared.embedding_model import embedding_model
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger(__name__)

def cosine_similarity(query_emb, db_embs):
    # Ensure both inputs are 2D arrays
    query_emb = query_emb.reshape(1, -1)  # shape: (1, D)
    return sklearn_cosine_similarity(db_embs, query_emb).flatten()  # shape: (N,)

def softmax(x):
    b = 100
    e_x = np.exp(b*x) # - np.max(x))  # numerical stability
    return e_x / e_x.sum()

def retrieve_closest_radlex_code(query_text, db, id_to_radlex_map, db_embs, db_ids, k=1):
    """
    db_embs: np.ndarray of all db embeddings
    db_ids: list of string IDs corresponding to db_embs
    """
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)

    # Query ChromaDB for top-k
    results_raw = db.query(query_embeddings=[query_embedding], n_results=k)
    indices = results_raw['ids'][0]
    distances = results_raw['distances'][0]

    # Compute cosine similarity and softmax using preloaded embeddings
    similarities = cosine_similarity(query_embedding, db_embs)
    probabilities = softmax(similarities)

    results = []
    for i in range(k):
        index_id = str(indices[i])
        if index_id in id_to_radlex_map:
            radlex_info = id_to_radlex_map[index_id]
            softmax_idx = db_ids.index(index_id)

            results.append({
                "radlex_code": radlex_info["code"],
                "radlex_description": radlex_info["description"],
                "similarity_score": float(distances[i]),
                "similarity_debug": float(similarities[softmax_idx]),
                "softmax_retrieval": float(probabilities[softmax_idx])
            })

    return results

def llm_reasoning(closest_radlex, query):
    """
    Given several candidate RadLex entries, prompt the shared LLM to pick
    the best one in context. Returns that single dict or None.
    """
    if not closest_radlex:
        return None

    # Build pick-the-best prompt
    prompt = f"Given this phrase: “{query}”\n"
    prompt += "Which RadLex code below best matches it? Reply with code only.\n\n"
    for item in closest_radlex:
        prompt += f"Code: {item['radlex_code']}\n"
        prompt += f"Desc: {item['radlex_description']}\n\n"

    # Ask the LLM via shared interface
    response = query_llm(
        prompt,
        generation_config=GenerationConfig(temperature=0.0, max_output_tokens=10)
    )
    choice = response.strip()

    # Find and return the matching dict
    for item in closest_radlex:
        if item["radlex_code"] == choice:
            return item

    logger.warning(f"LLM returned unexpected code: {choice}")
    return None
