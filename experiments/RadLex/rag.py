from vertexai.generative_models import GenerativeModel, GenerationConfig
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def cosine_similarity(query_emb, db_embs):
    # Ensure both inputs are 2D arrays
    query_emb = query_emb.reshape(1, -1)  # shape: (1, D)
    return sklearn_cosine_similarity(db_embs, query_emb).flatten()  # shape: (N,)

def softmax(x):
    b = 100
    e_x = np.exp(b*x) # - np.max(x))  # numerical stability
    return e_x / e_x.sum()

def retrieve_closest_radlex_code(query_embedding, db, id_to_radlex_map, k=1):
    """
    Retrieves the k closest RadLex code descriptions for a given query text (extracted entity) from the chromaDB.
    """
    #indices = db.query(query_texts=['left gastric artery'], n_results=k)['ids'][0]
    #distances = db.query(query_texts=['left gastric artery'], n_results=k)['distances'][0]

    indices = db.query(query_embeddings=[query_embedding], n_results=k)['ids'][0]
    distances = db.query(query_embeddings=[query_embedding], n_results=k)['distances'][0]
    #query_embedding = db.query(query_texts=[query_text], n_results=k)['embeddings'][0]

    # Query all entries (k=len(db))
    # Example: retrieve from chromadb (if allowed)
    entries = db.get(include=['embeddings', 'metadatas'])
    db_embs = np.array(entries['embeddings'])
    db_ids = entries['ids']

    # Compute query embedding
    query_emb = query_embedding #db._embedding_function(query_text)[0]  # adjust depending on your setup
    #query_emb = db._embedding_function('left gastric artery')[0]
    #print('EMBEDDING FROM CHROMADB',query_emb)

    # Compute cosine similarities and softmax
    similarities = cosine_similarity(query_emb, db_embs)
    similarities_softmax = softmax(similarities)

    #print(np.amax(similarities_softmax),  np.amin(similarities_softmax), len(similarities_softmax))
    #print(np.amax(similarities),  np.amin(similarities))
    #idx_max_cos = str(np.argmax(similarities))
    #print('MAX SIMILARITIES: ',np.amax(similarities), id_to_radlex_map[idx_max_cos])

    #idx_max_softmax = str(np.argmax(similarities_softmax))
    #print('MAX SOFTMAX: ', np.amax(similarities_softmax), id_to_radlex_map[idx_max_softmax])

    index_id = str(indices[0])
    radlex_info = id_to_radlex_map[index_id]
    rid_match = radlex_info['description']

    # Sort similarities in ascending order and reorder softmax accordingly
    sorted_indices = np.argsort(similarities)  # ascending
    sorted_similarities = similarities[sorted_indices]
    sorted_softmax = similarities_softmax[sorted_indices]

    # Select every 10th element in the sorted list
    subset_similarities = sorted_similarities[::100]
    x = np.arange(len(subset_similarities))
    # Plot histogram
    plt.figure(figsize=(12, 5))
    bar_width = 0.4
    # Bar plot for similarities
    plt.bar(x, subset_similarities, width=bar_width, label='Cosine Similarity', color='skyblue')
    plt.xlabel("Sorted Entry Index")
    plt.ylabel("Score")
    plt.title(f"Cosine Similarities (every 100th entry), code: {rid_match}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot histogram
    plt.figure(figsize=(12, 5))
    bar_width = 0.4
    subset_softmax = sorted_softmax[45500:]
    x = np.arange(len(subset_softmax))
    # Compute maximum y value across both sets
    y_max = np.max(subset_softmax) * 1.1  # add 10% margin
    plt.bar(x, subset_softmax, width=bar_width, label='Softmax Probabilities', color='salmon')
    plt.xlabel("Sorted Entry Index")
    plt.ylabel("Score")
    plt.title(f"Softmax Similarities (every entry, last values), code: {rid_match}")
    plt.ylim(0, y_max)  # <--- this rescales the y-axis
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

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
                "softmax_retrieval": float(similarities_softmax[softmax_idx])
            })

    return results

def llm_reasoning(model_name, closest_radlex, query):
    """
    Uses LLM reasoning to select the most appropriate RadLex code from the retrieved closest codes.
    """
    if not closest_radlex:
        return None

    # Prepare the prompt for LLM reasoning
    prompt = f"Given the following RadLex codes and their descriptions, select the code that most closely matches this medical context: {query}\n"
    for item in closest_radlex:
        prompt += f"RadLex Code: {item['radlex_code']}\nDescription: {item['radlex_description']}\n\n"
    
    prompt += "Output the selected RadLex code only, without any additional text."

    # Use Google Gemini or any other LLM to reason about the best code
    gemini_model = GenerativeModel(model_name)
    response = gemini_model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_output_tokens=100
        )
    )

    # Parse the response to extract the selected RadLex code
    selected_code = response.text.strip()
    print(selected_code)
    
    # Find the corresponding RadLex code in the original list
    for item in closest_radlex:
        if item["radlex_code"] == selected_code:
            return item

    return None