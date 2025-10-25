from vertexai.generative_models import GenerativeModel, GenerationConfig

def retrieve_closest_icd_code(query_text, db, id_to_icd_map, k=1):
    """
    Retrieves the k closest ICD code descriptions for a given query text (extracted entity) from the chromaDB.
    """
    indices = db.query(query_texts=[query_text], n_results=k)['ids'][0]
    distances = db.query(query_texts=[query_text], n_results=k)['distances'][0]
       
    results = []
    for i in range(k):
        index_id = str(indices[i])
        if index_id in id_to_icd_map:
            icd_info = id_to_icd_map[index_id]
            results.append({
                "icd_code": icd_info["code"],
                "icd_description": icd_info["description"],
                "similarity_score": float(distances[i]) 
            })

    return results

def llm_reasoning(model_name, closest_icds, query, temperature=0.0):
    """
    Uses LLM reasoning to select the most appropriate ICD code from the retrieved closest codes.
    """
    if not closest_icds:
        return None

    # Prepare the prompt for LLM reasoning
    prompt = f"Given the following ICD codes and their descriptions, select the code that most closely matches this medical context: {query}\n"
    for item in closest_icds:
        prompt += f"ICD Code: {item['icd_code']}\nDescription: {item['icd_description']}\n\n"
    
    prompt += "Output the selected ICD code only, without any additional text."

    # Use Google Gemini or any other LLM to reason about the best code
    gemini_model = GenerativeModel(model_name)
    response = gemini_model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=100
        )
    )

    # Parse the response to extract the selected ICD code
    selected_code = response.text.strip()
    print(selected_code)
    
    # Find the corresponding ICD code in the original list
    for item in closest_icds:
        if item["icd_code"] == selected_code:
            return item

    return None

def llm_thresholding(model_name, predicted_descriptions, report_text, temperature=0.0):
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

    # Use Google Gemini or any other LLM to filter the codes
    gemini_model = GenerativeModel(model_name)
    response = gemini_model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=500
        )
    )

    # Parse the response to extract relevant ICD codes
    relevant_codes = response.text.strip().split("\n")
    return [code.strip() for code in relevant_codes if code.strip()]