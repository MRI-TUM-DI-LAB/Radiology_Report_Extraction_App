from vertexai.generative_models import GenerativeModel, GenerationConfig

def extract_entities_llm(text, model_name, temperature=0.0, max_tokens=500):
    gemini_model = GenerativeModel(model_name)

    prompt = f'''
    You are an expert medical assistant and radiology terminology coder. Your task is to extract all concepts from the following unstructured medical report that correspond to individual entities in the RadLex ontology.

    Focus your extraction ONLY on the following categories:
    - Radiologic observations and findings (e.g., “ground-glass opacity”, “consolidation”, “mass effect”)
    - Anatomical entities (e.g., “left lower lobe”, “right femur”, “thoracic spine”)
    - Imaging modalities and techniques
    - Radiologic procedures and interventions (e.g., “CT-guided drainage”, “fluoroscopy”)
    - Devices or substances used in imaging (e.g., “contrast agent”, “stent”, “catheter”)
    - Features and properties (e.g., structure, color, size)

    Instructions:
    - Extract only concepts that are explicitly mentioned in the report.
    - If a phrase includes multiple concepts (e.g., an imaging modality and an anatomical location or a property), split it into **separate entries**, each belonging to a single category from above.
    - Keep each extracted concept as **simple and atomic** as possible, using the exact wording from the report.
    - Ignore general clinical diagnoses not related to imaging.
    - Output **one concept per line**.

    Medical report:
    {text}
    '''

    response = gemini_model.generate_content(
            contents=prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
    
    return response.text


def reformulate_entity_synonyms(entity, model_name, num_variations=5):
    gemini_model = GenerativeModel(model_name)

    prompt = f"""You are a medical expert. 
    Reformulate the following medical term or phrase in {num_variations} different ways using appropriate medical synonyms or paraphrases.
    Return only each reformulation on a new line.

    Original term: "{entity}"

    Reformulations:"""

    response = gemini_model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            temperature=0.7,
            max_output_tokens=150
        )
    )
    reformulations = [line.strip() for line in response.text.split("\n") if line.strip()]
    return reformulations[:num_variations]
