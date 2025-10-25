import logging
from vertexai.generative_models import GenerationConfig
from shared.llm_interface import query_llm

logger = logging.getLogger(__name__)

def extract_entities_llm(text, temperature=0.0, max_tokens=500):
    """
    Use the shared LLM interface to extract RadLex entities from a free-text report.
    Returns a string with one concept per line.
    """
    prompt = f'''
You are an expert medical assistant and radiology terminology coder. Your task is to extract all concepts
from the following unstructured medical report that correspond to individual entities in the RadLex ontology.

Focus your extraction ONLY on the following categories:
- Radiologic observations and findings (e.g., “ground-glass opacity”, “consolidation”, “mass effect”)
- Anatomical entities (e.g., “left lower lobe”, “right femur”, “thoracic spine”)
- Imaging modalities and techniques
- Radiologic procedures and interventions (e.g., “CT-guided drainage”, “fluoroscopy”)
- Devices or substances used in imaging (e.g., “contrast agent”, “stent”, “catheter”)
- Features and properties (e.g., structure, color, size)

Instructions:
- Extract only concepts that are explicitly mentioned in the report.
- If a phrase includes multiple concepts (e.g., an imaging modality + anatomical location),
  split it into separate entries.
- Keep each extracted concept as simple and atomic as possible.
- Ignore general clinical diagnoses not related to imaging.
- Output one concept per line.

Medical report:
{text}
'''
    return query_llm(
        prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )

def reformulate_entity_synonyms(entity, num_variations=5):
    """
    Use the shared LLM interface to generate paraphrases for a given entity,
    up to num_variations, one per line.
    """
    prompt = f'''
You are a medical expert.
Reformulate the following medical term or phrase in {num_variations} different ways
using appropriate medical synonyms or paraphrases. Return only each reformulation on a new line.

Original term: "{entity}"

Reformulations:
'''
    response = query_llm(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.7,
            max_output_tokens=150
        )
    )
    return [line.strip() for line in response.split("\n") if line.strip()][:num_variations]
