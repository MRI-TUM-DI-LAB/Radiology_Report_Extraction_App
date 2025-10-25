import logging
from vertexai.generative_models import GenerationConfig
from shared.llm_interface import query_llm
logger = logging.getLogger('root')

def extract_entities_llm(text, temperature=0.0, max_tokens=500):
    prompt = f'''
    You are an expert medical assistant and ICD-10-CM coder. Your task is to extract all confirmed diseases, medical conditions, and relevant health factors that the patient *does* have, based on the following unstructured medical report.

    Focus your extraction ONLY on the following categories:
    - Diagnosed diseases and conditions (e.g., infections, chronic diseases, genetic disorders)
    - Clinical signs and symptoms (e.g., “fever”, “abdominal pain”)
    - Abnormal findings (lab, imaging, physical exam)
    - Injuries and external causes
    - Social and structural health factors (e.g., “homelessness”, “unemployment”, “smoker”)
    - Interactions with the healthcare system (e.g., “palliative care”, “cancer screening”, “hospitalization”)

    Instructions:
    - Extract only conditions explicitly confirmed by the report.
    - Ignore any conditions explicitly negated (e.g., “no signs of diabetes”).
    - For each extracted condition, return the **exact phrase** from the report confirming it.
    - If the extracted condition is confirmed but lacks important clinical details (such as location, cause, type, or severity), add “, unspecified” to the extracted phrase.
    - Output one phrase per line.

    Medical report:
    {text}
    '''

    response = query_llm(prompt,
                generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
    
    return response


def reformulate_entity_synonyms(entity, num_variations=5):

    prompt = f"""You are a medical expert. 
    Reformulate the following medical term or phrase in {num_variations} different ways using appropriate medical synonyms or paraphrases. 
    If the phrase contains the word "unspecified", leave that word unchanged in the reformulation.
    Return only each reformulation on a new line.

    Original term: "{entity}"

    Reformulations:"""

    response = query_llm(prompt,
                generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=150
        )
    )
    reformulations = [line.strip() for line in response.split("\n") if line.strip()]
    return reformulations[:num_variations]