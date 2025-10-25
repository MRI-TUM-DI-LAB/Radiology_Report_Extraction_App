from vertexai.generative_models import GenerativeModel, GenerationConfig

def extract_icd(text, model_name, temperature=0.0, max_tokens=500):

    prompt = f'''You are a clinical coder, consider the medical report and assign the appropriate ICD 10 codes. Output one ICD code per line.
    
    Medical report:
    {text}

    ICD codes:
    '''

    gemini_model = GenerativeModel(model_name)
    response = gemini_model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )

    return response.text