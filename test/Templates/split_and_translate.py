import json
import os
import re
from pathlib import Path
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig

SOURCE = Path("templates_clean_reworked.json")          # your big file
DEST_DIR_EN = Path.cwd() / "../../templates/en/Keno"
DEST_DIR_DE = Path.cwd() / "../../templates/de/Keno"

os.makedirs(DEST_DIR_EN, exist_ok=True)
os.makedirs(DEST_DIR_DE, exist_ok=True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './vertexai_credentials.json'

GOOGLE_CLOUD_PROJECT = 'llm-structuring'
GOOGLE_CLOUD_REGION = 'us-central1'
VERTEX_MODEL_NAME = "gemini-2.0-flash"
aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
vertex_model = GenerativeModel(VERTEX_MODEL_NAME)
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=1.0,
    top_k=30,
#     max_output_tokens=8192,
)


def translate_keys(obj):
    "Use LLM to translate keys in the JSON object."
    prompt = (
        "Translate all keys (including nested ones) of the following JSON object from English to German."
        "Output the translated JSON object with the same structure.\n\n"
        f"{json.dumps(obj, indent=2)}"
    )
    response = vertex_model.generate_content(prompt, generation_config=generation_config)
    translated_json = response.text.strip()
 #   print(f"Translated JSON: {translated_json}...")  # Print first 1000 chars for debugging
    matches = re.search(r'\{.*\}', translated_json, re.DOTALL)
    if matches:
        return json.loads(matches.group(0))
    else:
        print(f"No valid JSON found in the response for {obj}.")
        print(f"Response: {translated_json}")

def translate_name(name):
    "Translate the name part of the template from English to German."
    prompt = f"Translate the following name from English to German: {name}. Output only the translated name."
    response = vertex_model.generate_content(prompt, generation_config=generation_config)
    return response.text.strip()


with SOURCE.open() as f:
    templates = json.load(f)

for name, payload in templates.items():
    orig_path = DEST_DIR_EN / f"{name.lower()}.json"
    print(f"Processing {name}...")
    with orig_path.open("w") as out:
        json.dump(payload, out, indent=4)
    print(f"Wrote {orig_path}")

    # 2. Translate keys and write
    translated_payload = translate_keys(payload)
    tech = name.split("_")[0]
    "Detailed name is everything after the first underscore"
    detailed_name = name[name.index("_") + 1:]
    name_translated = translate_name(detailed_name)
    trans_path = DEST_DIR_DE / f"{tech.lower()}_{name_translated.lower()}.json"
    with trans_path.open("w") as out:
        json.dump(translated_payload, out, indent=4, ensure_ascii=False)
    print(f"Wrote translated {trans_path}")

print(f"Done! {len(templates)} files")