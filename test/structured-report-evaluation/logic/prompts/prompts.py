from config import CONFIG
import logging
import os
import yaml
from enum import Enum

logger = logging.getLogger('root')

class PromptKey(Enum):
    CLASSIFY_TEMPLATE = "classify_template"
    FILL_TEMPLATE_STEP_BY_STEP = "fill_template_step_by_step"
    FILL_TEMPLATE_AT_ONCE = "fill_template_at_once"
    EXTRACT_METADATA = "extract_metadata"
    EXTRACT_SEARCH_PHRASE = "extract_search_phrase"

def load_prompts(directory="prompts"):
    prompts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            lang_code = os.path.splitext(filename)[0]
            logger.info(f"Loading prompts for language {lang_code}")
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[lang_code] = yaml.safe_load(f)

                # validate all keys are there
                if not isinstance(prompts[lang_code], dict):
                    raise Exception(f"Prompts for {lang_code} are not in the expected format")

                for key in PromptKey:
                    if key.value not in prompts[lang_code]:
                        raise Exception(f"Language {lang_code} is missing prompt {key.value}")
    return prompts

PROMPTS = load_prompts()
logger.info(f"Loaded languages: {PROMPTS.keys()}")

def get_prompt(key: PromptKey, language) -> str:
    if language not in PROMPTS.keys():
        raise Exception(f"Language {language} is not supported")

    if key.value not in PROMPTS[language].keys():
        raise Exception(f"Cannot find prompt {key.value} in language {language}")

    return PROMPTS[language][key.value]
