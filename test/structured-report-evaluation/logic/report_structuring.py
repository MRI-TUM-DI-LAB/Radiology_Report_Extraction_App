import json
import re
import logging
from typing import Tuple
from shared.chroma_interface import query
from shared.llm_interface import query_llm
from shared.prompts import *
from config import CONFIG
import re
from typing import Tuple
import logging
import json
import langid
from langdetect import detect
from shared.prompts import PromptKey, get_prompt
logger = logging.getLogger('root')

JSON_PLACEHOLDER = "__FILL__"

class ReportStructuringProcessor:
    def __init__(self, json_fill_mode=None):
        self.json_fill_mode = json_fill_mode # Added: json fill mode

    def process_freetext(self, freetext: str) -> dict:
        self.language, confidence = langid.classify(freetext)
        print(f"Detected language: {self.language}")
        mod, area = self.extract_metadata(freetext).values()
        print(f"Extracted modality: {mod}, area: {area}")
        phrase = self.extract_search_phrase(freetext, mod, area)
        print(f"Extracted search phrase: {phrase}")
        selected_name, selected_template = self.rag_llm_template_selection(phrase, mod)
        print(f"Selected template: {selected_name}")
        filled_template = self.fill_template(selected_template, freetext)
        matches = re.search(r'\{.*\}', filled_template, re.DOTALL)
        if matches:
            return json.loads(matches.group(0)), selected_name
        else:
            raise ValueError(f"Failed to detect JSON in result: {filled_template}")
        
        
    def rag_template_selection(self, freetext) -> Tuple[list[str], str]:
        """
        Perform RAG lookup and return the top 5 template filenames and the top-1 filename.
        """
        results = query(
            freetext,
            n_results=5
        )
        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        logger.info(f"Found {len(docs_list)} templates for RAG selection")
        return metas_list[0]["filename"], docs_list[0]
    
    def rag_llm_template_selection(self, freetext, modality) -> Tuple[list[str], str]:
        """
        RAG + LLM-based template selection:
        1. Query vector database for top-N templates.
        2. Filter templates by modality.
        3. Ask LLM to pick the best one among filtered candidates.
        """
        results = query(
            freetext, 
            n_results=10
        )  # 10 templates from RAG selection

        metas_list = results["metadatas"][0]
        docs_list = results["documents"][0]

        # Filter by modality
        filtered_candidates = [
            (meta["filename"], doc) 
            for meta, doc in zip(metas_list, docs_list) 
            if meta["modality"].lower() == modality.lower()
        ]
        # print out the length of filtered candidates
        print(f"Found {len(filtered_candidates)} templates for modality '{modality}' in RAG selection")

        # get the top 5 candidates
        filtered_candidates = filtered_candidates[:5]
        print(f"Filtered candidates: {[name for name, _ in filtered_candidates]}")

        if not filtered_candidates:
            raise ValueError(f"No templates found for modality '{modality}' in RAG selection")
        
        # Ask LLM to select the best template
        prompt = get_prompt(PromptKey.TEMPLATE_SELECTION, self.language).format(
            report=freetext,
            templates="\n".join(name for name, _ in filtered_candidates)
        )
        response = query_llm(prompt)
        logger.info(f"LLM selection response: {response}")
        # Parse response to get the selected template name
        selected_template_name = response.strip()
        selected_template = next((doc for name, doc in filtered_candidates if name == selected_template_name), None)

        if not selected_template:
            raise ValueError(f"Selected template '{selected_template_name}' not found in candidates")
        logger.info(f"Selected template: {selected_template_name}")

        return selected_template_name, selected_template
        


    def fill_template(self, template, freetext):
        filling_mode = CONFIG['report_structuring']['filling_mode']
        logger.info(f"Filling mode: {filling_mode}")

        if filling_mode == "step-by-step":
            return self.fill_template_step_by_step(template, freetext)
        elif filling_mode == "at-once":
            return self.fill_template_at_once(template, freetext)
        else:
            raise ValueError(f"Unknown filling mode: {filling_mode}")

    def fill_template_at_once(self, template, freetext):
        prompt = (
            f"{get_prompt(PromptKey.FILL_TEMPLATE_AT_ONCE, self.language)}"
            f" Current Template: \n\n {template} \n\n"
            f" Freetext Report: \n\n {freetext}"
        )
        prompt = f"{get_prompt(PromptKey.FILL_TEMPLATE_AT_ONCE, self.language)} Current Template: \n\n {template} \n\n Freetext Report: \n\n {freetext}"
        response = query_llm(prompt)
        return response.strip()

    def fill_template_step_by_step(self, template, freetext):
        data = json.loads(template)

        def walk_and_fill(obj):
            for key, val in obj.items():
                # Recurse into nested dicts
                if isinstance(val, dict):
                    walk_and_fill(val)

                # Handle lists: dict items vs placeholders
                elif isinstance(val, list):
                    for idx, item in enumerate(val):
                        if isinstance(item, dict):
                            walk_and_fill(item)
                        elif item == JSON_PLACEHOLDER:
                            prompt = get_prompt(PromptKey.FILL_TEMPLATE_STEP_BY_STEP, self.language).format(
                                field=key,
                                report=freetext
                            )
                            filled = query_llm(prompt).strip()
                            val[idx] = filled
                            logger.info(f"Filled placeholder in list at key '{key}': {filled}")

                # Handle standalone placeholders in dicts
                elif val == JSON_PLACEHOLDER:
                    prompt = get_prompt(PromptKey.FILL_TEMPLATE_STEP_BY_STEP, self.language).format(
                        field=key,
                        report=freetext
                    )
                    filled = query_llm(prompt).strip()
                    obj[key] = filled
                    logger.info(f"Filled placeholder at key '{key}': {filled}")

        walk_and_fill(data)
        return json.dumps(data, indent=2)


    def extract_metadata(self, report_text: str) -> dict:
        prompt = get_prompt(PromptKey.EXTRACT_METADATA, self.language).format(report=report_text)
        result = query_llm(prompt)
        logger.info(f"Metadata extraction result: {result}")
        matches = re.search(r'\{.*\}', result, re.DOTALL)
        if matches:
            return json.loads(matches.group(0))
        raise ValueError(f"Failed to detect JSON in metadata extraction result: {result}")

    def extract_search_phrase(self, report_text: str, modality: str, area: str) -> str:
        """Use Gemini to generate a concise search phrase."""
        prompt = (
            get_prompt(PromptKey.EXTRACT_SEARCH_PHRASE, self.language).format(
                language=self.language or '',
                modality=modality or '',
                area=area or '',
            )
            + "\nReport:\n" + report_text
        )
        result = query_llm(prompt)
        return result
