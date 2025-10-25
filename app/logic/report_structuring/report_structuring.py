import json
import streamlit as st
import re
import logging
import copy
from typing import Tuple
from streamlit.components.v1 import html
from shared.chroma_interface import query
from shared.llm_interface import query_llm
from logic.report_structuring.json_to_html import json_to_html_table
from config import CONFIG
import streamlit as st
import re
from typing import Tuple
import logging
import json
import langid
from logic.report_structuring.validation_utils import (
    flatten_fields, set_in_json,
    validate_locally, llm_validate
)
from logic.report_structuring.prompts import PromptKey, get_prompt

JSON_PLACEHOLDER = CONFIG['report_structuring']['json_placeholder']
FILLING_MODE = CONFIG['report_structuring']['filling_mode']

logger = logging.getLogger('root')
logger.info(f"Filling mode: {FILLING_MODE}")

class ReportStructuringProcessor:
    # Modified: added pipeline_format parameter
    def __init__(self, progress_placeholder, result_placeholder, json_fill_mode=None):
        self.json_fill_mode = json_fill_mode # Added: json fill mode
        self.progress_placeholder = progress_placeholder
        self.result_placeholder = result_placeholder
        self.progress_bar = self.progress_placeholder.progress(0, text="Initializing...")

    def process_freetext(self, freetext):
        """Main processing function for structuring a radiology report."""

        self.progress_bar.progress(10, text="Detecting language...")
        self.language, confidence = langid.classify(freetext)
        logger.info(f"Extracted language: {self.language}")

        self.progress_bar.progress(20, text="Extracting metadata...")
        mod, area = self.extract_metadata(freetext).values()
        mod = mod.upper()
        logger.info(f"Extracted metadata: modality={mod}, area={area}")

        self.progress_bar.progress(40, text="Extracting search phrase...")
        phrase = self.extract_search_phrase(freetext, mod, area)
        logger.info(f"Extracted search phrase: {phrase}")

        self.progress_bar.progress(60, text="Selecting template...")
        template_name, template = self.rag_llm_template_selection(phrase, mod)
        logger.info(f"Selected template: {template_name}")

        self.progress_bar.progress(80, text=f"Filling {template_name}…")
        filled_template = self.fill_template(template, freetext)

        self.progress_bar.progress(100, text="Done")
        self.progress_placeholder.empty()

        matches = re.search(r'\{.*\}', filled_template, re.DOTALL)
        if matches:
            json_result = json.loads(matches.group(0))
        else:
            raise ValueError(f"Failed to detect JSON in  result: {json_result}")

        with self.result_placeholder.container():

            st.info(f"Template selected from Database: **{template_name}**")
            st.json(json_result, expanded=True)

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_result, indent=2),
                file_name="structured_report.json",
                mime="application/json"
            )

            html = json_to_html_table(json_result)
            st.download_button(
                label="📄 Download HTML",
                data=html,
                file_name="structured_report.html",
                mime="text/html"
            )
            

    def rag_llm_template_selection(self, freetext, modality) -> Tuple[list[str], str]:
        """
        RAG + LLM-based template selection:
        1. Query vector database for top-N templates.
        2. Filter templates by modality.
        3. Ask LLM to pick the best one among filtered candidates.
        """
        results = query(
            freetext, 
            n_results=5,
            where={"$and": [
                {"modality": {"$in": [modality, "MULTIMODAL"]}},
                {"language": {"$eq": self.language}}
            ]}
        )

        logger.info(f"Template query results: {results}")

        metas_list = results["metadatas"][0]
        docs_list = results["documents"][0]

        # Filter by modality
        filtered_candidates = [
            (meta["filename"], doc) 
            for meta, doc in zip(metas_list, docs_list) 
        ]

        # print out the length of filtered candidates
        logger.info(f"Found {len(filtered_candidates)} templates for modality '{modality}' "
                    f"and language {self.language} in RAG selection: {[name for name, _ in filtered_candidates]}")

        if not filtered_candidates:
            raise ValueError(f"No templates found for modality '{modality}' and language {self.language} in RAG selection")

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

        if FILLING_MODE == "step-by-step":
            # return self.fill_template_with_context(template, freetext)
            return self.fill_template_step_by_step(template, freetext)
        elif FILLING_MODE == "step-by-step-context":
            return self.fill_template_with_context(template, freetext)

        elif FILLING_MODE == "at-once":
            return self.fill_template_at_once(template, freetext)
        else:
            raise ValueError(f"Unknown filling mode: {FILLING_MODE}")

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
    
    def fill_template_with_context(self, template, freetext):
        """
        Refined step-by-step filling: carries forward prior answers so
        each prompt sees the full schema, all completed fields, and the report.
        """
         ##
        # 1. Parse the JSON template and prepare state
        original = json.loads(template)
        partial = {}                       # stores answers so far
        fields = flatten_fields(original)  # e.g. ["HISTORY", "FINDINGS.CTA_HEAD", ...]
        total = len(fields)

        # 2. Fill each field in sequence
        for idx, field in enumerate(fields):
            # 2a. Reconstruct the schema with already-filled values
            merged = copy.deepcopy(original)
            for k, v in partial.items():
                set_in_json(merged, k, v)

            logger.info(f"[{idx+1}/{total}] Filling field: {field}")
            self.progress_bar.progress(
                80 + int(20 * (idx + 1) / total),
                text=f"Filling field {idx+1}/{total}: {field}"
            )
            # 2b. Build the mini-prompt
            prompt_parts = []
            # Optional CoT for nested/complex fields
            if "." in field:
                prompt_parts.append(
                    get_prompt(PromptKey.COT_STEPS, self.language)
                )
            prompt_parts.append(
                get_prompt(PromptKey.STEP_FIELD, self.language).format(
                    schema=json.dumps(original, indent=2),
                    partial_json=json.dumps(partial, indent=2),
                    report=freetext,
                    field=field,
                )
            )
            full_prompt = "\n\n".join(prompt_parts)
            logger.debug(f"Prompt for field '{field}':\n{full_prompt}") #
            # st.code(full_prompt, language="yaml")
            # 2c. Query the LLM
            answer = query_llm(full_prompt).strip()
            old_answer = answer

            # st.text(f" Initial answer for `{field}`:") #
            # st.code(answer, language="plain") #

            # 2d. On-the-fly validation & retry if needed
            MAX_RETRIES = 2
            retry_prompt = get_prompt(PromptKey.RETRY_FIELD, self.language).format(field=field, old=old_answer, report = freetext)
            used_local = False
            used_llm = False
            for attempt in range(1, MAX_RETRIES + 1):
                # first try the fast local check, then the LLM yes/no check
                # if validate_locally(field, answer, freetext) or llm_validate(field, answer, freetext):
                #     break
                if validate_locally(field, answer, freetext):
                    used_local = True
                    break
                if llm_validate(field, answer, freetext):
                    used_llm = True
                    break
                logger.warning(f"[{field}] validation attempt {attempt}/{MAX_RETRIES} failed: {answer!r}")
                # st.text(f" Retry #{attempt} for field `{field}`:") #
                # st.code(retry_prompt, language="yaml") #
                answer = query_llm(retry_prompt).strip()
                # st.text(f" Retry #{attempt} answer for `{field}`:") #
                # st.code(answer, language="plain") #
            else:
                # after MAX_RETRIES, give up
                logger.error(f"[{field}] all {MAX_RETRIES} validation attempts failed; defaulting to 'Not mentioned'")
                answer = "Not mentioned"
            # st.text(f" {field}: local_valid={used_local}  llm_valid={used_llm}  final_value={answer!r}")

            # 2e. Save the validated answer
            set_in_json(partial, field, answer)
            set_in_json(merged, field, answer)

        # 3. Return the fully populated JSON
        return json.dumps(merged, indent=2)



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
