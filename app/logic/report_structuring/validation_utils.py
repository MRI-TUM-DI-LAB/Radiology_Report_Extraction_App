import re
from difflib import SequenceMatcher
from shared.llm_interface import query_llm

def flatten_fields(schema: dict, prefix=""):
    """
    Turn nested JSON keys into a flat list like
    ['INDICATION', 'FINDINGS.supraspinatus', ...]
    """
    fields = []
    for k, v in schema.items():
        path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            fields += flatten_fields(v, path)
        else:
            fields.append(path)
    return fields

def set_in_json(obj: dict, path: str, value):
    """
    Given a dict and a dotted path like "FINDINGS.CTA_HEAD",
    walk/create nested dicts and set the final key to value.
    """
    parts = path.split(".")
    for key in parts[:-1]:
        obj = obj.setdefault(key, {})
    obj[parts[-1]] = value


def validate_locally(field: str, value: str, report: str) -> bool:
    """
    Fast Python check to see if the LLM’s answer is supported by the report:
      1. Reject empty strings.
      2. Accept the special "Not mentioned".
      3. Accept if >50% of the answer’s words appear in the report.
      4. Otherwise, accept if a fuzzy-match ratio > 0.6.
    """
    v = value.strip()
    if not v:
        return False
    if v == "Not mentioned":
        return True

    # Token overlap check (>50% of answer words in report)
    rpt_tokens = set(re.findall(r"\w+", report.lower()))
    val_tokens = set(re.findall(r"\w+", v.lower()))
    if val_tokens and len(val_tokens & rpt_tokens) / len(val_tokens) > 0.9:
        return True

    # Fuzzy substring check
    if SequenceMatcher(None, v, report).quick_ratio() > 0.95:
        return True

    return False

def llm_validate(field: str, content: str, report: str) -> bool:
    """
    Short LLM prompt that returns YES/NO if the proposed content for the field
    is supported by the report.
    """
    prompt = (
        f"Report:\n{report}\n\n"
        f"Field: {field}\n"
        f"Proposed content: \"{content}\"\n\n"
        "Based solely on the report text, is this the correct content for that field? "
        "Answer YES or NO."
    )
    resp = query_llm(prompt).strip().upper()
    return resp.startswith("Y")


    