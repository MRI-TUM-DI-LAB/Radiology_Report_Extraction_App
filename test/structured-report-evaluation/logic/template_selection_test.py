import re
import os

def extract_keywords(filename):
    name = os.path.splitext(filename)[0].upper()  # Removes extension like .json or .txt
    parts = name.split('_')
    
    modality = parts[0]
    body_parts = [p for p in parts[1:] if not p.isdigit()]
    body_str = '_'.join(body_parts)
    
    return modality, body_str


def check_mapping(mapping_dict):
    mismatches = []
    total_reports = len(mapping_dict)
    for report, template in mapping_dict.items():
        r_mod, r_body = extract_keywords(report)
        t_mod, t_body = extract_keywords(template)

        modality_match = r_mod == t_mod
        body_match = (r_body in t_body) or (t_body in r_body)

        if not (modality_match and body_match):
            mismatches.append((report, template, r_mod, t_mod, r_body, t_body))

    percentage = len(mismatches) / total_reports if total_reports > 0 else 0
    return mismatches, percentage

def check_mapping_topk(mapping_dict):
    """
    mapping_dict: Dict[str, List[str]]
        A dict mapping each report to a list of 5 predicted templates.
    """
    mismatches = []
    total_reports = len(mapping_dict)

    for report, templates in mapping_dict.items():
        r_mod, r_body = extract_keywords(report)

        match_found = False
        for template in templates:
            t_mod, t_body = extract_keywords(template)
            modality_match = r_mod == t_mod
            body_match = (r_body in t_body) or (t_body in r_body)

            if modality_match and body_match:
                match_found = True
                break  # no need to check the rest

        if not match_found:
            mismatches.append((report, templates, r_mod, r_body))

    percentage = len(mismatches) / total_reports if total_reports > 0 else 0
    return mismatches, percentage

def test_template_mapping(mapping_dict):
    mismatches, percentage = check_mapping(mapping_dict)
    
    for report, template, r_mod, t_mod, r_body, t_body in mismatches:
        print(f"❌ Mismatch found for report '{report}' and template '{template}':")
        print(f"   Modalities: {r_mod} vs {t_mod}")
        print(f"   Body parts: {r_body} vs {t_body}\n")
    
    print(f"Total reports: {len(mapping_dict)}")
    print(f"Total mismatches: {len(mismatches)}")
    print(f"Percentage of mismatches: {percentage:.2%}")