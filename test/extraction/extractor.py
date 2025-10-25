# extractor.py

import requests

# 1) Ollama server location & model
OLLAMA_URL = "http://ollama:11434"
MODEL      = "medllama2:latest"

# 2) Seven in-domain few-shot examples drawn from PARROT (with placeholders for modality/area)
FEW_SHOT = [
    # MR • head (no. 275)
    {
        "report": (
            "MRI of the brain, pre- and post-contrast, following the standard protocol:\n\n"
            "A previous MRI from an external facility dated 16.10.2023 is available for comparison.\n\n"
            "The organization of the gyri and sulci is appropriate, and the gray-white matter differentiation is preserved.\n\n"
            "There are no signs of acute ischemia, hemorrhage, or expansive processes in the brain parenchyma.\n\n"
            "In the supratentorial deep white matter, nonspecific T2/FLAIR hyperintense lesions are visible, likely corresponding to chronic microangiopathic changes - Fazekas grade 1.\n\n"
            "No areas of diffusion restriction are observed.\n\n"
            "The ventricular system is midline, symmetrical, and of appropriate width.\n\n"
            "No extra-axial collections are present.\n\n"
            "The appearance of the optic chiasm, cavernous sinuses, cerebellopontine angles, and the craniocervical junction is normal.\n\n"
            "No areas of pathological contrast enhancement are observed on post-contrast T1 images."
        ),
        "phrase": "No acute ischemia, hemorrhage, or mass effect with chronic microangiopathic T2/FLAIR lesions ({modality} of the {area})"
    },
    # RX • abdomen (no. 604)
    {
        "report": (
            "No free air. Several small bowel loops are significantly dilated, possibly indicating a mechanical obstruction. A CT scan is recommended."
        ),
        "phrase": "Significant small bowel dilation suggesting mechanical obstruction ({modality} of the {area})"
    },
    # US • chest (no. 1580)
    {
        "report": (
            "LV IVSd 1.2 cm, LVIDd 5.3 cm, LVIDs 3.9 cm, LVPWd 0.9 cm; LV mass 220 g, LV mass index 104 g/m², LVEF 60 %; RV RVOT 3.0 cm, RVIDd 3.6 cm; IVC insp 9.4 mm, exp 21.1 mm; LA 4.9 cm, LA volume 93 ml (44 ml/m²); RA area 14.5 cm². Valves: mitral E 0.73 m/s E/A 0.80, decel 243 ms; aortic Vmax 1.90 m/s; trace regurgitations in mitral, tricuspid, pulmonary; no pericardial fluid. Segmental LV hypokinesia in basal inferoseptal segment."
        ),
        "phrase": "Segmental LV hypokinesia with diastolic dysfunction and LA enlargement ({modality} of the {area})"
    },
    # CT • head (no. 4)
    {
        "report": (
            "Irregular saccular aneurysmal dilation at the origin of the left PICA in the vertebral artery, with subarachnoid hemorrhage and intraventricular extension."
        ),
        "phrase": "Saccular aneurysmal dilation at left PICA with subarachnoid hemorrhage ({modality} of the {area})"
    },
    # OPH • orbit (no. 283)
    {
        "report": (
            "The examination conducted using radial linear scans through the fovea (Radial Line) and 3D reconstruction (Retina Map) revealed: "
            "The internal profile of the retina appears regular with normal foveal depression in both eyes. All retinal layers, especially the ellipsoid "
            "zone and external limiting membrane, are clearly identifiable. No abnormal fluids. Retinal pigment epithelium and choroid are normal for age. "
            "Retinal thickness is within normal limits."
        ),
        "phrase": "Normal retinal architecture with preserved foveal depression and no abnormal fluids ({modality} of the {area})"
    },
    # PET/CT • chest, abdomen, pelvis (no. 1648)
    {
        "report": (
            "A PET/CT scan of the chest and abdomen was performed emergently for malignancy assessment. Increased FDG uptake in the left lung suggests a tumor; "
            "FDG-avid masses in the liver (5.0 × 4.0 × 3.5 cm) and pancreas; enlarged lymph nodes at the pulmonary hilum and along the aorta, consistent with metastases; "
            "no other pathological abdominal findings."
        ),
        "phrase": "Disseminated FDG-avid malignancy in lung, liver, pancreas and nodal metastases ({modality} of the {area})"
    },
    # ENDOSCOPY • abdomen (no. 1529)
    {
        "report": (
            "Endoscope introduced under direct vision: numerous raised fungal plaques in the lower esophagus; cardia and subcardial areas unremarkable; "
            "gastric mucosa hyperemic and slightly edematous; normal pylorus with positive urease test; duodenal bulb and descending duodenum intact. "
            "Diagnosis: erythematous gastropathy and Grade II esophageal candidiasis."
        ),
        "phrase": "Grade II esophageal candidiasis with erythematous gastropathy ({modality} of the {area})"
    }
]

def build_messages(full_text: str, modality: str, area: str) -> list:
    """
    Build a chat-style prompt:
      1) system instruction (with modality/area placeholders)
      2) all FEW_SHOT examples (injecting modality/area)
      3) the actual report text
      4) final 'Search Phrase:' cue
    """
    system = {
        "role": "system",
        "content": (
            "You are a radiology assistant.  "
            "I will paste a JSON record whose `translation` field contains the English text of a radiology report.  "
            "From that `translation`, extract exactly one short sentence capturing the single most important finding, "
            "then append `({modality} of the {area})` with no extra words or lines."
        )
    }

    msgs = [system]
    for ex in FEW_SHOT:
        example = ex["phrase"].format(modality=modality, area=area)
        msgs.append({"role": "user",      "content": f"Report:\n{ex['report']}"})
        msgs.append({"role": "assistant", "content": example})

    msgs.append({"role": "user", "content": f"Report:\n{full_text}"})
    msgs.append({"role": "user", "content": "Search Phrase:"})
    return msgs

def extract_phrase(full_text: str, modality: str, area: str) -> str:
    """
    Call Ollama v1 chat endpoint with a low temperature for determinism,
    then return the first line of the assistant’s reply.
    """
    messages = build_messages(full_text, modality, area)
    resp = requests.post(
        f"{OLLAMA_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.2
        }
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # only keep the first line as the “search phrase”
    return content.splitlines()[0]


# # extractor.py

# import requests

# # 1) Ollama server location & model
# OLLAMA_URL = "http://ollama:11434"
# MODEL      = "medllama2:latest"

# # 2) Four representative few-shot examples spanning MR-head, RX-abdomen, US-chest, CT-head
# FEW_SHOT = [
#     # MR • head (no. 275)
#     {
#         "report": (
#             "MRI of the brain, pre- and post-contrast, following the standard protocol:\n\n"
#             "A previous MRI from an external facility dated 16.10.2023 is available for comparison.\n\n"
#             "The organization of the gyri and sulci is appropriate, and the gray-white matter differentiation is preserved.\n\n"
#             "There are no signs of acute ischemia, hemorrhage, or expansive processes in the brain parenchyma.\n\n"
#             "In the supratentorial deep white matter, nonspecific T2/FLAIR hyperintense lesions are visible, likely corresponding to chronic microangiopathic changes - Fazekas grade 1.\n\n"
#             "No areas of diffusion restriction are observed.\n\n"
#             "The ventricular system is midline, symmetrical, and of appropriate width.\n\n"
#             "No extra-axial collections are present.\n\n"
#             "The appearance of the optic chiasm, cavernous sinuses, cerebellopontine angles, and the craniocervical junction is normal.\n\n"
#             "No areas of pathological contrast enhancement are observed on post-contrast T1 images."
#         ),
#         "phrase": "No acute ischemia, hemorrhage, or mass effect with chronic microangiopathic T2/FLAIR lesions ({modality} of the {area})"
#     },
#     # RX • abdomen (no. 604)
#     {
#         "report": (
#             "No free air. Several small bowel loops are significantly dilated, possibly indicating a mechanical obstruction. A CT scan is recommended."
#         ),
#         "phrase": "Significant small bowel dilation suggesting mechanical obstruction ({modality} of the {area})"
#     },
#     # US • chest (no. 1580)
#     {
#         "report": (
#             "LV IVSd 1.2 cm LVIDd 5.3 cm LVIDs 3.9 cm LVPWd 0.9 cm LV mass (ASE) 220 g LV mass index (ASE) 104 g/m2 LVEF (MOD) 60% RV RVOT diameter 3.0 cm RVIDd 3.6 cm IVC inspiratory 9.4 mm IVC expiratory 21.1 mm LA / LAA Mean left atrial diameter 4.9 cm Left atrial volume 93 ml Left atrial volume index 44 ml/m2 RA RAA (d) 14.5 cm2 Mitral valve E wave velocity 0.73 m/s E/A mitral valve 0.80 Deceleration time mitral valve 243 ms E' septal MV 5.5 cm/s E' lateral mitral valve 5.7 cm/s E/E' lateral mitral valve 12.7 Mean E/E' mitral valve 12.9 Aortic valve Vmax aortic valve 1.90 m/s Maximum pressure gradient aortic valve 14.49 mmHg Aorta Mean aortic root diameter 4.1 cm Mean ascending aorta diameter 3.7 cm Pulmonary valve MPA 2.3 cm Vmax in PV 0.76 m/s Acceleration time pulmonary valve 136 ms Mitral valve: Leaflet edges intensely thickened. Regurgitation present. Aortic valve: Leaflets are intensely thickened with preserved mobility. No regurgitation present. Tricuspid valve: Normal; gradient not delineated. Trace regurgitation. Pulmonary valve: Normal. Trace regurgitation. Pericardium: No fluid. Left ventricle: Segmental systolic function: Wall contractility: hypokinesis of the lower septal segment. Summary: Left ventricle is not enlarged with segmental contractility disturbances mentioned above with a normal ejection fraction. Grade 1 diastolic dysfunction. Enlargement of the left atrium. Mild mitral regurgitation. Right ventricle is not enlarged with normal systolic function. Aortic bulb enlargement. RVSP not assessable."
#         ),
#         "phrase": "Segmental LV hypokinesia with grade 1 diastolic dysfunction and LA enlargement ({modality} of the {area})"
#     },
#     # CT • head (no. 4)
#     {
#         "report": (
#             "Irregular saccular aneurysmal dilation at the origin of the left PICA in the vertebral artery, with subarachnoid hemorrhage and intraventricular extension."
#         ),
#         "phrase": "Saccular aneurysmal dilation at left PICA with subarachnoid hemorrhage ({modality} of the {area})"
#     }
# ]

# def build_messages(full_text: str, modality: str, area: str) -> list:
#     """
#     Build a chat-style prompt with:
#       1) System instruction
#       2) Few-shot examples (modality/area injected)
#       3) The actual report text
#       4) A final "Search Phrase:" cue
#     """
#     system = {
#         "role": "system",
#         "content": (
#             "You are a radiology assistant.  "
#             "I will provide the full English translation of a radiology report.  "
#             "Extract exactly one short sentence capturing the most important finding, "
#             "then append “({modality} of the {area})” with no extra words or lines."
#         )
#     }
#     msgs = [system]
#     for ex in FEW_SHOT:
#         example_phrase = ex["phrase"].format(modality=modality, area=area)
#         msgs.append({"role": "user",      "content": f"Report:\n{ex['report']}"})
#         msgs.append({"role": "assistant", "content": example_phrase})
#     msgs.append({"role": "user", "content": f"Report:\n{full_text}"})
#     msgs.append({"role": "user", "content": "Search Phrase:"})
#     return msgs

# def extract_phrase(full_text: str, modality: str, area: str) -> str:
#     """
#     Send the prompt to MedLlama2 and return the first line
#     of the assistant’s reply (the search phrase).
#     """
#     messages = build_messages(full_text, modality, area)
#     resp = requests.post(
#         f"{OLLAMA_URL}/v1/chat/completions",
#         json={"model": MODEL, "messages": messages, "max_tokens": 50, "temperature": 0.2}
#     )
#     resp.raise_for_status()
#     content = resp.json()["choices"][0]["message"]["content"].strip()
#     return content.splitlines()[0]

# -----
# # extractor.py

# import re
# import requests

# # 1) Configuration
# OLLAMA_URL = "http://ollama:11434"   # inside Docker, 'ollama' must match your service name
# MODEL      = "medllama2:latest"

# # 2) Few-shot examples with format placeholders
# FEW_SHOT = [
#     {
#         "report": "No evidence of acute intracranial hemorrhage or mass effect. Follow-up recommended in 6 months.",
#         "phrase": "No acute intracranial hemorrhage or mass effect ({modality} of the {area})"
#     },
#     {
#         "report": "There is a 5 mm enhancing lesion in the right frontal lobe suggestive of metastasis.",
#         "phrase": "5 mm enhancing lesion in the right frontal lobe suggestive of metastasis ({modality} of the {area})"
#     }
# ]

# def extract_conclusion(text: str) -> str:
#     """
#     Pull out the 'Impression', 'Conclusion' or 'Imaging Comment' block if present;
#     otherwise return the last two sentences.
#     """
#     pattern = (
#         r"(?mi)^(Impression|Conclusion|Imaging Comment)\s*[:\-–]\s*"
#         r"(.+?)(?=\n^(?:[A-Z][a-z]+|$)[^:]*[:\-–]|\Z)"
#     )
#     m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
#     if m:
#         return m.group(2).strip()
#     # fallback to last two sentences
#     parts = re.split(r'(?<=[.!?])\s+', text.strip())
#     return ' '.join(parts[-2:]).strip()

# def build_messages(conc_text: str, modality: str, area: str) -> list:
#     """
#     Build a chat-style prompt: one system instruction, two few-shots (with injected
#     modality/area), then the real query.
#     """
#     msgs = [{
#         "role": "system",
#         "content": (
#             "You are a radiology assistant.\n"
#             "I will give you a free-text report and two examples of how to pick exactly "
#             "one short sentence capturing the most important finding—including its "
#             "modality and area in parentheses.\n"
#             "Do exactly the same for the new report."
#         )
#     }]
#     # inject placeholders for each example
#     for ex in FEW_SHOT:
#         example = ex["phrase"].format(modality=modality, area=area)
#         msgs.append({"role": "user",      "content": f"Report:\n{ex['report']}"})
#         msgs.append({"role": "assistant", "content": example})
#     # now the real query
#     msgs.append({"role": "user", "content": f"Report:\n{conc_text}"})
#     msgs.append({"role": "user", "content": "Search Phrase:"})
#     return msgs

# def post_validate(raw: str) -> str:
#     """
#     Keep only the first line, trim extra sentences.
#     """
#     line = raw.strip().splitlines()[0]
#     if line.count('.') > 1:
#         line = line.split('.', 1)[0].strip() + '.'
#     return line

# def extract_phrase(text: str, modality: str, area: str) -> str:
#     """
#     Full workflow: conclusion → prompt build → Ollama call → clean up.
#     """
#     conc     = extract_conclusion(text)
#     messages = build_messages(conc, modality, area)
#     resp     = requests.post(
#         f"{OLLAMA_URL}/v1/chat/completions",
#         json={
#             "model":        MODEL,
#             "messages":     messages,
#             "max_tokens":   50,
#             "temperature":  0.0
#         }
#     )
#     resp.raise_for_status()
#     raw = resp.json()["choices"][0]["message"]["content"]
#     return post_validate(raw)
