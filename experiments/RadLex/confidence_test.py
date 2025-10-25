import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Load environment variables
load_dotenv()

# Set project and location
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "llm-structuring")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_ID = "gemini-2.0-flash"

# Initialize client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Define prompt
text = """Clinical Information, Question, Justification: Right arm paresis, aphasia, slowness, unclear time window. M1 occlusion on the left. Recanalization requested. Consent and Explanation: Emergency indication. Due to the patient's clinical condition, it was not possible to obtain a risk history, explanation, or consent. [...]"""
task = f"""
You are an expert medical assistant. Your task is to extract all confirmed diseases, medical conditions, and relevant health factors that the patient *does* have, based on the following unstructured medical report.

Focus your extraction ONLY on the following categories:
- Diagnosed diseases and conditions
- Clinical signs and symptoms
- Abnormal findings
- Injuries and external causes
- Social and structural health factors
- Interactions with the healthcare system

Instructions:
- Extract only conditions explicitly confirmed by the report.
- Ignore negated conditions.
- Return the **exact phrase** from the report confirming it.
- Add ", unspecified" if clinical detail is lacking.
- Output one phrase per line.

Medical report:
{text}
"""

# Query the model with logprobs enabled
response = client.models.generate_content(
    model=MODEL_ID,
    contents=task,
    config=GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
        response_logprobs=True,
        logprobs=5
    ),
)

print(response.text)

# Helper to print logprobs
def print_logprobs(response):
    if response.candidates and response.candidates[0].logprobs_result:
        logprobs_result = response.candidates[0].logprobs_result
        for i, chosen_candidate in enumerate(logprobs_result.chosen_candidates):
            print(f"Token: '{chosen_candidate.token}' ({chosen_candidate.log_probability:.4f})")
            if i < len(logprobs_result.top_candidates):
                top_alternatives = logprobs_result.top_candidates[i].candidates
                for alt in top_alternatives:
                    if alt.token != chosen_candidate.token:
                        print(f"  - Alternative: '{alt.token}' ({alt.log_probability:.4f})")
            print("-" * 20)

# Print token-level logprobs
print_logprobs(response)
