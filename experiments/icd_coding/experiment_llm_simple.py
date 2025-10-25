import json
import os
import pandas as pd
from google.cloud import aiplatform
from google import genai
from dotenv import load_dotenv
import simple_icd_10_cm as cm
from tqdm import tqdm

from ner import *
from rag import *
from llm import *
from evaluation import run_evaluation

load_dotenv()

aiplatform.init(project="llm-structuring", location="us-central1")
client = genai.Client(vertexai=True, project="llm-structuring", location="us-central1")

DATASET = 'parrot' # Set to 'parrot' or 'codiesp' 
LANG = 'de' # Set to 'en' or 'de' -> this determines the name of the ouput directory
GEMINI_MODEL_NAME = "gemini-2.0-flash" # LLM model name
ICD_DATABASE_PATH = "data/d_icd_diagnoses.csv"
MEDICAL_REPORTS_PATH = "data/medical_reports"
MAX_DOCS = 20 # how many medical reoprts to load for experiments
ICD_MAPPINGS_PATH = "data/icd_mappings.json" 
GT_ICD_LABELS_PATH = 'data/gt_codes'
LLM_TEMPERATURE = 0.0 # Set the temperature for the LLM


def load_icd_codes(path):
    """Loads ICD codes and descriptions from a CSV file."""
    print(f"Loading ICD codes from {path}...")
    icd_data = pd.read_csv(path)
    icd_data = icd_data[icd_data['icd_version'] == 10]
    print(f"Loaded {len(icd_data)} ICD codes.")
    return icd_data


if __name__ == "__main__":
    # load the medical reports from the JSON file
    with open(os.path.join(MEDICAL_REPORTS_PATH, f'{DATASET}.json'), "r", encoding="utf-8") as f:
        reports = json.load(f)
    reports = reports[:MAX_DOCS]

    # Create the ground truth ICD labels TSV file if it does not exist
    if not (os.path.exists(os.path.join(GT_ICD_LABELS_PATH, f'{DATASET}.tsv'))):
        with open(os.path.join(GT_ICD_LABELS_PATH, f'{DATASET}.tsv'), "w") as f:
            for entry in reports:
                report_no = entry["no"]
                icd_codes = [code.strip() for code in entry["icd"].split(',')]
                
                for code in icd_codes:
                    try:
                        clean_code = cm.remove_dot(code)
                        f.write(f"{report_no}\t{clean_code}\n")
                    except ValueError as e:
                        print(f"Skipping invalid code '{code}': {e}")
                        continue

    icd_data = load_icd_codes(ICD_DATABASE_PATH)
    descriptions = icd_data['long_title'].tolist()  
    codes = icd_data['icd_code'].tolist() 

    # create or load the mapping between ICD codes and their descriptions
    if not (os.path.exists(ICD_MAPPINGS_PATH)):
        id_to_icd_map = {i: {"code": codes[i], "description": descriptions[i]} for i in range(len(icd_data))}
        with open(ICD_MAPPINGS_PATH, "w") as f:
            json.dump(id_to_icd_map, f, indent=2)
    else:
        with open(ICD_MAPPINGS_PATH, "r") as f:
            id_to_icd_map = json.load(f)

    # create output file path
    output_file = f'data/output/experiment_llm_simple/{DATASET}/{LANG}/icd_predictions.json'
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load results from already processed reports
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            code_map = json.load(f)
    else:
        code_map = {}
    
    # Process medical reports
    for report in tqdm(reports, desc="Processing medical reports"):
        report_id = report["no"]
        if report_id in code_map.keys():
            print(f'skipping report {report_id} because already processed')
            continue # Skip if report already processed
        if LANG == 'en':
            report_text = report["translation"]
        else:
            report_text = report["report"]

        # Predict ICD codes using the LLM
        pred_codes = extract_icd(report_text, GEMINI_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=500)
        try:
            split_outputs = [cm.remove_dot(x) for x in pred_codes.split("\n") if x]
        except ValueError as e:
            print(f"Skipping invalid code: {e}")
            continue

        predicted_descriptions = []
        for code in split_outputs:
            result = next((entry for entry in id_to_icd_map.values() if entry["code"] == code), None)
            if result:
                predicted_descriptions.append(result['description'])
            else:
                print(f"Code {code} not found.")


        code_map[report_id] = {
            "ICD-10 codes": split_outputs,
            'ICD-10 descriptions': predicted_descriptions,
        }

        # Save the predicted ICD codes and descriptions to a JSON file
        with open(output_file, "w") as f:
            json.dump(code_map, f, indent=4)

        # Save predictions to TSV file for evaluation
        file_tsv_predictions = []
        for key, value in code_map.items():

            if value == []:
                file_tsv_predictions.append([key, ""])

            else:
                value = value.get("ICD-10 codes")
                for code in value:
                    file_tsv_predictions.append([key, cm.remove_dot(code)])

        df_pred = pd.DataFrame(file_tsv_predictions)
        df_pred.to_csv(output_file.replace(".json", ".tsv"), sep="\t", index=False, header=False)

    # Run evaluation
    run_evaluation(DATASET, 'llm_simple', LANG)