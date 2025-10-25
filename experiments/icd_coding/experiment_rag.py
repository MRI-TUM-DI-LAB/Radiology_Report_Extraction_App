import json
import os
import pandas as pd
from google.cloud import aiplatform
from google import genai
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import simple_icd_10_cm as cm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch


from ner import *
from rag import *
from evaluation import run_evaluation

load_dotenv()

aiplatform.init(project="llm-structuring", location="us-central1")
client = genai.Client(vertexai=True, project="llm-structuring", location="us-central1")

DATASET = 'parrot' # Set to 'parrot' or 'codiesp' 
LANG = 'en' # Set to 'en' or any other language prefix -> this only determines the name of the ouput directory

####### CHOOSE EMBEDDING MODEL FOR EXPERIMENTS #########
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3" # or "sentence-transformers/all-MiniLM-L6-v2"
########################################################

GEMINI_MODEL_NAME = "gemini-2.0-flash" # LLM model name
ICD_DATABASE_PATH = "data/d_icd_diagnoses.csv"
MEDICAL_REPORTS_PATH = "data/medical_reports"
MAX_DOCS = 20 # how many medical reoprts to load for experiments
ICD_MAPPINGS_PATH = "data/icd_mappings.json" 
GT_ICD_LABELS_PATH = 'data/gt_codes'
REASONING = False # Set to False to skip LLM reasoning - generation part of RAG pipeline
REFORMULATE_ENTITIES = True  # Set to False to skip entity reformulation
THRESHOLDING = True  # Set to False to skip LLM thresholding
LLM_TEMPERATURE = 0.0 # Set the temperature for the LLM


# custom embedding function for Jina embeddings
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {self.device}")
        self.model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=self.device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, device=self.device)

# depending on embedding model name, we return the appropriate embedding function
def get_embedding_model():
    """Returns the embedding model for generating embeddings."""
    if EMBEDDING_MODEL_NAME.startswith("jinaai/"):
        ef = MyEmbeddingFunction()
        # ef = embedding_functions.JinaEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_MODEL_NAME.startswith("sentence-transformers/"):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    
    return ef

# creates a ChromaDB collection if the collection does not exist, adds documents to the existing collection or fretches the existing collection if no new docuements are provided
def create_chroma_db(documents):

    if EMBEDDING_MODEL_NAME.startswith("jinaai/"):
        name = 'jina_chroma_db'
    else:
        name = 'icd_chroma_db'

    chroma_client = chromadb.PersistentClient(path=name) 

    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_model(),
        configuration={"hnsw": {"space": "cosine"}}
    )
    print("Number of documents:", db.count())
    # Check if the collection already exists
    current_doc_count = db.count()
    num_new_documents = len(documents)

    if current_doc_count < num_new_documents:
        start_index = current_doc_count
        documents_to_add = documents[start_index:]
        ids_to_add = [str(start_index + i) for i in range(len(documents_to_add))]
        print(f"Collection '{name}' contains {current_doc_count} documents. "
              f"Adding {len(documents_to_add)} new documents starting from index {start_index}...")

        batch_size = 500 # Adjust this based on your system's memory and performance -> speeds up embedding computations especially with Jina
        for i in tqdm(range(0, len(documents_to_add), batch_size), desc="Adding documents in batches"):
            batch_docs = documents_to_add[i : i + batch_size]
            batch_ids = ids_to_add[i : i + batch_size]
            
            db.add(
                documents=batch_docs,
                ids=batch_ids
            )
        print(f"Successfully added {len(documents_to_add)} documents to collection '{name}'.")
    else:
        print(f"Collection '{name}' already contains {current_doc_count} documents, "
              f"which is equal to or more than the {num_new_documents} provided. Skipping document addition.")
        
    return db


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

    # create or load the ChromaDB collection with ICD codes
    db = create_chroma_db(descriptions)

    # create or load the mapping between ICD codes and their descriptions
    if not (os.path.exists(ICD_MAPPINGS_PATH)):
        id_to_icd_map = {i: {"code": codes[i], "description": descriptions[i]} for i in range(len(icd_data))}
        with open(ICD_MAPPINGS_PATH, "w") as f:
            json.dump(id_to_icd_map, f, indent=2)
    else:
        with open(ICD_MAPPINGS_PATH, "r") as f:
            id_to_icd_map = json.load(f)

    # create output file path
    output_file = f'data/output/experiment_rag/{DATASET}/{LANG}/icd_predictions.json'
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

        # extract medical entities from the report text using LLM
        extracted_entities = extract_entities_llm(report_text, GEMINI_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=500)
        split_outputs = [x for x in extracted_entities.split("\n") if x]
        # print(f"Extracted entities for report {report_id}: {split_outputs}")

        chosen_reformulations = []
        predicted_codes = []
        predicted_descriptions = []
        similarity_scores = []

        for phrase in split_outputs:

            candidate_phrases = [phrase]

            # If reformulation is enabled, generate 5 variations of the entity
            if REFORMULATE_ENTITIES:
                try:
                    reformulations = reformulate_entity_synonyms(phrase, GEMINI_MODEL_NAME, num_variations=5, temperature=LLM_TEMPERATURE)
                    candidate_phrases.extend(reformulations)
                except Exception as e:
                    print(f"Error during reformulation: {e}")

            best_icd = None
            best_score = float("inf")
            best_description = None

            for candidate in candidate_phrases:
                # If reasoning is enabled, retrieve the closest 3 ICD codes from the embedding database
                if REASONING:
                    closest_icds = retrieve_closest_icd_code(candidate, db, id_to_icd_map, k=3)
                    # use LLM to choose the best ICD code from the closest 3
                    closest_icd = llm_reasoning(GEMINI_MODEL_NAME, closest_icds, phrase, temperature=LLM_TEMPERATURE)

                else:
                    # Retrieve the closest ICD code from the embedding database
                    closest_icd = retrieve_closest_icd_code(candidate, db, id_to_icd_map)[0]

                # update the best ICD code for the prediction if the similarity score of the reformulation is better than the current best
                if closest_icd is not None and closest_icd['similarity_score'] < best_score:
                    best_icd = closest_icd['icd_code']
                    best_description = closest_icd['icd_description']
                    best_score = closest_icd['similarity_score']
                    chosen_reformulation = candidate

            if best_icd is None:
                continue

            chosen_reformulations.append(chosen_reformulation)
            predicted_codes.append(best_icd)
            predicted_descriptions.append(best_description)
            similarity_scores.append(best_score)

        # If thresholding is enabled, apply LLM reasoning to the predicted descriptions
        if THRESHOLDING:
            final_descriptions = llm_thresholding(GEMINI_MODEL_NAME, predicted_descriptions, report_text, temperature=LLM_TEMPERATURE)
            removed_indices = [i for i, desc in enumerate(predicted_descriptions) if desc not in final_descriptions]

            predicted_descriptions = final_descriptions
            predicted_codes = [code for i, code in enumerate(predicted_codes) if i not in removed_indices]
            similarity_scores = [score for i, score in enumerate(similarity_scores) if i not in removed_indices]

        code_map[report_id] = {
                "report_id": report_id,
                "predicted_phrases": split_outputs,
                'chosen_reformulations': chosen_reformulations,
                "ICD-10 codes": predicted_codes,
                'ICD-10 descriptions': predicted_descriptions,
                "similarity_scores": similarity_scores
            }
        
        # Save the ICD predictions to a JSON file
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
    run_evaluation(DATASET, 'rag', LANG)