import json
import os
import pandas as pd
from google.cloud import aiplatform
from google import genai
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from ner import *
from rag import *
from graph import build_and_visualize_subgraph

load_dotenv()

aiplatform.init(project="llm-structuring", location="us-central1")
client = genai.Client(vertexai=True, project="llm-structuring", location="us-central1")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
# EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
GEMINI_MODEL_NAME = "gemini-2.0-flash"  
RADLEX_DATABASE_PATH = "data/d_radlex_entities.csv"
MEDICAL_REPORTS_PATH = "../icd_coding/data/PARROT_sample_reports/german_reports.json"

MAX_DOCS = 1
CHROMADB_PATH = "radlex_chroma_db" 
RADLEX_MAPPINGS_PATH = "data/radlex_mappings.json"

REASONING = False # Set to False to skip LLM reasoning - generation part of RAG pipeline
REFORMULATE_ENTITIES = True  # Set to False to skip entity reformulation
BUILD_REPORT_GRAPH = True # Set to False to skip generation of .html with RadLex knowledge graph for each report


def get_embedding_model():
    """Returns the embedding model for generating embeddings."""
    if EMBEDDING_MODEL_NAME.startswith("jinaai/"):
        ef = embedding_functions.JinaEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_MODEL_NAME.startswith("sentence-transformers/"):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    return ef
    

def create_chroma_db(documents, name):
    chroma_client = chromadb.PersistentClient(path=name)

    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_model()
    )

    # Check if the collection already exists
    current_doc_count = db.count()
    num_new_documents = len(documents)

    if current_doc_count < num_new_documents:
        start_index = current_doc_count
        documents_to_add = documents[start_index:]
        print(f"Collection '{name}' contains {current_doc_count} documents. "
              f"Adding {len(documents_to_add)} new documents starting from index {start_index}...")

        # Add documents from the calculated start_index
        for i, d in enumerate(documents_to_add):
            db.add(
                documents=d,
                ids=str(start_index + i)
            )
        print(f"Successfully added {len(documents_to_add)} documents to collection '{name}'.")
    else:
        print(f"Collection '{name}' already contains {current_doc_count} documents, "
              f"which is equal to or more than the {num_new_documents} provided. Skipping document addition.")
        
    return db


def load_radlex_codes(path):
    """Loads RadLex codes and descriptions from a JSON file."""
    print(f"Loading RadLex codes from {path}...")
    radlex_data = pd.read_csv(path)
    print(f"Loaded {len(radlex_data)} RadLex codes.")
    return radlex_data


if __name__ == "__main__":

    with open(MEDICAL_REPORTS_PATH, "r", encoding="utf-8") as f:
        reports = json.load(f)
    reports = reports[:MAX_DOCS]
    #import pdb; pdb.set_trace()
    #if not (os.path.exists(GT_ICD_LABELS_PATH)):
    #    with open(GT_ICD_LABELS_PATH, "w") as f:
    #        for entry in reports:
    #            report_no = entry["no"]
    #            icd_codes = [code.strip() for code in entry["icd"].split(',')]
                
    #            for code in icd_codes:
    #                f.write(f"{report_no}\t{cm.remove_dot(code)}\n")

    radlex_data = load_radlex_codes(RADLEX_DATABASE_PATH)
    descriptions = radlex_data['preferred_description'].tolist()  
    codes = radlex_data['radlex_code'].tolist() 

    db = create_chroma_db(descriptions, CHROMADB_PATH)

    # create or load the mapping between ICD codes and their descriptions
    if not (os.path.exists(RADLEX_MAPPINGS_PATH)):
        id_to_radlex_map = {i: {"code": codes[i], "description": descriptions[i]} for i in range(len(radlex_data))}
        with open(RADLEX_MAPPINGS_PATH, "w") as f:
            json.dump(id_to_radlex_map, f, indent=2)
    else:
        with open(RADLEX_MAPPINGS_PATH, "r") as f:
            id_to_radlex_map = json.load(f)
  
    print("\n--- Processing Medical Reports ---")
    df_results = []

    # for filename in os.listdir(MEDICAL_REPORTS_PATH):
    #     if filename.endswith(".txt"):
    #         report_id = filename.replace(".txt", "")
    #         with open(os.path.join(MEDICAL_REPORTS_PATH, filename), "r") as f:
    #             report_text = f.read()
    #             print(report_text)

    for report in reports:
        report_id = report["no"]
        report_text = report["translation"]

        extracted_entities = extract_entities_llm(report_text, GEMINI_MODEL_NAME, temperature=0.0, max_tokens=500)
        split_outputs = [x for x in extracted_entities.split("\n") if x]

        predicted_codes = []
        predicted_descriptions = []
        similarity_scores = []
        softmax_probs = []

        for phrase in split_outputs:
            print(f"Original phrase: {phrase}")

            candidate_phrases = [phrase]

            if REFORMULATE_ENTITIES:
                try:
                    reformulations = reformulate_entity_synonyms(phrase, GEMINI_MODEL_NAME, num_variations=5)
                    candidate_phrases.extend(reformulations)
                except Exception as e:
                    print(f"Error during reformulation: {e}")

            print(f"Candidate phrases: {candidate_phrases}")

            best_radlex = None
            best_score = float("inf")
            best_description = None
            similarity_debug = None

            for candidate in candidate_phrases:
                if REASONING:
                    closest_radlex_codes = retrieve_closest_radlex_code(candidate, db, id_to_radlex_map, k=3)
                    print(f"Retrieved for '{candidate}': {closest_radlex_codes}")

                    closest_radlex_code = llm_reasoning(GEMINI_MODEL_NAME, closest_radlex_codes, phrase)
                    print(f"Retrieved for '{candidate}': {closest_radlex_codes}")

                else:
                    embedding_function=get_embedding_model()
                    candidate_embedding = embedding_function([candidate])[0]
                    closest_radlex_code = retrieve_closest_radlex_code(candidate_embedding, db, id_to_radlex_map)[0]
                    print(f"Retrieved for '{candidate}': {closest_radlex_code}")

                    #embedding_function=get_embedding_model()
                    #print('EMBEDDING FROM MODEL', embedding_function(["left gastric artery"])[0])

                if closest_radlex_code is not None and closest_radlex_code['similarity_score'] < best_score:
                    best_radlex = closest_radlex_code['radlex_code']
                    best_description = closest_radlex_code['radlex_description']
                    best_score = closest_radlex_code['similarity_score']
                    similarity_debug = closest_radlex_code['similarity_debug']
                    best_softmax = closest_radlex_code['softmax_retrieval']

            if best_radlex is None:
                print(f"No suitable RadLex code found for: {phrase}")
                continue

            predicted_codes.append(best_radlex)
            predicted_descriptions.append(best_description)
            similarity_scores.append(best_score)
            softmax_probs.append(best_softmax)

            print(f"Best RadLex code for '{phrase}': {best_radlex} - {best_description} (Similarity: {best_score:.4f}, Probability: {best_softmax:.4f})")

        # Save visualization HTML per report
        if predicted_codes and BUILD_REPORT_GRAPH:
            output_html_path = f"data/output/graph_{report_id}.html"
            try:
                build_and_visualize_subgraph(
                    codes=predicted_codes,
                    root='RID1',  # or any other root as needed
                    output_path=output_html_path
                )
                print(f"Saved graph visualization for report {report_id} to {output_html_path}")
            except Exception as e:
                print(f"Error generating graph for report {report_id}: {e}")
        else:
            print(f"Skipping graph generation.")

        df_results.append({
            "report_id": report_id,
            "predicted_phrases": split_outputs,
            "RadLex codes": predicted_codes,
            'RadLex descriptions': predicted_descriptions,
            "similarity_scores": similarity_scores,
            "softmax_probs": softmax_probs
        })

    df = pd.DataFrame(df_results)
    df.to_csv("data/output/radlex_predictions.csv", index=False)
    flat_results = []

    for result in df_results:
        report_id = result["report_id"]
        codes = result["RadLex codes"]
        for code in codes:
            flat_results.append({"report_id": report_id, "RadLex code": code})

    df_flat = pd.DataFrame(flat_results)
    df_flat.to_csv("data/output/radlex_predictions.tsv", index=False, sep="\t", header=False)