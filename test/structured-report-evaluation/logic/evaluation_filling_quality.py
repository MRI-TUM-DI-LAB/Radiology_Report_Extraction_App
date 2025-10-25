import json
from sentence_transformers import SentenceTransformer, util
import torch
from report_structuring import ReportStructuringProcessor
from template_selection_test import test_template_mapping
import os
from tqdm import tqdm

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def legal_json(file_path):
    """Check if a JSON file is well-formed."""
    try:
        with open(file_path, 'r') as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        print(f"JSON format is not legal: {e}")
        return False
    
def json_to_flat_text(data, parent_key=""):
    lines = []

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and k.lower() in {"findings", "impression", "diagnosis", "conclusion"}:
                section_title = k.capitalize() + ":"
                lines.append(section_title)
                lines.extend(json_to_flat_text(v))
            else:
                lines.extend(json_to_flat_text(v, k))

    elif isinstance(data, list):
        for item in data:
            lines.extend(json_to_flat_text(item))

    else:
        if parent_key:
            key_text = parent_key.replace('_', ' ').capitalize()
            if isinstance(data, str) and data.strip().lower().startswith(key_text.lower()):
                lines.append(data.strip())  # Already starts with key_text
            else:
                lines.append(f"{key_text} is {data}.")

    return lines


def compute_similarity(freetext: str, structured_json: dict) -> float:
    """
    Compute the semantic similarity between free text and structured JSON report.
    """
    # Convert structured JSON to flat text
    flatten_generated = " ".join(json_to_flat_text(structured_json))
    print(f"Free text: {freetext[:500]}...")  # Print first 500 characters for debugging
    print(f"Flattened generated report: {flatten_generated[:500]}...")  # Print first 500 characters for debugging

    # Encode both texts 
    emb1 = model.encode(freetext, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = model.encode(flatten_generated, convert_to_tensor=True, normalize_embeddings=True)


    # Compute cosine similarity
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity

if __name__ == "__main__":

 
    input_folder = '../free_text'
    output_folder = '../output_sequential'

    os.makedirs(output_folder, exist_ok=True)
    
    processor = ReportStructuringProcessor()
    template_assignment = {}
    
    # Generate structured reports from free text files
    for filename in tqdm(os.listdir(input_folder)):
        # check if it is already processed
        if filename.endswith(".txt"):
            print(f"Processing {filename}...")
            if os.path.exists(os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")):
                print(f"Skipping {filename}, already processed.")
                continue
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    freetext = f.read()
                    structured_json, template_name = processor.process_freetext(freetext)
                    template_assignment[filename] = template_name

                # Save to individual JSON file
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(structured_json, out_f, indent=2, ensure_ascii=False)
                print(f"✅ Processed {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
            print("------------------------------")

    print(template_assignment)

    # Evaluate the generated structured reports    
    n_total = [f for f in os.listdir(output_folder) if f.endswith('.json')]

    count_legal_json = 0
    similarity_scores = []
    similarity_store = {}

    for filename in tqdm(os.listdir(output_folder)):
        if filename.endswith(".json"):

            generated_file = os.path.join(output_folder, filename)
            generated_report = load_json(generated_file)
            print(f"\nEvaluating {filename}...")

            # 1. Check if the generated JSON files are well-formed
            if not legal_json(generated_file):
                print("The generated report has illegal JSON format.")
                continue
            else:
                count_legal_json += 1

            # 2. Test template selection

            # 3. Test the filling quality
            freetext_file = os.path.join(input_folder, filename.replace('.json', '.txt'))
            with open(freetext_file, 'r', encoding='utf-8') as f:
                freetext = f.read()
            similarity = compute_similarity(freetext, generated_report)
            # Store similarity score
            similarity_store[filename] = similarity
            similarity_scores.append(similarity)
            print(f"Semantic similarity: {similarity:.4f}")

    print(f"Total legal JSON files: {count_legal_json}/{len(n_total)}")
    print(f"Average semantic similarity: {sum(similarity_scores) / len(similarity_scores):.4f}")
    test_template_mapping(template_assignment)

    # Save similarity scores to a JSON file
    similarity_output_path = os.path.join(output_folder, "similarity_sequential.json")
    with open(similarity_output_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_store, f, indent=2, ensure_ascii=False)
    print(f"Similarity scores saved to {similarity_output_path}")

            