import json
import os
import pandas as pd
import chromadb
import logging
from tqdm import tqdm
import numpy as np
from logic.icd_coding.icd_files.ner import *
from logic.icd_coding.icd_files.rag import *
from config import CONFIG
from shared.embedding_model import embedding_model
logger = logging.getLogger('root')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ICDInference:
    def __init__(self):

        # Configuration parameters
        self.max_docs = CONFIG['icd_coding']['max_docs']
        self.chromadb_path = CONFIG['icd_coding']['chromadb_path']
        self.reasoning = CONFIG['icd_coding']['reasoning']
        self.reformulate_entities = CONFIG['icd_coding']['reformulate_entities']
        self.batch_size = CONFIG['icd_coding']['batch_size']
        self.chromadb_collection_name = CONFIG['icd_coding']['chromadb_collection_name']

        # Paths to data files
        self.icd_database_path = os.path.join(BASE_DIR, "data/d_icd_diagnoses.csv")
        self.icd_mappings_path = os.path.join(BASE_DIR, "data/icd_mappings.json")
  
    def create_chroma_db(self, documents):
        logger.info("Running create_chroma_db()")
        chroma_client = chromadb.PersistentClient(path=self.chromadb_path)

        db = chroma_client.get_or_create_collection(
            name=self.chromadb_collection_name,
            configuration={"hnsw": {"space": "cosine"}}
        )

        # Check if the collection already exists
        current_doc_count = db.count()
        num_new_documents = len(documents)
 
        
        if current_doc_count < num_new_documents:
            logger.info(f"Adding {num_new_documents} new documents to the ChromaDB collection '{self.chromadb_collection_name}'")
            for i in tqdm(range(0, num_new_documents, self.batch_size), desc="Encoding and adding to DB"):
                batch_docs = documents[i:i + self.batch_size]
                embeddings = embedding_model.encode(
                    batch_docs,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ).tolist()

                db.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    ids=[str(j) for j in range(current_doc_count + i, current_doc_count + i + len(batch_docs))]
                )
            logger.info(f"Successfully added {len(documents)} documents to collection '{self.chromadb_collection_name}'.")
        else:
            logger.info(f"Collection '{self.chromadb_collection_name}' already contains {current_doc_count} documents, "
                        f"which is equal to or more than the {num_new_documents} provided. Skipping document addition.")
        
        return db

    def load_icd_codes(self, path):

        """Loads ICD codes and descriptions from a JSON file."""
        
        logger.info(f"Loading ICD codes from {path}...")
        icd_data = pd.read_csv(path)
        icd_data = icd_data[icd_data['icd_version'] == 10]
        logger.info(f"Loaded {len(icd_data)} ICD codes.")
        return icd_data
    
    def predict_icd(self, freetext):
        """
        Define as a function to integrate with the app interface
        """
        icd_data = self.load_icd_codes(self.icd_database_path)
        descriptions = icd_data['long_title'].tolist()  
        codes = icd_data['icd_code'].tolist() 

        db = self.create_chroma_db(
            documents=descriptions
        )
        '''entries = db.get(include=["embeddings", "metadatas"])
        db_embs = np.array(entries["embeddings"])  # shape: (N, D)
        db_ids = entries["ids"]'''

        # Load ICD mappings
        if not os.path.exists(self.icd_mappings_path):
            id_to_icd_map = {i: {"code": codes[i], "description": descriptions[i]} for i in range(len(icd_data))}
            with open(self.icd_mappings_path, "w") as f:
                json.dump(id_to_icd_map, f, indent=2)
        else:
            with open(self.icd_mappings_path, "r") as f:
                id_to_icd_map = json.load(f)
    
        report_text = freetext
        extracted_entities = extract_entities_llm(report_text, temperature=0.0, max_tokens=500)
        split_outputs = [x for x in extracted_entities.split("\n") if x]
        
        original_phrase = []
        predicted_codes = []
        predicted_descriptions = []
        predicted_similarity_scores = []
        softmax_scores = []
        
        for phrase in split_outputs: 
            candidate_phrases = [phrase]
            if self.reformulate_entities:
                try:
                    reformulations = reformulate_entity_synonyms(phrase, 
                                                                 num_variations=5)
                    logger.info(f"Reformulated phrases for '{phrase}': {reformulations}")
                    candidate_phrases.extend(reformulations)
                except Exception as e:
                    logger.error(f"Error during reformulation: {e}")
            
            logger.info(f"Processing phrase: {phrase}")

            best_icd = None
            best_score = float("inf")
            best_description = None


            for candidate in candidate_phrases:
                # Retrieve closest ICD codes for each the candidate phrase
                if self.reasoning:
                    closest_icds = retrieve_closest_icd_code(
                        candidate, db, id_to_icd_map, k=3)
                    closest_icd = llm_reasoning(
                        closest_icds=closest_icds,
                        phrase=phrase
                    )
                else:
                    closest_icd = retrieve_closest_icd_code(
                        candidate, db, id_to_icd_map)[0]

                if closest_icd and closest_icd['similarity_score'] < best_score:
                    best_icd = closest_icd['icd_code']
                    best_description = closest_icd['icd_description']
                    best_score = closest_icd['similarity_score']
                    # similarity_debug = closest_icd['similarity_debug']
                    # best_softmax = closest_icd['softmax_retrieval']

            if best_icd:
                original_phrase.append(phrase)
                predicted_codes.append(best_icd)
                predicted_descriptions.append(best_description)
                predicted_similarity_scores.append(best_score)
                # softmax_scores.append(best_softmax)
                logger.info(f"Best ICD code for phrase '{phrase}': {best_icd} - {best_description} (Distance: {best_score:.3f})")
                #logger.info(f"Best ICD code for phrase '{phrase}': {best_icd} - {best_description} (Distance: {best_score:.3f}, Probability: {best_softmax:.3f})")
            else:
                logger.info(f"No ICD code found for phrase: {phrase}")
        logger.info(f"Predicted ICD codes: {predicted_codes}")
        logger.info(f"Predicted ICD descriptions: {predicted_descriptions}")

        return original_phrase, predicted_codes, predicted_descriptions, predicted_similarity_scores