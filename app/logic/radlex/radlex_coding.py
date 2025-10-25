import os
import json
import logging
import pandas as pd
# from more_itertools import chunked
import chromadb
import numpy as np

from config import CONFIG
from logic.radlex.radlex_files.ner import extract_entities_llm, reformulate_entity_synonyms
from logic.radlex.radlex_files.rag import retrieve_closest_radlex_code, llm_reasoning
from logic.radlex.radlex_files.graph import build_and_visualize_subgraph, get_subgraph_html
import streamlit as st
from streamlit.components.v1 import html as st_html
from shared.embedding_model import embedding_model

logger = logging.getLogger("radlex")

class RadLexInference:
    def __init__(self):
        # load configuration
        cfg = CONFIG["radlex"]
        self.mapping_path         = cfg["mapping_path"]
        self.chromadb_path        = cfg["chromadb_path"]
        self.max_docs             = cfg["max_docs"]
        self.reasoning            = cfg["reasoning"]
        self.batch_size           = cfg["batch_size"]
        self.reformulate          = cfg["reformulate_entities"]
        self.chromadb_collection_name = cfg["chromadb_collection_name"]

        # paths under logic/radlex/data/
        base = os.path.join(os.path.dirname(__file__), "data")
        self.csv_path   = os.path.join(base, "d_radlex_entities.csv")
        self.graph_path = os.path.join(base, "RadLex_graph.gpickle")

        # load mappings (id → code+description)
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.id_to_radlex = json.load(f)

        # bootstrap ChromaDB
        self.db = self._init_chroma()
        entries = self.db.get(include=["embeddings", "metadatas"])
        self.db_embs = np.array(entries["embeddings"])  # shape: (N, D)
        self.db_ids = entries["ids"]

    def _init_chroma(self):
        logger.info("Initializing RadLex ChromaDB…")
        client = chromadb.PersistentClient(path=self.chromadb_path)
        coll = client.get_or_create_collection(
            name=self.chromadb_collection_name
        )

        # load descriptions from CSV
        df = pd.read_csv(self.csv_path)
        docs = df["preferred_description"].tolist()

        total = len(docs)
        current = coll.count()
        to_add = docs[current:]
        if to_add:
            max_bs = self.batch_size
            logger.info(f"Adding {len(to_add)} docs to RadLex DB in chunks of {max_bs}…")
            # slice-based chunking without extra dependency
            start = current
            for offset in range(0, len(to_add), max_bs):
                chunk = to_add[offset : offset + max_bs]
                end   = start + len(chunk)
                ids   = [str(i) for i in range(start, end)]
                embeddings = embedding_model.encode(
                    chunk,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                coll.add(documents=chunk, ids=ids, embeddings=embeddings)
                start = end
            logger.info("RadLex ChromaDB population complete.")
        else:
            logger.info("RadLex ChromaDB already up-to-date; skipping add.")
        return coll

    def predict_radlex(self, text: str):
        """
        Full pipeline: NER → optional reformulation → RAG lookup.
        Returns lists: (phrases, codes, descriptions, similarity_scores)
        """
        # 1. extract entities
        raw = extract_entities_llm(text)
        phrases = [line for line in raw.split("\n") if line.strip()]

        codes, descs, similarity_scores, softmax_scores = [], [], [], []

        for phrase in phrases[: self.max_docs]:
            candidates = [phrase]

            # 2. reformulation of entities
            if self.reformulate:
                try:
                    syns = reformulate_entity_synonyms(phrase)
                    candidates.extend(syns)
                except Exception as e:
                    logger.warning(f"Reformulation failed: {e}")

            best = None
            best_score = float("inf")
            similarity_debug = None
            best_softmax = None


            # 3. retrieve closest RadLex codes for each candidate phrase
            for cand in candidates:
                hits = retrieve_closest_radlex_code(
                    cand,
                    self.db,
                    self.id_to_radlex,
                    self.db_embs,
                    self.db_ids,
                    k=3
                )
                pick = None
                if self.reasoning:
                    pick = llm_reasoning(hits, phrase)
                else:
                    pick = hits[0] if hits else None

                if pick and pick["similarity_score"] < best_score:
                    best = pick
                    best_score = pick["similarity_score"]
                    similarity_debug = pick['similarity_debug']
                    best_softmax = pick['softmax_retrieval']

            if best:
                codes.append(best["radlex_code"])
                descs.append(best["radlex_description"])
                similarity_scores.append(best_score)
                softmax_scores.append(best_softmax)
                logger.info(f"Best RadLex code for phrase '{phrase}': {best['radlex_code']} - {best['radlex_description']} (Distance: {best_score:.3f}, Probability: {best_softmax:.3f})")

            else:
                codes.append(None)
                descs.append(None)
                similarity_scores.append(None)
                softmax_scores.append(None)
                logger.info(f"No RadLex code found for phrase: {phrase}")

        return phrases, codes, descs, similarity_scores, softmax_scores
    
    def visualize(self, codes: list[str], root: str = "RID1"):
        """
        Build an HTML visualization in-memory (no files), returning:
          - html_str: the full HTML page as a string
          - html_bytes: the UTF-8 bytes for download
        """
        html_str = get_subgraph_html(codes=codes, root=root)
        html_bytes = html_str.encode("utf-8")
        return html_str, html_bytes
    
class RadLexProcessor:
    def __init__(self, progress_placeholder, result_placeholder, root: str = "RID1"):
        """
        progress_placeholder, result_placeholder: from chat.py via status_block()
        root: which RID to anchor shortest-paths at (default "RID1")
        """
        self.progress = progress_placeholder
        self.result   = result_placeholder
        self.root     = root

    def run(self, text: str):
        # 1) Run extraction & graph build under the progress box
        with self.progress:
            with st.spinner("Running RadLex mapping…"):
                rl = RadLexInference()
                phrases, codes, descs, sims, softmax = rl.predict_radlex(text)
                graph_html, graph_bytes = rl.visualize(codes, root=self.root)

        # 2) Render list, graph, and download BTN all inside result box
        with self.result.container():
            if not codes:
                st.warning("No RadLex codes found.")
                return

            # a) bullet list of phrase→code→desc(score)
            table_data = {
                "Phrase": phrases,
                "ICD Code": codes,
                "Description": descs,
                "Distance": [f"{sc:.3f}" for sc in sims],
                "Softmax Prob": [f"{sm:.3f}" for sm in softmax],
            }

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # b) interactive PyVis graph
            st_html(graph_html, height=600)

            # c) download button for the same HTML
            st.download_button(
                label="📥 Download RadLex subgraph",
                data=graph_bytes,
                file_name="radlex_subgraph.html",
                mime="text/html"
            )