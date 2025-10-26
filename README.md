# Multilingual Radiology Report Structuring with RAG-Powered LLMs

This repository provides an application with a Streamlit webinterface that structures freetext radiology reports by 
  1. Assigning ICD-10-CM Codes
  2. Mapping RadLex terminologies
  3. Selecting a template from the internal database and filling it with extracted information

This is the result of a project at TUM Data Innovation Lab in cooperation with Klinikum Rechts der Isar.  
If you want to know in detail how the pipelines work, check out the [written report](https://www.mdsi.tum.de/di-lab/vergangene-projekte/ss25-tum-klinikum-rechts-der-isar-leveraging-llms-for-information-extraction-in-radiology-reports/).

> [!NOTE]
> Currently **English** and **German** reports are supported. If you want to add another language, take a look at section 3.

## 1. Setup

> [!IMPORTANT]  
> Make sure you have git lfs installed. Otherwise the pipelines will build their internal knowledge representation databases when being executed for the first time.

First clone the project:
```bash
git clone git@github.com:MRI-TUM-DI-LAB/radiology-app.git
```

Then place your Vertex AI service-account JSON under `app/vertexai_credentials.json`. It should look like this:

```json
{
  "type": "service_account",
  "project_id": "",
  "private_key_id": "",
  "private_key": "-----BEGIN PRIVATE KEY-----\n YOURKEY \n-----END PRIVATE KEY-----\n",
  "client_email": "",
  "client_id": "",
  "auth_uri": "",
  "token_uri": "",
  "auth_provider_x509_cert_url": "",
  "client_x509_cert_url": "",
  "universe_domain": ""
}
```

> [!NOTE]  
> Using a local Ollama instance as LLM service is also supported. For this go into `app/config.yaml`, set the `general.llm.llm_service` value to `ollama` and adjust the config options under `general.llm.ollama` if needed.

Afterwards you can simply run the app using docker:
```bash
cd radiology-app && docker compose up
```
Alternatively you can run the app without docker by going into the `app` folder, installing the requirements and running `python -m streamlit run app.py`. 
SentenceTransformer will download the embedding model on startup which might take a little bit.
Afterwards the interface will be available at [http://localhost:8501](http://localhost:8501).

##  2. Repository Structure

```text
.
├── docker-compose.yaml
├── app
│   ├── app.py                      # Streamlit entry point
│   ├── config.yaml                 # Configuration possibilities for the pipelines including LLM and embedding model choice
│   ├── requirements.txt
│   ├── chroma_data/                # ChromaDB data for all 3 pipelines is stored here.
|   ├── logic/                      # Contains the logic for each of the pipelines
|   ├── pages/                      # Streamlit pages
|   └── shared
|       ├── embedding_model.py      # Embedding model loader
|       └── llm_interface.py        # LLM interface. If you want to add another LLM, do it here.
|
├── experiments                     # Experiments that were carried out
|   ├── icd_coding/                
|   └── RadLex/
|
├── templates                       # Template collection for the report structuring pipeline
│   ├── de/
|   ├── en/
|
└── test/                           # Tests and utility scripts

```

## 3. Adding new languages

  1. In `app/config.yaml` add your language code to the values under `report_structuring.detect_languages`
  2. In `app/logic/report_structuring/prompts/` create a file using your language code containing the translated prompts from en.yaml
  3. On the Manage Template page, upload a set of templates in your language

##  4. Features

### 📁 Template Management

Manage your JSON report templates from the **Manage Templates** page:

- **Upload, List & Delete** — Interact with templates stored in **ChromaDB**
- **Filter & Sort** — By language, modality, name, etc.
- **Language Adjustment** — Manually fix template classification if needed

### 🏥 ICD-10 Coding

Automated coding combining NER, vector search, and optional LLM refinement:

- **Entity Recognition** — Extract disease mentions using NER
- **ICD-10 Retrieval** — Vector search in **ChromaDB** for closest matches
- **LLM Refinement (Optional)** — Accept, modify, or reject codes
- **Output Table** — Includes:
  - Matched phrases  
  - ICD-10 codes  
  - Descriptions  
  - Similarity scores

### 🧬 RadLex Mapping

Turn radiology text into structured RadLex concepts:

- **Concept Extraction** — Use NER for imaging-related terms
- **Semantic Search** — RAG lookup via **RadLex embeddings**
- **Ontology Graph** — Build and visualize subgraphs from RadLex ontology

### 🧾 Report Structuring

Convert free-text reports into structured templates:

- **Language & Metadata Detection** — Identify language and core info
- **Template Search** — Generate semantic queries to retrieve templates
- **Filling Modes:**
  - **All-at-once** — Instant full JSON fill  
  - **Step-by-step** — Field-by-field placeholder filling  
  - **Validated** — LLM + local validation during step-by-step fill

> [!NOTE]
> The RadLex graph as well as the filled template are downloadable after computation

## 5. Running ICD Experiments Locally

### 5.1. Configure Workspace

```bash
cd experiments/icd_coding
```

The entire ICD pipeline works with Gemini as the LLM, thus 

- **Add** `.env` and set your Google Cloud Project credentials
- **Place** your Vertex AI service-account JSON in the current folder

### 5.2. Configure RAG Experiment

Set your experiment parameters in `experiment_rag.py` before running the pipeline. Below are the available configuration options:

| Variable                | Description |
|-------------------------|-------------|
| `DATASET`               | Select the dataset: `PARROT` or `CodiESP` |
| `LANG`                  | Choose English for the translation or keep the original report language |
| `EMBEDDING_MODEL_NAME`  | Name of the embedding model to use |
| `GEMINI_MODEL_NAME`     | Name of the Gemini model to use |
| `MAX_DOCS`              | Limit the number of medical reports loaded |
| `REASONING`             | LLM selects best ICD code from top 3 matches. Default = `False` |
| `REFORMULATE_ENTITIES`  | LLM reformulates the extracted entities. Default = `True` |
| `THRESHOLDING`          | LLM validates the predicted ICD codes. Default = `True` |
| `LLM_TEMPERATURE`       | Set temperature value for LLM response variability |

---

### 5.3. Run the Experiment

Run the following command in your terminal:

```bash
python experiment_rag.py
```

This will:

- Load medical reports from the selected dataset
- Generate ground truth ICD label file (if not already available)
- Create or load a ChromaDB collection of ICD descriptions 
- Create a mapping between ICD codes and their descriptions (if not already available)
- Execute the RAG pipeline and save results to: `data/output/experiment_rag`
- Automatically evaluate performance metrics and save the results

