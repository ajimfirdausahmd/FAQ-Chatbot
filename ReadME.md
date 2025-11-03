## Malaysia Births Chatbot (DOSM) — RAG + Citations

A small FAQ/data chatbot over Demography: Births (signs of life) using LangChain + LangGraph + FAISS.
It retrieves from 5 DOSM datasets (national, state, sex/ethnic, state+sex+ethnic, district+sex)

### QuickStart(≤10 min)

1. Clone & Install

```bash
git clone https://github.com/your/repo.git
cd <your-repo>
python -m venv .venv 
source .venv/bin/activate   
pip install -r requirements.txt
```

2. Prepare environment

Create .env in the repo root

```bash
OPENAI_API_KEY=<redacted>
```

3. Put data

Place these CSVs in data/ :
* birth.csv — Annual number of births (national total)
* birth_state.csv — Annual number of births by state
* birth_sex_ethnic.csv — Annual births by sex and ethnic group (national)
* birth_sex_ethnic_state.csv — Annual births by sex, ethnic group, and state
* birth_district_sex.csv — Annual births by district and sex

4. Run Evaluation

```bash
python app.py
```

This will:
* Build or load a FAISS index in /faiss_index/
* Run 15 evaluation queries
* Save results to eval/results.jsonl

### Data Card 




### RAG Design

Ingestion & Indexing
* CSV Loader: Each row = one document. We attach metadata.source (file path) and metadata.row (row index).
* Chunking: Not required for CSV (rows are short). For long text (HTML/PDF), we’d use RecursiveCharacterTextSplitter (size ~1000, overlap ~120).
* Embeddings: text-embedding-3-small (fast, good recall for numeric/text mix).
* Vector DB: FAISS; stored at src/faiss_index/.

Retrieval
* Retriever: k=5 nearest neighbors (tuneable).
* Optional filtering by topic classifier (births-only).

Answering
* LLM prompt enforces grounded answers only from retrieved context; asks a clarifying question or refuses if missing.

Low-confidence policy
* If fewer than N relevant docs (e.g., < 2) or retrieval returns nothing:
    * Ask a targeted clarifying question (e.g., “Which year/state/district?”) or
    * Refuse politely when off-scope (e.g., CPI queries).

### Evaluation Methodology & Results



### Limitation & Future work