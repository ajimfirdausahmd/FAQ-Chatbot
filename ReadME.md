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

Place these CSVs in `data/` :
* `birth.csv` — Annual live Births
* `birth_state.csv` — Annual live Births by state
* `birth_sex_ethnic.csv` — Annual live births by sex and ethnicity
* `birth_sex_ethnic_state.csv` — Annual live births by state, sex, & Ethnicity
* `birth_district_sex.csv` — Annual live births by district and sex

4. Run Evaluation

```bash
python app.py
```

This will:
* Build or load a FAISS index in `/faiss_index/`
* Run 15 evaluation queries
* Save results to `eval/results.jsonl`

5. Tool Choice & Model Provider

* Workflow/ Orchestration : LangGraph (python)
* RAG Framework : LangChain
* Vector Store : FAISS(local)
* Embeddings : `text-embedding-3-small` (OpenAI)
* LLM : `gpt-4o-mini` (OpenAI)
* Storage : Local filesystem (`data/`,`/faiss_index/`,`eval/`)

### Data Card 

Topic : Demography - Births
Scope : 5 datasets

1. Annual live Births
    
   * Source (URL) : The dataset was downloaded from the [National Registration Department & Department of Statistics Malaysia](https://open.dosm.gov.my/data-catalogue/births_annual).
   * Refresh candence : Annual
   * License : This data is made open under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
   * Last Update : 17 Oct 2024, 12:00

2. Annual live Births by state

   * Source (URL) : The dataset was downloaded from the [National Registration Department & Department of Statistics Malaysia](https://open.dosm.gov.my/data-catalogue/births_annual_state).
   * Refresh candence : Annual
   * License : This data is made open under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
   * Last Update : 17 Oct 2024, 12:00

3. Annual live births by sex and ethnicity

   * Source (URL) : The dataset was downloaded from the [DOSM Open Data Portal](https://open.dosm.gov.my/data-catalogue/births_annual_sex_ethnic).
   * Refresh candence : Annual
   * License : This data is made open under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
   * Last Update : 17 Oct 2024, 12:00

4. Annual live births by state, sex, & Ethnicity

   * Source (URL) : The dataset was downloaded from the [DOSM Open Data Portal](https://open.dosm.gov.my/data-catalogue/births_annual_sex_ethnic_state).
   * Refresh candence : Annual
   * License : This data is made open under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
   * Last Update : 17 Oct 2024, 12:00

5. Annual live births by district and sex

   * Source (URL) : The dataset was downloaded from the [DOSM Open Data Portal](https://open.dosm.gov.my/data-catalogue/births_district_sex).
   * Refresh candence : Annual
   * License : This data is made open under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
   * Last Update : 17 Oct 2024, 12:00


### RAG Design

Ingestion & Indexing
* CSV Loader: Each row = one document. We attach `metadata.source` (file path) and `metadata.row` (row index).
* Chunking: Not required for CSV (rows are short). For long text (HTML/PDF), we’d use `RecursiveCharacterTextSplitter` (size ~1000, overlap ~120).
* Embeddings: `text-embedding-3-small` (fast, good recall for numeric/text mix).
* Vector DB: FAISS; stored at `/faiss_index/`.

Retrieval
* Retriever: `k=5` nearest neighbors (tuneable).
* Optional filtering by topic classifier (births-only).

Answering
* LLM prompt enforces grounded answers only from retrieved context; asks a clarifying question or refuses if missing.

Low-confidence policy
* If fewer than N relevant docs (e.g., `< 2`) or retrieval returns nothing:
    * Ask a targeted clarifying question (e.g., “Which year/state/district?”) or
    * Refuse politely when off-scope (e.g., CPI queries).

### Evaluation Methodology & Results

Eval set(15 queries)

Focuses on:
* National totals and trends
* State comparisons
* Sex/ethnic breakdowns(national & by state)
* District queries
* One ambiguous region question(should ask a clarifying question)
* One off-scope question (CPI - should refuse)

metrics we record (per run)
* Latency: p50/p95 (ms)
* Retrieval hit-rate: % of queries with ≥1 relevant retrieved doc
* Hallucination rate: Non-empty answer with no valid "Sources: block(or empty citations) counts as hallucination

How we compute

`python app.py`

* Run all queries -> writes `eval/results.jsonl`
* Aggregates and prints metrics:
    p50: median of `latency_ms`
    p95: 95th percentile
    hit-rate: `retrieval_hit / total`
    hallucination-rate: `hallucination / total`


### Limitation & Future work

Limitations 

* Snapshot may lag behind the latest DOSM updates; answers are only as current as the ingested files.
* Simple keyword retrieval may miss relevant rows when queries are highly compositional (e.g., multi-constraint filters).

Future work

* Switch to text-embedding-3-large or bge-m3 for recall improvements; try PgVector if you need multi-tenant persistence.
* Enhance guardrails (e.g., automated refusal for out-of-scope topics) and confidence scoring

