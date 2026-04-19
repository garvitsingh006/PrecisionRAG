# Precision RAG 🎯

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline served via **FastAPI**, built with LangGraph combining Corrective RAG (CRAG) and Self-RAG techniques for high-precision, hallucination-resistant question answering over documents.

---

## Overview

Precision RAG goes beyond standard RAG by layering multiple self-checking mechanisms:

1. **Decides** whether retrieval is even necessary
2. **Rewrites** the query for optimal vector retrieval
3. **Retrieves & evaluates** documents for relevance
4. **Falls back to web search** when local docs aren't good enough
5. **Refines** context by filtering out irrelevant chunks
6. **Generates** an answer grounded in the refined context
7. **Checks for hallucinations** and revises until the answer is fully supported
8. **Checks usefulness** and rewrites the retrieval query if the answer misses the point

---

## Architecture

```
START
  └─► decide_retrieval
        ├── (no retrieval needed) ──► direct_generate ──► END
        └── (retrieval needed)   ──► rewrite_query
                                         └─► retrieve
                                               └─► evaluate_retrieved_docs
                                                     ├── (good docs) ──► refine
                                                     └── (bad docs)  ──► web_search ──► refine
                                                                               └─► generate_answer
                                                                                     └─► is_sup (hallucination check)
                                                                                           ├── (fully supported) ──► is_useful
                                                                                           └── (not supported)   ──► revise_answer ──► is_sup (loop)
                                                                                                                         ├── (useful)       ──► END
                                                                                                                         ├── (not useful)   ──► rewrite_query (loop)
                                                                                                                         └── (max retries)  ──► no_answer ──► END
```

![Alt text](./Graph_Architecture.png)

---

## Key Features

| Feature | Description |
|---|---|
| **Smart Retrieval Decision** | Skips retrieval for timeless/conceptual questions; always retrieves for time-sensitive or rule-based queries |
| **Query Rewriting** | Rewrites the user question into a vector-retrieval-optimized query before every retrieval attempt |
| **Document Relevance Scoring** | Each retrieved chunk is scored 0–1; only chunks above a threshold are kept |
| **CRAG Web Fallback** | If local docs score poorly, Tavily web search is triggered automatically |
| **Context Refinement** | Retrieved context is split into fine strips and filtered chunk-by-chunk for relevance |
| **Hallucination Detection** | Verifies every claim in the generated answer against the refined context |
| **Answer Revision Loop** | Unsupported answers are rewritten using only context-grounded phrases (up to `MAX_HALLU_RETRIES` times) |
| **Usefulness Check** | Checks whether the answer actually addresses the question (not just factually grounded) |
| **Query Rewriting Loop** | If the answer is not useful, the retrieval query is rewritten and the full pipeline reruns (up to `MAX_USEFUL_RETRIES` times) |
| **Confidence Scoring** | Final output includes a composite confidence score across support, usefulness, and relevance |
| **Experiment Configs** | Pipeline parameters (model, chunk size, top_k, temperature, etc.) are managed as named configs via API |
| **Run Resume** | Failed or pending runs can be resumed from their last checkpointed state |
| **Persistent Checkpointing** | Graph state is persisted via PostgreSQL (`PostgresSaver`) for run continuity |

---

## Tech Stack

- **LLM**: DeepSeek-V3 via HuggingFace Endpoint (Novita provider)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **Graph Orchestration**: LangGraph (`StateGraph`)
- **Web Search**: Tavily
- **API**: FastAPI
- **App DB**: PostgreSQL via SQLAlchemy (runs, evaluations, configs)
- **Graph Checkpointing**: PostgreSQL (`PostgresSaver`)
- **Output Parsing**: Pydantic + LangChain `PydanticOutputParser` / `OutputFixingParser`

---

## Setup

> Requires **Python 3.12+**

### 1. Create a virtual environment

```bash
python -m venv myvenv
.\myvenv\Scripts\Activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL

You need a running PostgreSQL instance. Both `DB_URI_FASTAPI` and `DB_URI_GRAPH` can point to the same database — SQLAlchemy and LangGraph manage their own tables independently. Tables are created automatically on startup; no manual migrations needed.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
TAVILY_API_KEY=your_tavily_key
DB_URI_FASTAPI=postgresql+psycopg://user:password@localhost:5432/yourdb
DB_URI_GRAPH=postgresql://user:password@localhost:5432/yourdb
```

> `DB_URI_FASTAPI` uses the `postgresql+psycopg://` scheme — SQLAlchemy requires the explicit `+psycopg` driver suffix.  
> `DB_URI_GRAPH` uses plain `postgresql://` — LangGraph's `PostgresSaver` handles the driver internally.

### 5. Add your PDF documents

Place your PDF files in `models/docs/`. Example:

```
models/
  docs/
    document1.pdf
    document2.pdf
```

The pipeline auto-discovers all `.pdf` / `.PDF` files in that folder on startup and builds a FAISS index. If a `faiss_index/` folder already exists, it is loaded from disk instead of being rebuilt.

### 6. Run the server

```bash
uvicorn app:app --reload
```

---

## API

### Health check

```
GET /
```

Returns: `{"response": "Welcome to Precision RAG made by Garvit Singh"}`

---

### Create an experiment config

```
POST /config/new
```

```json
{
  "id": "exp_v1",
  "model": "deepseek-ai/DeepSeek-V3",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunk_size": 700,
  "chunk_overlap": 150,
  "top_k": 4,
  "temperature": 0.7
}
```

### Run a new evaluation

```
POST /evaluation/new
```

```json
{
  "question": "Your question here",
  "config_id": "exp_v1"
}
```

### Resume a failed/pending run

```
POST /evaluation/resume/{run_id}
```

Only runs with status `failed` or `pending` can be resumed. The graph replays from its last PostgreSQL checkpoint — no tokens are wasted re-running completed nodes.

---

## Response Payload

```json
{
  "answer": "...",
  "evaluation": {
    "confidence": 0.85,
    "retrieval_relevance": 0.72,
    "support": {
      "label": "fully_supported",
      "score": 1.0,
      "reason": "..."
    },
    "usefulness": {
      "label": "useful",
      "score": 1.0,
      "reason": "..."
    }
  },
  "pipeline": {
    "retrieval_used": true,
    "web_search_used": false,
    "hallucination_retries": 0,
    "usefulness_retries": 1
  },
  "performance": {
    "latency_ms": 4231.0
  },
  "experiment": {
    "model": "deepseek-ai/DeepSeek-V3",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 700,
    "chunk_overlap": 150,
    "top_k": 4,
    "temperature": 0.7
  }
}
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_HALLU_RETRIES` | `3` | Max answer revision attempts before accepting the current answer |
| `MAX_USEFUL_RETRIES` | `3` | Max query rewrite attempts before returning "no answer found" |
| `UPPER_TH` | `0.7` | Document relevance score above which retrieval is deemed "correct" |
| `LOWER_TH` | `0.3` | Document relevance score below which a doc is discarded |
| `chunk_size` | `700` | Chunk size for initial PDF splitting (set per config) |
| `chunk_overlap` | `150` | Overlap between PDF chunks (set per config) |
| `top_k` | `4` | Number of documents retrieved per query (set per config) |

---

## Getting Started

A complete workflow from zero to first answer:

**curl**
```bash
# 1. Create a config
curl -X POST http://localhost:8000/config/new \
  -H "Content-Type: application/json" \
  -d '{"id": "exp_v1", "model": "deepseek-ai/DeepSeek-V3", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "chunk_size": 700, "chunk_overlap": 150, "top_k": 4, "temperature": 0.7}'

# 2. Run an evaluation
curl -X POST http://localhost:8000/evaluation/new \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the judge's verdict?", "config_id": "exp_v1"}'
```

**Python**
```python
import requests

base = "http://localhost:8000"

requests.post(f"{base}/config/new", json={
    "id": "exp_v1",
    "model": "deepseek-ai/DeepSeek-V3",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 700,
    "chunk_overlap": 150,
    "top_k": 4,
    "temperature": 0.7
})

response = requests.post(f"{base}/evaluation/new", json={
    "question": "What was the judge's verdict?",
    "config_id": "exp_v1"
})
print(response.json())
```

---

## Confidence Score Formula

```
confidence = (support_score × 0.5) + (usefulness_score × 0.3) + (relevance_score × 0.2)
```

Where:
- `support_score`: `1.0` (fully supported) → `0.8–1.0`, `0.5` (partially) → `0.2–0.8`, `0.0` (no support) → `0.0–0.2`
- `usefulness_score`: `1.0` (useful), `0.0` (not useful)
- `relevance_score`: average relevance score of retrieved documents

---

## Optimization Tips

**`chunk_size` / `chunk_overlap`**  
Smaller chunks (400–600) improve precision for narrow factual questions. Larger chunks (700–1000) work better for questions requiring broader context. Increase `chunk_overlap` if answers feel cut off at chunk boundaries. Note: changing these requires deleting `faiss_index/` so the index is rebuilt.

**`top_k`**  
Higher values increase recall but add latency and noise — more chunks means more LLM calls during refinement. `4` is a good default; go up to `6–8` only for complex multi-part questions.

**`temperature`**  
Keep it low (`0.2–0.5`) for factual/legal documents where precision matters. Higher values (`0.6–0.9`) can help when answers need to be more synthesized or conversational.

---

## Project Structure

```
app.py                  # FastAPI app with all endpoints
db.py                   # SQLAlchemy engine and session
db_models.py            # ORM models: Run, Evaluation, MetricScore, ExperimentConfig
schemas.py              # Pydantic request/response schemas
utils.py                # run_new_evaluation, resume_evaluation, create_config
models/
  precision_rag.py      # Full LangGraph pipeline (nodes, edges, graph builder)
  docs/                 # Place your PDF documents here
faiss_index/            # Auto-generated FAISS index (persisted to disk)
.env                    # Environment variables (not committed)
requirements.txt
README.md
```
