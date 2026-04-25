# Precision RAG 🎯

A production-grade **Retrieval-Augmented Generation (RAG)** system — a FastAPI backend running a LangGraph pipeline, paired with a React frontend. Combines Corrective RAG (CRAG) and Self-RAG techniques for high-precision, hallucination-resistant question answering over your own documents.

---

## Overview

Precision RAG goes beyond standard RAG by layering multiple self-checking mechanisms:

1. **Decides** whether retrieval is even necessary
2. **Rewrites** the query for optimal vector retrieval
3. **Retrieves & evaluates** documents for relevance (single batched LLM call)
4. **Corrective re-retrieval** — if docs are ambiguous, rewrites the query and re-retrieves once before falling back to web search
5. **Falls back to web search** (Tavily) only when local docs are genuinely insufficient
6. **Refines** context by filtering all strips in a single batched LLM call
7. **Generates** an answer grounded in the refined context
8. **Checks for hallucinations** and revises until the answer is fully supported
9. **Checks usefulness** and rewrites the retrieval query if the answer misses the point

---

## Architecture

```
START
  └─► decide_retrieval
        ├── (no retrieval needed) ──► direct_generate ──► END
        └── (retrieval needed)   ──► rewrite_query ◄─────────────────────────────────────────┐
                                         └─► retrieve (parent-document retrieval)             │
                                               └─► evaluate_retrieved_docs                    │
                                                     ├── (correct)   ──► refine               │
                                                     ├── (ambiguous) ──► corrective_rewrite    │
                                                     │                         └─► retrieve    │
                                                     │                               └─► evaluate_retrieved_docs
                                                     │                                     ├── (correct)       ──► refine
                                                     │                                     └── (bad/ambiguous) ──► web_search ──► refine
                                                     └── (incorrect) ──► web_search ──► refine
                                                                               └─► generate_answer
                                                                                     └─► is_sup (hallucination check)
                                                                                           ├── (fully_supported) ──► is_useful
                                                                                           └── (not supported)   ──► revise_answer ──► is_sup (loop, max MAX_HALLU_RETRIES)
                                                                                                 ├── (useful)      ──► END
                                                                                                 ├── (not useful)  ──────────────────────────────────────────────────────────┘
                                                                                                 └── (max retries) ──► no_answer ──► END
```

![Pipeline Graph](./backend/Graph_Architecture.png)

---

## Key Features

| Feature | Description |
|---|---|
| **Smart Retrieval Decision** | Skips retrieval for timeless/conceptual questions; always retrieves for time-sensitive or document-specific queries |
| **Query Rewriting** | Rewrites the user question into a vector-retrieval-optimized query before every retrieval attempt |
| **Batched Doc Evaluation** | All retrieved chunks are scored in a single LLM call instead of one call per chunk |
| **Corrective Re-retrieval** | If docs are ambiguous, rewrites the query with a different angle and re-retrieves once before falling back to web search — the core CRAG insight |
| **CRAG Web Fallback** | Tavily web search is only triggered after corrective re-retrieval fails or docs score as fully incorrect |
| **Batched Context Refinement** | All context strips are filtered in a single LLM call — not one call per strip |
| **Parent-Document Retrieval** | Child chunks (small) are embedded for retrieval; matched children are swapped for their larger parent chunks before evaluation — better context, same precision |
| **Retriever Caching** | FAISS index is built once per unique (docs + config) combination and cached in memory — no re-embedding on query rewrites |
| **Hallucination Detection** | Verifies every claim in the generated answer against the refined context |
| **Answer Revision Loop** | Unsupported answers are rewritten using only context-grounded phrases (up to `MAX_HALLU_RETRIES` times) |
| **Usefulness Check** | Checks whether the answer actually addresses the question (not just factually grounded) |
| **Query Rewriting Loop** | If the answer is not useful, the retrieval query is rewritten and the full pipeline reruns (up to `MAX_USEFUL_RETRIES` times) |
| **Confidence Scoring** | Final output includes a composite confidence score across support, usefulness, and relevance |
| **Experiment Configs** | Pipeline parameters (model, chunk size, top_k, temperature, etc.) are managed as named configs via API and frontend |
| **Run Resume** | Failed or pending runs can be resumed from their last PostgreSQL checkpoint — no tokens wasted re-running completed nodes |
| **Document Upload** | PDFs are uploaded via the frontend to Cloudinary for storage and simultaneously sent to the backend, which saves them to `models/docs/` and invalidates the retriever cache |

---

## Tech Stack

### Backend
- **LLM**: DeepSeek-V3 via HuggingFace Endpoint (Novita provider)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (in-memory, cached per config)
- **Graph Orchestration**: LangGraph (`StateGraph`)
- **Web Search**: Tavily
- **API**: FastAPI with CORS
- **App DB**: PostgreSQL via SQLAlchemy (runs, evaluations, configs)
- **Graph Checkpointing**: PostgreSQL (`PostgresSaver`)
- **Output Parsing**: Pydantic + LangChain `PydanticOutputParser` / `OutputFixingParser`
- **Document Storage**: Cloudinary (raw upload)

### Frontend
- **Framework**: React 19 + Vite
- **Styling**: Tailwind CSS (Ollama-inspired design system — pure grayscale, pill shapes, zero shadows)
- **Routing**: React Router v6
- **HTTP**: Axios
- **File Upload**: react-dropzone

---

## Project Structure

```
Precision RAG (P1)/
├── backend/
│   ├── models/
│   │   ├── docs/               # Place PDF documents here (auto-discovered)
│   │   └── precision_rag.py    # Full LangGraph pipeline
│   ├── app.py                  # FastAPI app — all endpoints + CORS + Cloudinary
│   ├── db.py                   # SQLAlchemy engine and session
│   ├── db_models.py            # ORM models: Run, Evaluation, MetricScore, ExperimentConfig
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── utils.py                # run_new_evaluation, resume_evaluation, create_config
│   ├── .env                    # Backend environment variables (not committed)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Nav.jsx         # Sticky top navigation
│   │   │   ├── FileUploader.jsx # Drag-and-drop PDF upload with stage feedback
│   │   │   ├── AnswerPanel.jsx  # Full response display with score bars and metrics
│   │   │   ├── PipelineVisualizer.jsx # Animated pipeline step tracker
│   │   │   ├── ConfigForm.jsx  # Experiment config creation form
│   │   │   └── Loader.jsx      # Spinner component
│   │   ├── pages/
│   │   │   ├── Landing.jsx     # Hero + pipeline diagram + feature highlights
│   │   │   ├── Upload.jsx      # Document upload page
│   │   │   ├── Query.jsx       # Ask question + config selector + results
│   │   │   ├── Config.jsx      # Create and list experiment configs
│   │   │   └── Resume.jsx      # Resume a failed/pending run by ID
│   │   ├── App.jsx             # Router setup
│   │   └── index.css           # Tailwind base
│   ├── .env                    # Frontend environment variables (not committed)
│   └── package.json
├── DESIGN.md                   # Ollama-inspired design system reference
└── README.md
```

---

## Setup

> Requires **Python 3.12+** and **Node.js 18+**

### Backend

**1. Create a virtual environment**

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up PostgreSQL**

You need a running PostgreSQL instance. Both `DB_URI_FASTAPI` and `DB_URI_GRAPH` can point to the same database — SQLAlchemy and LangGraph manage their own tables independently. Tables are created automatically on startup; no manual migrations needed.

**4. Configure environment variables**

Create `backend/.env`:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
TAVILY_API_KEY=your_tavily_key

DB_URI_FASTAPI=postgresql+psycopg://user:password@localhost:5432/yourdb
DB_URI_GRAPH=postgresql://user:password@localhost:5432/yourdb

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

FRONTEND_URL=http://localhost:5173
```

> `DB_URI_FASTAPI` uses the `postgresql+psycopg://` scheme — SQLAlchemy requires the explicit `+psycopg` driver suffix.  
> `DB_URI_GRAPH` uses plain `postgresql://` — LangGraph's `PostgresSaver` handles the driver internally.

**5. Run the backend**

```bash
uvicorn app:app --reload
```

Backend runs at `http://localhost:8000`.

---

### Frontend

**1. Install dependencies**

```bash
cd frontend
npm install
```

**2. Configure environment variables**

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
VITE_CLOUDINARY_CLOUD_NAME=your_cloud_name
VITE_CLOUDINARY_UPLOAD_PRESET=your_unsigned_upload_preset
```

> `VITE_CLOUDINARY_UPLOAD_PRESET` must be an **unsigned** upload preset. Create one in Cloudinary Dashboard → Settings → Upload → Upload presets → Add preset → set Signing Mode to **Unsigned**.

**3. Run the frontend**

```bash
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Getting Started

```bash
# 1. Start the backend
cd backend && uvicorn app:app --reload

# 2. Start the frontend
cd frontend && npm run dev

# 3. Open http://localhost:5173
#    → /config  : create an experiment config
#    → /upload  : upload a PDF document
#    → /query   : ask a question and see the full pipeline output
```

Or via API directly:

```bash
# Create a config
curl -X POST http://localhost:8000/config/new \
  -H "Content-Type: application/json" \
  -d '{"id": "exp_v1", "model": "deepseek-ai/DeepSeek-V3", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "chunk_size": 700, "chunk_overlap": 150, "top_k": 4, "temperature": 0.7}'

# Run a query
curl -X POST http://localhost:8000/evaluation/new \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the judge verdict?", "config_id": "exp_v1"}'
```

---

## Frontend Pages

| Route | Page | Description |
|---|---|---|
| `/` | Landing | Hero section, pipeline diagram, feature highlights |
| `/upload` | Upload | Drag-and-drop PDF upload with live stage feedback |
| `/query` | Query | Ask a question, select a config, view full results |
| `/config` | Config | Create experiment configs, view existing ones |
| `/resume` | Resume | Resume a failed or pending run by run ID |

---

## End-to-End Workflow

### Document Upload

1. User drags a PDF onto `/upload` (FileUploader.jsx → react-dropzone)
2. Frontend uploads the raw file to **Cloudinary** via unsigned upload preset → gets back a secure URL
3. Frontend simultaneously POSTs the raw file bytes to `POST /upload-doc` (FastAPI)
4. Backend saves the file to `backend/models/docs/` and clears the in-memory retriever cache
5. On the next query, the FAISS index is rebuilt from all docs in `models/docs/` using the selected config's `chunk_size`, `chunk_overlap`, and `embedding_model`

> Documents are chunked and embedded **at query time**, not at upload time — so the config's hyperparameters are always applied correctly.

### Query Execution

1. User selects a config and submits a question on `/query` (Query.jsx)
2. Frontend POSTs to `POST /evaluation/new` with `{ question, config_id }`
3. FastAPI looks up the config from PostgreSQL and calls `run_new_evaluation` (utils.py)
4. `run_new_evaluation` invokes the compiled LangGraph pipeline (`build_graph`) with a PostgreSQL checkpointer
5. The pipeline runs through the nodes described in the Architecture section above
6. **Retrieval path** — `build_or_load_retriever` builds (or cache-hits) a FAISS index using parent-document retrieval:
   - PDFs are split into large **parent chunks** (for answer generation) and small **child chunks** (for embedding)
   - Child chunks are embedded with `HuggingFaceEmbeddings` and stored in FAISS
   - At query time, child hits are swapped for their parent chunks before evaluation
7. Final state is persisted to PostgreSQL (Run + Evaluation + MetricScore rows)
8. FastAPI returns the full `EvaluationResponse` JSON to the frontend
9. Frontend renders the answer, score bars, and pipeline trace in AnswerPanel.jsx + PipelineVisualizer.jsx

### Run Resume

1. User navigates to `/resume` and enters a run ID
2. Frontend POSTs to `POST /evaluation/resume/{run_id}`
3. FastAPI checks the run status — only `failed` or `pending` runs can be resumed
4. LangGraph replays from the last PostgreSQL checkpoint, skipping already-completed nodes

---

## Retriever Caching & Parent-Document Retrieval

`build_or_load_retriever` in `precision_rag.py` implements a two-level retrieval strategy:

- **Child chunks** (`chunk_size`) are embedded and stored in FAISS — used for similarity search
- **Parent chunks** (`chunk_size × 3`) are stored in `_parent_chunks_cache` — returned to the pipeline for evaluation and generation
- Each child chunk carries a `parent_id` metadata field linking it back to its parent

The FAISS index is keyed on `(pdf_files, chunk_size, chunk_overlap, embedding_model)` and cached in memory:

- Query rewrites within a run reuse the cached index — no re-embedding
- Different configs each get their own cached index
- Uploading a new document calls `_retriever_cache.clear()`, forcing a rebuild on the next query

---

## Backend API

### Health check
```
GET /
```
Returns: `{"response": "Welcome to Precision RAG made by Garvit Singh"}`

### Upload a document
```
POST /upload-doc
Content-Type: multipart/form-data
```
Field: `file` (PDF). Saves to `models/docs/` and clears the retriever cache.

> The frontend also uploads the same file directly to Cloudinary (client-side, unsigned preset) for cloud backup. The backend endpoint handles only local storage and cache invalidation.

### Cloudinary upload (client-side)
```
POST https://api.cloudinary.com/v1_1/{VITE_CLOUDINARY_CLOUD_NAME}/raw/upload
Content-Type: multipart/form-data
```
Fields: `file`, `upload_preset` (unsigned). Called directly from FileUploader.jsx — no backend proxy.

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

### List all configs
```
GET /configs
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
Only runs with status `failed` or `pending` can be resumed. The graph replays from its last PostgreSQL checkpoint.

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

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MAX_HALLU_RETRIES` | `3` | Max answer revision attempts before accepting the current answer |
| `MAX_USEFUL_RETRIES` | `3` | Max query rewrite attempts before returning "no answer found" |
| `UPPER_TH` | `0.7` | Doc relevance score above which retrieval is deemed "correct" |
| `LOWER_TH` | `0.3` | Doc relevance score below which a doc is discarded |
| `chunk_size` | `700` | Chunk size for PDF splitting (set per config) |
| `chunk_overlap` | `150` | Overlap between PDF chunks (set per config) |
| `top_k` | `4` | Number of documents retrieved per query (set per config) |
| `chunk_size × 3` | — | Parent chunk size used for answer generation (derived automatically) |

---

## Confidence Score Formula

```
confidence = (support_score × 0.5) + (usefulness_score × 0.3) + (relevance_score × 0.2)
```

- `support_score`: `0.8–1.0` (fully supported), `0.2–0.8` (partially), `0.0–0.2` (no support)
- `usefulness_score`: `1.0` (useful), `0.0` (not useful)
- `relevance_score`: average relevance score of retrieved document chunks

---

## Optimization Tips

**`chunk_size` / `chunk_overlap`**  
Smaller chunks (400–600) improve precision for narrow factual questions. Larger chunks (700–1000) work better for questions requiring broader context. Increase `chunk_overlap` if answers feel cut off at chunk boundaries. Changing these creates a new cached index automatically — no manual steps needed.

**`top_k`**  
Higher values increase recall but the evaluation and refinement steps still run in single batched calls, so the cost increase is modest. `4` is a good default; go up to `6–8` for complex multi-part questions.

**`temperature`**  
Keep it low (`0.1–0.3`) for factual/legal documents where precision matters. Higher values (`0.6–0.9`) can help when answers need to be more synthesized or conversational.
