# Precision RAG 🎯

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with LangGraph that combines Corrective RAG (CRAG) and Self-RAG techniques for high-precision, hallucination-resistant question answering over documents.

---

## Overview

Precision RAG goes beyond standard RAG by layering multiple self-checking mechanisms:

1. **Decides** whether retrieval is even necessary
2. **Retrieves & evaluates** documents for relevance
3. **Falls back to web search** when local docs aren't good enough
4. **Refines** context by filtering out irrelevant sentences
5. **Generates** an answer grounded in the refined context
6. **Checks for hallucinations** and revises until the answer is fully supported
7. **Checks usefulness** and rewrites the retrieval query if the answer misses the point

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
| **Document Relevance Scoring** | Each retrieved chunk is scored 0–1; only chunks above a threshold are kept |
| **CRAG Web Fallback** | If local docs score poorly, Tavily web search is triggered automatically |
| **Context Refinement** | Retrieved context is split into fine strips and filtered sentence-by-sentence for relevance |
| **Hallucination Detection** | Verifies every claim in the generated answer against the refined context |
| **Answer Revision Loop** | Unsupported answers are rewritten using only context-grounded phrases (up to `MAX_HALLU_RETRIES` times) |
| **Usefulness Check** | Checks whether the answer actually addresses the question (not just factually grounded) |
| **Query Rewriting Loop** | If the answer is not useful, the retrieval query is rewritten and the full pipeline reruns (up to `MAX_USEFUL_RETRIES` times) |
| **Confidence Scoring** | Final output includes a composite confidence score across support, usefulness, and relevance |
| **Persistent Checkpointing** | Graph state is persisted via PostgreSQL for conversation continuity across sessions |

---

## Tech Stack

- **LLM**: DeepSeek-R1 via HuggingFace Endpoint (Hyperbolic provider)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **Graph Orchestration**: LangGraph (`StateGraph`)
- **Web Search**: Tavily
- **Checkpointing**: PostgreSQL (`AsyncPostgresSaver`)
- **Output Parsing**: Pydantic + LangChain `PydanticOutputParser` / `OutputFixingParser`

---

## Setup

### 3. Create a Virtual Enviornment
```bash
python -m venv myvenv
```
```bash
.\myvenv\Scripts\Activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
TAVILY_API_KEY=your_tavily_key
DB_URI_GRAPH=postgresql://user:password@localhost:5432/yourdb
```

### 4. Add your PDF documents

Place your PDF files in the project root and update the loader in the notebook:

```python
docs = PyPDFLoader("your_doc1.pdf").load() + PyPDFLoader("your_doc2.pdf").load() + ...
```

---

## Usage

Run the notebook cells in order. The pipeline is invoked as an async LangGraph call:

```python
initial_state = {
    "question": "Your question here",
    "retrieval_query": "",
    "max_hallucination_retries": MAX_HALLU_RETRIES,   # default: 3
    "max_usefulness_retries": MAX_USEFUL_RETRIES,     # default: 3
    "hallucination_retries": 0,
    "usefulness_retries": 0,
    "answer": ""
}

final_state = await precision_rag.ainvoke(initial_state, config=config)
```

### Output payload

```python
{
    "answer": "...",
    "metrics": {
        "confidence": 0.85,
        "relevance_score": 0.72,
        "support": "fully_supported",
        "usefulness": "useful",
        "usefulness_reason": "...",
        "hallucination_retries": 0,
        "usefulness_retries": 1,
        "latency_ms": 4231.0,
        "retrieval_used": True,
        "web_search_used": False
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
| `chunk_size` | `700` | Chunk size for initial PDF splitting |
| `chunk_overlap` | `150` | Overlap between PDF chunks |

---

## Confidence Score Formula

```
confidence = (support_score × 0.4) + (usefulness_score × 0.4) + (relevance_score × 0.2)
```

Where:
- `support_score`: `1.0` (fully supported), `0.5` (partially), `0.0` (no support)
- `usefulness_score`: `1.0` (useful), `0.0` (not useful)
- `relevance_score`: average relevance score of retrieved documents

---

## Project Structure

```
precision_rag.ipynb     # Main notebook with full pipeline
.env                    # Environment variables (not committed)
wudc.pdf / uadc.pdf     # Example source documents (replace with your own)
README.md               # This file
```