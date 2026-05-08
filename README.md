# Retrieval-Augmented Generation (RAG) — A Quick Introduction

## What is RAG?

Imagine you have a company handbook, a research paper, or your own notes.
A standard LLM doesn't know about your specific documents — it only knows
what it was trained on. RAG solves this:

**RAG = Give the LLM your documents as context at query time.**

Instead of retraining the model (expensive), you:
1. Break your documents into chunks
2. Convert each chunk into a vector (embedding)
3. Store the vectors in a searchable database
4. When a question arrives, find the most relevant chunks
5. Include those chunks in the LLM's prompt

The LLM answers based on YOUR documents, not just its training data.

---

## The Two Phases of RAG

### Phase 1 — Ingestion (Offline)

```
[Your Document]
     │
     ▼
[Chunker]         → Splits into ~500 char overlapping pieces
     │
     ▼
[Embedding Model] → Converts each chunk into a vector (list of numbers)
(nomic-embed-text via Ollama)
     │
     ▼
[Vector Store]    → Stores vectors + raw text on disk
(ChromaDB)
```

### Phase 2 — Query (Online, per user request)

```
[User Question]
     │
     ▼
[Embedding Model] → Converts question into a vector
(same model as ingestion!)
     │
     ▼
[Vector Store]    → Finds the most similar document chunks (similarity search)
     │
     ▼
[LLM Prompt]      → "Here are relevant passages: [...]. Answer: {question}"
     │
     ▼
[LLM]             → Generates a grounded answer
(llama3.2 via Ollama)
     │
     ▼
[Response]        → Answer + sources (which chunks were used)
```

---

## What's in This Template

```
fastAPI/
├── app/
│   ├── main.py               ← FastAPI app, lifespan, health checks
│   ├── config.py             ← All settings via .env or environment variables
│   ├── models.py             ← Pydantic request/response schemas
│   ├── routers/
│   │   ├── ingest.py         ← POST /ingest — ingestion pipeline
│   │   └── chat.py           ← POST /chat   — query pipeline
│   └── services/
│       ├── chunker.py        ← Text splitting (why & how)
│       ├── embedder.py       ← Embedding generation (why & how)
│       ├── vector_store.py   ← ChromaDB wrapper (why & how)
│       └── llm.py            ← Prompt building + LLM call
├── data/
│   └── sample.txt            ← Sample document to test with
├── .env.example              ← Copy to .env to configure the app
├── requirements.txt
└── README.md                 ← This file
```

Every service file contains extensive comments explaining the **why** behind
each design decision — great for learning.

---

## Quick Start

### 1. Install Ollama

Download and install from [https://ollama.com](https://ollama.com), then:

```bash
# Start the Ollama server (runs in background)
ollama serve

# Pull the required models (one-time download)
ollama pull nomic-embed-text   # ~270 MB — the embedding model
ollama pull llama3.2           # ~2 GB   — the chat model
```

### 2. Install uv (if you haven't already)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Set Up the Python Environment

`uv` replaces `pip` + `venv` in one command. It reads `pyproject.toml` and
creates an isolated virtual environment automatically.

```bash
# Install all dependencies and create .venv in one step
uv sync

# To also install dev dependencies (pytest, httpx):
uv sync --dev
```

### 4. Configure the App

```bash
# Copy the example config (defaults work out of the box)
cp .env.example .env
```

### 5. Run the Server

```bash
uv run uvicorn app.main:app --reload
```

You should see:
```
✅ ChromaDB initialized
📖 Swagger UI: http://localhost:8000/docs
```

### 5. Check the Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected output (once models are pulled):
```json
{
  "api": "ok",
  "chroma": "ok",
  "ollama": "ok",
  "ollama_models": ["nomic-embed-text:latest", "llama3.2:latest"]
}
```

---

## Using the API

### Step A — Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@data/sample.txt" \
  -F "collection_name=default"
```

Response:
```json
{
  "message": "Successfully ingested 'sample.txt'",
  "chunks_embedded": 8,
  "collection_name": "default"
}
```

### Step B — Ask a Question

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG and why is it useful?",
    "collection_name": "default"
  }'
```

Response:
```json
{
  "answer": "RAG, or Retrieval-Augmented Generation, is a technique that...",
  "sources": [
    {
      "content": "RAG combines retrieval systems with generative models...",
      "distance": 0.12,
      "metadata": { "source": "sample.txt", "chunk_index": 2 }
    }
  ],
  "model_used": "llama3.2"
}
```

### Using Swagger UI (Recommended for Learning)

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
The interactive UI lets you upload files and run queries without writing any curl commands.

---

## Tuning Parameters

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Larger = more context per chunk, less precise retrieval |
| `CHUNK_OVERLAP` | 50 | Larger = better boundary handling, more storage |
| `TOP_K_RESULTS` | 4 | More results = more context for LLM, but also more noise |

Edit `.env` and restart the server to apply changes.

---

## How to Extend This Template

- **Add a new file type** → extend `_extract_pdf_text` in `routers/ingest.py`
- **Change the LLM** → update `OLLAMA_CHAT_MODEL` in `.env` (e.g., `mistral`, `phi3`)
- **Add metadata filtering** → use ChromaDB's `where` parameter in `query_collection`
- **Add streaming** → replace `ollama.chat()` with `ollama.chat(stream=True)` and use `StreamingResponse`
- **Add a frontend** → the API has CORS enabled; just call it from any JS framework

---

## Key Concepts Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A list of numbers (vector) representing the semantic meaning of text |
| **Chunk** | A small piece of a document, sized to fit within context limits |
| **Vector Store** | A database optimized for similarity search on embeddings |
| **Cosine Distance** | How different two vectors are (0 = identical, 1 = unrelated) |
| **Top-K Retrieval** | Fetching the K most similar chunks to a query vector |
| **Grounding** | Constraining the LLM to answer only from provided context |
| **RAG** | Retrieval-Augmented Generation — the full pipeline above |
