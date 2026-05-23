"""
main.py — FastAPI application entry point.

This file:
  1. Creates the FastAPI app instance with metadata for auto-generated docs
  2. Registers a lifespan handler to initialize ChromaDB on startup
  3. Adds CORS middleware (so a browser-based frontend can call this API)
  4. Mounts the ingest and chat routers
  5. Provides health-check endpoints
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import chat, collections, ingest
from app.services.vector_store import get_client

STATIC_DIR = Path(__file__).parent.parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
# FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
# Code before `yield` runs on startup; code after runs on shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup and clean up on shutdown."""

    logger.info("🚀 Starting FastAPI RAG API...")

    # Initialize the ChromaDB client once at startup.
    # This creates the persist directory if it doesn't exist and warms the
    # HNSW index so the first query isn't slow.
    get_client()
    logger.info("✅ ChromaDB initialized")
    logger.info("📖 Swagger UI: http://localhost:8000/docs")
    logger.info("📖 ReDoc:      http://localhost:8000/redoc")

    yield  # ← the app handles requests here

    logger.info("🛑 Shutting down — goodbye!")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🔍 FastAPI RAG Template",
    description="""
## A Learning Template for Retrieval-Augmented Generation (RAG)

This API demonstrates the full RAG pipeline end-to-end:

### How RAG Works

```
INGESTION (offline):
  Document → Chunk → Embed (nomic-embed-text) → Store (ChromaDB)

QUERY (online):
  Question → Embed → Similarity Search → Context → LLM (llama3.2) → Answer
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/ingest` | POST | Upload a document → chunks it, embeds it, stores in ChromaDB |
| `/chat`   | POST | Ask a question → retrieves relevant chunks → LLM answers |
| `/health` | GET  | Check Ollama + ChromaDB status |

### Prerequisites

Make sure [Ollama](https://ollama.com) is installed and running, then pull the required models:

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # chat/generation model
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local development.
# In production, restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(collections.router)


# ── Static UI ─────────────────────────────────────────────────────────────────
# Serve static assets (CSS, JS, images) if you ever split them out.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the RAG Explorer web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["🩺 Health"], summary="Detailed health check")
async def health():
    """
    Return the health status of each dependency:
      - **api**    — always 'ok' if this endpoint responds
      - **chroma** — 'ok' if ChromaDB is accessible
      - **ollama** — 'ok' if Ollama server is reachable
    """
    settings = get_settings()
    status: dict = {"api": "ok", "chroma": "unknown", "ollama": "unknown"}

    # Check ChromaDB
    try:
        client = get_client()
        collections = client.list_collections()
        status["chroma"] = "ok"
        status["chroma_collections"] = [c.name for c in collections]
    except Exception as exc:
        status["chroma"] = f"error: {exc}"

    # Check Ollama
    try:
        models_response = ollama.list()
        model_names = [m["name"] for m in models_response.get("models", [])]
        status["ollama"] = "ok"
        status["ollama_models"] = model_names

        # Warn if required models are missing
        required = {settings.ollama_embed_model, settings.ollama_chat_model}
        # Normalize names: "llama3.2:latest" → "llama3.2"
        available = {name.split(":")[0] for name in model_names}
        missing = required - available
        if missing:
            status["ollama_warning"] = (
                f"Required model(s) not found: {missing}. "
                f"Run: ollama pull {' && ollama pull '.join(missing)}"
            )
    except Exception as exc:
        status["ollama"] = f"error: {exc} — Is Ollama running? (ollama serve)"

    return status
