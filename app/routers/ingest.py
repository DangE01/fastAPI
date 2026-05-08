"""
routers/ingest.py — POST /ingest

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG PIPELINE — INGESTION PHASE (Steps 1–4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Upload file]
       │
       ▼
  [1. Read raw text]          ← extract text from .txt or .pdf
       │
       ▼
  [2. Chunk the text]         ← split into overlapping pieces (see services/chunker.py)
       │
       ▼
  [3. Embed each chunk]       ← convert text → vector (see services/embedder.py)
       │
       ▼
  [4. Store in ChromaDB]      ← persist vectors + raw text (see services/vector_store.py)

After ingestion, the document is "indexed" and ready for semantic search.
"""

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import IngestResponse
from app.services import chunker, embedder, vector_store

router = APIRouter(prefix="/ingest", tags=["📥 Ingestion"])


@router.post("/", response_model=IngestResponse, summary="Ingest a document")
async def ingest_document(
    file: UploadFile = File(
        ...,
        description="A .txt or .pdf file to ingest into the vector store.",
    ),
    collection_name: str = Form(
        default="default",
        description=(
            "Name of the ChromaDB collection to store chunks in. "
            "Use the same name when querying via /chat."
        ),
    ),
):
    """
    ## 📥 Ingest a Document

    Upload a `.txt` or `.pdf` file. This endpoint will:

    1. **Extract** raw text from the file
    2. **Chunk** the text into overlapping pieces
    3. **Embed** each chunk using `nomic-embed-text` (local via Ollama)
    4. **Store** the chunks + embeddings in ChromaDB

    After ingestion, query the document via `POST /chat`.

    > **Tip**: Use the `collection_name` field to organise different documents
    > into separate namespaces (e.g. `"finance"`, `"legal"`, `"default"`).
    """

    # ── Step 1: Read raw text ─────────────────────────────────────────────────
    raw_bytes = await file.read()

    filename = file.filename or "unknown"

    if filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="replace")
    elif filename.endswith(".pdf"):
        text = _extract_pdf_text(raw_bytes, filename)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{filename}'. Only .txt and .pdf are accepted.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file appears to be empty or contains no extractable text.",
        )

    # ── Step 2: Chunk the text ────────────────────────────────────────────────
    # See services/chunker.py for a detailed explanation of why we chunk
    # and how RecursiveCharacterTextSplitter works.
    chunks = chunker.split_text(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not produce any text chunks. The file may be too short.",
        )

    # ── Step 3: Embed each chunk ──────────────────────────────────────────────
    # This calls Ollama locally — no internet required.
    # See services/embedder.py for what embeddings are and why they matter.
    embeddings = embedder.embed_batch(chunks)

    # ── Step 4: Store in ChromaDB ─────────────────────────────────────────────
    # We give each chunk a unique ID so we can upsert safely.
    # Format: "<basename>-chunk-<index>-<random_suffix>"
    collection = vector_store.get_or_create_collection(collection_name)
    base_name = filename.rsplit(".", 1)[0]
    ids = [f"{base_name}-chunk-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    vector_store.add_documents(
        collection=collection,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return IngestResponse(
        message=f"Successfully ingested '{filename}'",
        chunks_embedded=len(chunks),
        collection_name=collection_name,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_pdf_text(content: bytes, filename: str) -> str:
    """
    Extract plain text from PDF bytes using pypdf.

    pypdf reads each page and concatenates the text. Quality depends on
    whether the PDF has a text layer (scanned PDFs without OCR will be empty).
    """
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF '{filename}': {exc}",
        ) from exc
