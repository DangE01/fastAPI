"""
routers/chat.py — POST /chat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG PIPELINE — QUERY PHASE (Steps 5–8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [User question]
       │
       ▼
  [5. Embed the question]     ← same model used during ingestion (CRITICAL)
       │
       ▼
  [6. Similarity search]      ← find top-K closest chunks in ChromaDB
       │
       ▼
  [7. Build RAG prompt]       ← inject chunks as context (see services/llm.py)
       │
       ▼
  [8. LLM generates answer]   ← local Ollama llama3.2 reads context + answers
       │
       ▼
  [Return answer + sources]   ← show the user what evidence was used

The "sources" in the response are the actual chunks retrieved from ChromaDB.
Returning them lets you verify that the answer is grounded in your documents.
"""

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, Source
from app.services import embedder, llm, vector_store

router = APIRouter(prefix="/chat", tags=["💬 Chat"])


@router.post("/", response_model=ChatResponse, summary="Ask a question about your documents")
async def chat(request: ChatRequest):
    """
    ## 💬 Ask a Question (RAG Query)

    Given a question, this endpoint will:

    1. **Embed** the question into a vector (using `nomic-embed-text`)
    2. **Search** ChromaDB for the most semantically similar chunks
    3. **Build** a RAG prompt with those chunks as context
    4. **Generate** an answer using the local LLM (`llama3.2` via Ollama)
    5. **Return** the answer AND the source chunks used (for transparency)

    > **Make sure to ingest a document first** via `POST /ingest`.
    > Use the same `collection_name` you used during ingestion.
    """
    settings = get_settings()

    # ── Step 5: Embed the user's question ────────────────────────────────────
    # We MUST use the same embedding model that was used during ingestion.
    # If we used a different model, the query vector would live in a different
    # vector space and similarity search would return garbage results.
    query_embedding = embedder.embed_text(request.question)

    # ── Step 6: Retrieve the most relevant chunks ─────────────────────────────
    # ChromaDB computes cosine distance between the query vector and all stored
    # vectors, then returns the top_k closest matches.
    # Lower distance = higher relevance.
    try:
        collection = vector_store.get_or_create_collection(request.collection_name)
        count = collection.count()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing collection '{request.collection_name}': {exc}",
        ) from exc

    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Collection '{request.collection_name}' exists but is empty. "
                "Please ingest a document first via POST /ingest."
            ),
        )

    results = vector_store.query_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=min(settings.top_k_results, count),  # can't request more than we have
    )

    # ChromaDB returns list-of-lists because it supports batch queries.
    # Since we send a single query, we always take index [0].
    retrieved_docs: list[str] = results["documents"][0]
    distances: list[float] = results["distances"][0]
    metadatas: list[dict] = results["metadatas"][0]

    if not retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Try rephrasing your question.",
        )

    # ── Steps 7 & 8: Build prompt and generate answer ─────────────────────────
    # See services/llm.py for the full prompt template and explanation of
    # how injecting context "grounds" the LLM's response.
    answer = llm.generate_answer(
        question=request.question,
        context_chunks=retrieved_docs,
    )

    # ── Build the response with sources ───────────────────────────────────────
    # Including sources lets the user verify that the answer came from their
    # actual documents, not from the LLM's training data.
    # This is a RAG best practice called "citation / attribution".
    sources = [
        Source(content=doc, distance=dist, metadata=meta)
        for doc, dist, meta in zip(retrieved_docs, distances, metadatas)
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        model_used=settings.ollama_chat_model,
    )
