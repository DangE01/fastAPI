"""
models.py — Pydantic request/response schemas for the RAG API.

FastAPI uses these schemas to:
  - Validate incoming request bodies (auto 422 on bad data)
  - Serialize outgoing responses
  - Generate the OpenAPI/Swagger documentation automatically
"""

from pydantic import BaseModel, Field
from typing import List


# ── Collections ───────────────────────────────────────────────────────────────

class CollectionInfo(BaseModel):
    """Summary of a single ChromaDB collection."""

    name: str = Field(..., description="Collection name.")
    count: int = Field(..., description="Number of embedded chunks stored.")


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned after a document is successfully ingested."""

    message: str = Field(..., description="Human-readable status message.")
    chunks_embedded: int = Field(
        ..., description="Number of text chunks that were embedded and stored."
    )
    collection_name: str = Field(
        ..., description="The ChromaDB collection where chunks were stored."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Successfully ingested 'sample.txt'",
                "chunks_embedded": 12,
                "collection_name": "default",
            }
        }
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    """
    A single retrieved chunk that the LLM used to generate its answer.

    Returning sources gives users transparency into WHAT context was used.
    This is a RAG best-practice — it makes the system auditable.
    """

    content: str = Field(..., description="The raw text of the retrieved chunk.")
    distance: float = Field(
        ...,
        description=(
            "Cosine distance from the query (0 = identical, 1 = unrelated). "
            "Lower is more relevant."
        ),
    )
    metadata: dict = Field(
        ..., description="Source file name and chunk index for traceability."
    )


class ChatRequest(BaseModel):
    """Payload for the /chat endpoint."""

    question: str = Field(..., description="The question to answer from your documents.")
    collection_name: str = Field(
        default="default",
        description="Which ChromaDB collection to search. Must match what was used during ingestion.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is RAG and why is it useful?",
                "collection_name": "default",
            }
        }
    }


class ChatResponse(BaseModel):
    """Returned after the RAG pipeline processes a question."""

    answer: str = Field(..., description="The LLM-generated answer.")
    sources: List[Source] = Field(
        ..., description="The chunks retrieved from ChromaDB that informed the answer."
    )
    model_used: str = Field(..., description="The Ollama model that generated the answer.")
