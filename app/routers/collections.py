"""
routers/collections.py — GET /collections

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS TEACHES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A real RAG app manages MULTIPLE collections — one per document set,
topic, or user. This endpoint lets the frontend discover what collections
already exist without hardcoding names.

This is the foundation of multi-document RAG:
  - Ingest "finance_docs" into collection "finance"
  - Ingest "legal_docs"   into collection "legal"
  - Query each independently so contexts don't bleed into each other
"""

from fastapi import APIRouter

from app.models import CollectionInfo
from app.services.vector_store import get_client

router = APIRouter(prefix="/collections", tags=["📚 Collections"])


@router.get("/", response_model=list[CollectionInfo], summary="List all collections")
async def list_collections():
    """
    ## List All ChromaDB Collections

    Returns every collection that exists in the local vector store,
    along with the number of chunks stored in each.

    The UI uses this to populate the collection picker — no hardcoded names needed.
    """
    client = get_client()
    cols = client.list_collections()

    # c.count() queries ChromaDB for the number of embedded chunks in that collection.
    return [CollectionInfo(name=c.name, count=c.count()) for c in cols]


@router.delete("/{name}", summary="Delete a collection")
async def delete_collection(name: str):
    """
    ## Delete a Collection

    Permanently removes a ChromaDB collection and all its embeddings.
    Useful for resetting a document set without restarting the server.

    > **Warning**: This cannot be undone. Re-ingest your documents to rebuild it.
    """
    client = get_client()
    client.delete_collection(name=name)
    return {"message": f"Collection '{name}' deleted successfully."}
