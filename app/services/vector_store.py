"""
vector_store.py — A thin wrapper around ChromaDB for storing and querying embeddings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A VECTOR STORE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A vector store is a database optimized for storing and searching high-dimensional
vectors (embeddings). Unlike a regular database that searches by exact match or
SQL filters, a vector store finds records whose vectors are CLOSEST to a query
vector (approximate nearest neighbor search).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CHROMADB?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ChromaDB is:
  - Fully local (no cloud, no API keys, no cost)
  - Easy to set up (pip install chromadb — that's it)
  - Stores vectors on disk (PersistentClient) so data survives restarts
  - Uses HNSW (Hierarchical Navigable Small World) index for fast ANN search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS COSINE SIMILARITY / DISTANCE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We use cosine distance as our similarity metric:
  - cosine distance = 1 - cosine_similarity
  - Range: 0 (identical direction) → 2 (opposite direction)
  - For text, relevant results typically have distance < 0.3

ChromaDB returns results sorted from smallest distance (most relevant) to
largest (least relevant).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A ChromaDB "collection" is like a table in a relational database. Each
collection groups related embeddings. You might have:
  - One collection per document ("finance_report_2024")
  - One collection per topic ("product_manuals")
  - A single default collection for everything
"""

import chromadb
from app.config import get_settings

# Module-level singleton — created once, reused across all requests.
# Using a singleton avoids re-opening the database on every request.
_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """
    Return the shared ChromaDB PersistentClient, creating it if needed.

    PersistentClient saves data to disk at chroma_persist_dir so your
    embeddings survive server restarts.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """
    Get a named collection, or create it if it doesn't exist yet.

    Args:
        name: Collection name (e.g. "default", "my_docs"). Must be alphanumeric
              with underscores/hyphens — no spaces.

    Returns:
        A ChromaDB Collection object with cosine distance configured.
    """
    client = get_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # use cosine distance for text
    )
    return collection


def add_documents(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Store document chunks + their embeddings in a ChromaDB collection.

    Each record stored in ChromaDB has four parts:
      - id:        Unique string identifier (we use filename + chunk index)
      - embedding: The vector representation of the chunk
      - document:  The raw text (returned in query results so we can show it)
      - metadata:  Arbitrary key-value pairs (source filename, chunk index, etc.)

    Args:
        collection: The target ChromaDB collection.
        ids:        Unique IDs, one per chunk.
        embeddings: Embedding vectors, one per chunk.
        documents:  Raw text strings, one per chunk.
        metadatas:  Metadata dicts, one per chunk.
    """
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> dict:
    """
    Find the top_k chunks most similar to the query embedding.

    HOW IT WORKS:
      1. ChromaDB uses its HNSW index to efficiently search all stored vectors.
      2. It computes cosine distance between the query vector and every stored vector.
      3. It returns the top_k closest matches, sorted by ascending distance.

    Args:
        collection:      The ChromaDB collection to search.
        query_embedding: The embedded user question (from embedder.embed_text).
        top_k:           Number of results to return.

    Returns:
        A dict with keys: 'ids', 'documents', 'distances', 'metadatas'.
        Each value is a list-of-lists because ChromaDB supports batch queries;
        we always query one at a time, so we use results["documents"][0], etc.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    return results
