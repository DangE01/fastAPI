"""
embedder.py — Convert text into vector embeddings using Ollama (local, free).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AN EMBEDDING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An embedding is a list of floating-point numbers (a vector) that represents
the *meaning* of a piece of text in high-dimensional space.

  "The dog chased the ball"  →  [0.21, -0.54, 0.88, ...]  (768 numbers)
  "A puppy ran after a toy"  →  [0.20, -0.51, 0.85, ...]  (very similar!)
  "Interest rates rose 2%"   →  [-0.33, 0.71, -0.12, ...] (very different)

Key insight: semantically similar texts produce vectors that are CLOSE
to each other in vector space, even if they use different words.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY DOES THIS MATTER FOR RAG?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When a user asks a question, we embed the question into a vector, then find
document chunks whose vectors are closest to it. This is "semantic search" —
it finds relevant content even when the exact words don't match.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL: nomic-embed-text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nomic-embed-text is a high-quality open-source embedding model that runs
locally via Ollama. It produces 768-dimensional vectors.

Pull it once with: ollama pull nomic-embed-text

IMPORTANT: You must use the SAME embedding model for both ingestion and
querying. Mixing models will produce vectors in different spaces, making
similarity search meaningless.
"""

import ollama
from app.config import get_settings


def embed_text(text: str) -> list[float]:
    """
    Embed a single string into a vector using the configured Ollama model.

    Args:
        text: The text to embed (a document chunk or a user query).

    Returns:
        A list of floats representing the text in embedding space.
    """
    settings = get_settings()

    response = ollama.embeddings(
        model=settings.ollama_embed_model,
        prompt=text,
    )

    # The response dict contains one key: "embedding"
    # which is a list[float] of length 768 for nomic-embed-text
    return response["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings one by one and return a list of vectors.

    Note: Ollama's Python client currently handles one embedding per call.
    For production scale, consider batching or async parallel calls.

    Args:
        texts: A list of document chunks to embed.

    Returns:
        A list of embedding vectors, one per input text.
    """
    return [embed_text(text) for text in texts]
