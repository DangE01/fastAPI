"""
config.py — Central configuration using Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
Pydantic Settings will automatically read from the .env file at startup.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama ───────────────────────────────────────────────────────────────
    # Ollama runs a local server at port 11434 by default.
    # The embed model converts text → vectors (numbers).
    # The chat model generates natural language answers.
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"  # pull: ollama pull nomic-embed-text
    ollama_chat_model: str = "llama3.2"            # pull: ollama pull llama3.2

    # ── Chunking ─────────────────────────────────────────────────────────────
    # chunk_size:    max characters per chunk (not tokens!)
    # chunk_overlap: characters shared between adjacent chunks so context
    #                near chunk boundaries isn't lost
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ────────────────────────────────────────────────────────────
    # How many chunks to retrieve from ChromaDB per query.
    # Higher = more context for the LLM, but also more noise.
    top_k_results: int = 4

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    # ChromaDB will persist its data to this local directory.
    # This means your embeddings survive server restarts.
    chroma_persist_dir: str = "./chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    @lru_cache ensures we only read the .env file once per process,
    not on every request. This is a common FastAPI pattern.
    """
    return Settings()
