"""
chunker.py — Split long documents into smaller, overlapping chunks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY DO WE CHUNK?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLMs have a fixed context window (token limit). A full document might be
thousands of tokens — too large to embed meaningfully or to pass to the LLM
all at once. Chunking lets us:

  1. Embed each small piece independently → precise semantic vectors
  2. Retrieve only the RELEVANT pieces at query time (not the whole document)
  3. Fit those pieces into the LLM's context window

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY OVERLAP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a key sentence sits right on the boundary between two chunks, a hard split
would cut it in half — losing context. Overlap ensures adjacent chunks share
some text, so boundary information is preserved in at least one chunk.

Example (chunk_size=20, chunk_overlap=5):
  Text:    "The quick brown fox jumps over the lazy dog"
  Chunk 1: "The quick brown fox"
  Chunk 2: "fox jumps over the"     ← 'fox' repeated from chunk 1
  Chunk 3: "the lazy dog"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS RecursiveCharacterTextSplitter?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
It tries to split on natural boundaries in order of preference:
  1. "\n\n"  — paragraph breaks (best: preserves semantic units)
  2. "\n"    — line breaks
  3. " "     — word boundaries
  4. ""      — character-level (last resort)

It only falls back to a finer separator when a chunk would exceed chunk_size.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings


def split_text(text: str) -> list[str]:
    """
    Split a document string into a list of overlapping text chunks.

    Args:
        text: The full document text to split.

    Returns:
        A list of string chunks ready for embedding.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,       # measure chunk size in characters
        is_separator_regex=False,  # treat separators as literal strings
    )

    chunks = splitter.split_text(text)
    return chunks
