"""
llm.py — Build RAG prompts and call the local Ollama LLM to generate answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE RAG GENERATION STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
By the time we reach this service, we've already:
  1. Embedded the user's question
  2. Retrieved the top-K most relevant chunks from ChromaDB

Now we:
  3. Inject those chunks as "context" into an LLM prompt
  4. Ask the LLM to answer the question using ONLY that context

This "grounding" is the key idea of RAG: the LLM can't hallucinate facts it
doesn't have because we explicitly provide the relevant evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We use a two-message format (system + user):

  [SYSTEM]  — Tells the model how to behave: stay grounded, don't make things up.
  [USER]    — The retrieved context + the user's actual question.

Keeping system instructions separate from user content is a best practice for
instruction-tuned models (like llama3.2).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL: llama3.2 (via Ollama)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
llama3.2 is Meta's open-source instruction-tuned model. It runs fully locally
via Ollama — no internet connection needed after the initial `ollama pull`.

Pull it once with: ollama pull llama3.2
"""

from collections.abc import AsyncGenerator

import ollama
from app.config import get_settings

# ── System Prompt ──────────────────────────────────────────────────────────────
# The system prompt sets the LLM's behavior for the entire conversation.
# "Only use the provided context" is the crucial RAG constraint — it prevents
# the model from mixing in its training knowledge and potentially hallucinating.

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY \
on the provided context passages.

Rules:
- Answer ONLY from the context provided below.
- If the answer is not in the context, say: "I don't have enough information \
to answer that from the provided documents."
- Do NOT make up information or use knowledge outside the context.
- Be concise, accurate, and cite which part of the context supports your answer \
when possible.
"""


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Assemble the user-turn prompt by injecting retrieved chunks as context.

    We separate chunks with a visible delimiter ("---") so the LLM can tell
    where one chunk ends and another begins.

    Args:
        question:       The user's original question.
        context_chunks: List of retrieved document chunks from ChromaDB.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    # Join chunks with a separator for readability
    formatted_context = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""Use the following context passages to answer the question.

CONTEXT:
{formatted_context}

QUESTION: {question}

ANSWER:"""

    return prompt


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generate an answer by calling the local Ollama LLM with the RAG prompt.

    Args:
        question:       The user's question.
        context_chunks: Retrieved document chunks to use as context.

    Returns:
        The LLM's generated answer as a string.
    """
    settings = get_settings()
    prompt = build_rag_prompt(question, context_chunks)

    response = ollama.chat(
        model=settings.ollama_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    # response["message"]["content"] is the assistant's reply text
    return response["message"]["content"]


async def generate_answer_stream(
    question: str, context_chunks: list[str]
) -> AsyncGenerator[str, None]:
    """
    Stream an answer token-by-token from Ollama using an async generator.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    HOW OLLAMA STREAMING WORKS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    When stream=True, ollama.chat() returns an iterator of partial response
    objects. Each chunk has a 'message' dict with a 'content' key containing
    the next token(s). We yield each token so the caller can forward it to
    the client immediately — no waiting for the full response.

    This is an *async* generator (uses `yield`) so FastAPI's StreamingResponse
    can await each token without blocking the event loop.

    Args:
        question:       The user's question.
        context_chunks: Retrieved document chunks to use as context.

    Yields:
        Individual token strings as they are produced by the LLM.
    """
    settings = get_settings()
    prompt = build_rag_prompt(question, context_chunks)

    # stream=True makes Ollama return tokens incrementally.
    # The SDK returns a synchronous iterator, so we wrap it to stay async-friendly.
    stream = ollama.chat(
        model=settings.ollama_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:  # skip empty tokens
            yield token
