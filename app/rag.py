"""
NeuroSpin Wiki RAG — Core Pipeline

query_rag(question) → (answer_stream, sources)

Steps:
  1. Embed the user question with bge-m3 (loaded once at module import).
  2. Search Qdrant for the top-K most relevant wiki chunks.
  3. Build a grounded system prompt with those chunks as context.
  4. Stream a response from the vLLM endpoint (OpenAI-compatible API).
  5. Return the async stream and a list of source dicts for citation display.
"""

import os
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Configuration (all overridable via environment variables) ─────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "neurospin_wiki")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

SYSTEM_PROMPT = """\
You are a helpful assistant for the NeuroSpin neuroimaging research wiki.

Rules:
- Answer ONLY using the information provided in the CONTEXT section below.
- If the answer cannot be found in the context, say exactly:
  "Je ne trouve pas cette information dans le wiki NeuroSpin." (for French questions)
  or "I cannot find this information in the NeuroSpin wiki." (for English questions)
- Do NOT invent commands, file paths, email addresses, or procedures.
- Always cite the source page(s) you used at the end of your answer.
- Respond in the same language as the user's question (English or French).
- Be concise but complete. Use bullet points or numbered steps where appropriate.
"""

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_embed_model: SentenceTransformer | None = None
_qdrant: QdrantClient | None = None
_openai: AsyncOpenAI | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed",  # vLLM doesn't require a key
        )
    return _openai


# ── Source dataclass ──────────────────────────────────────────────────────────

@dataclass
class Source:
    page_title: str
    source_file: str
    chunk_index: int
    score: float
    excerpt: str  # first 300 chars of the chunk


# ── Main RAG function ─────────────────────────────────────────────────────────

async def query_rag(question: str) -> tuple[AsyncIterator[str], list[Source]]:
    """
    Embed the question, retrieve relevant chunks, and stream an LLM answer.

    Returns:
        (token_stream, sources) — consume the stream to get the answer tokens,
        and display sources as citation cards.
    """
    # 1. Embed the question
    model = _get_embed_model()
    q_vector = model.encode(question, normalize_embeddings=True).tolist()

    # 2. Search Qdrant
    qdrant = _get_qdrant()
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vector,
        limit=TOP_K,
        with_payload=True,
    )

    sources = [
        Source(
            page_title=h.payload["page_title"],
            source_file=h.payload["source_file"],
            chunk_index=h.payload["chunk_index"],
            score=h.score,
            excerpt=h.payload["text"][:300],
        )
        for h in hits
    ]

    # 3. Build context block
    context_parts = []
    for i, hit in enumerate(hits):
        context_parts.append(
            f"[Source {i+1}: {hit.payload['page_title']} "
            f"(file: {hit.payload['source_file']}, chunk {hit.payload['chunk_index']})]\n"
            f"{hit.payload['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    # 4. Call vLLM with streaming
    client = _get_openai()
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )

    async def token_generator() -> AsyncIterator[str]:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return token_generator(), sources
