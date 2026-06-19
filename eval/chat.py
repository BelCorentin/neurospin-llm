#!/usr/bin/env python3
"""
NeuroSpin Wiki RAG — Interactive Terminal Chat

A readline-enabled REPL that runs the full RAG pipeline locally.
Works over SSH without any browser.

Usage:
    source ingest/.venv/bin/activate
    python eval/chat.py

Commands in the chat:
    /quit or /exit or Ctrl-C   exit
    /clear                     clear the conversation history
    /sources                   toggle source display (default: on)
    /log                       show path to the conversation log file

Env vars:
    QDRANT_URL        (default: http://localhost:6333)
    VLLM_BASE_URL     (default: http://localhost:8000/v1)
    LLM_MODEL         (default: Qwen/Qwen2.5-7B-Instruct)
    EMBED_MODEL       (default: BAAI/bge-m3)
    RAG_TOP_K         (default: 5)
    RAG_LOG_FILE      (default: ~/.neurospin-rag-chat.log)
"""

import json
import os
import readline  # noqa: F401 — enables arrow-key history in input()
import sys
from datetime import datetime

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL    = os.environ.get("QDRANT_URL",          "http://localhost:6333")
QDRANT_PATH   = os.environ.get("QDRANT_PATH")  # if set, embedded on-disk mode (no server)
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL",       "http://localhost:8000/v1")
LLM_MODEL     = os.environ.get("LLM_MODEL",           "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL   = os.environ.get("EMBED_MODEL",         "BAAI/bge-m3")
COLLECTION    = os.environ.get("QDRANT_COLLECTION",   "neurospin_wiki")
TOP_K         = int(os.environ.get("RAG_TOP_K",       "5"))
LOG_FILE      = os.path.expanduser(
    os.environ.get("RAG_LOG_FILE", "~/.neurospin-rag-chat.log")
)


def _write_log(question: str, answer: str, sources: list[dict], lang: str) -> None:
    """Append one Q/A turn to LOG_FILE as a JSONL record."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lang": lang,
        "question": question,
        "answer": answer,
        "sources": [
            {"page": s["page_title"], "score": round(s["score"], 4)}
            for s in sources
        ],
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # non-fatal


SYSTEM_PROMPT = """\
You are a helpful assistant for the NeuroSpin neuroimaging research center.
Answer questions using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Cite the source page(s) you used at the end of your answer.
IMPORTANT: you MUST reply in the SAME language as the QUESTION.
If the question is in French, your entire answer must be in French.
If the question is in English, answer in English."""


def _detect_language(text: str) -> str:
    """Return 'fr' or 'en' using word-boundary regex + accented-char scoring."""
    import re
    t = text.lower()
    fr_words = [
        r"je", r"tu", r"il", r"elle", r"nous", r"vous", r"ils", r"elles",
        r"comment", r"pourquoi", r"depuis", r"chez", r"puis",
        r"qu(?:e|oi|el|elle|els|elles|')",
        r"est-ce", r"c'est", r"c'\u00e9tait",
        r"dans", r"avec", r"pour", r"les", r"des", r"une", r"qui",
        r"au\b", r"aux", r"du", r"le\b", r"la\b",
        r"est\b", r"sont", r"avoir", r"faire",
    ]
    score = sum(
        1 for w in fr_words
        if re.search(r"(?<![\w-])" + w + r"(?![\w-])", t)
    )
    score += min(len(re.findall(r"[éèêëàâùûîïôçœæ]", t)) * 2, 6)
    n = len(t.split())
    threshold = 1 if n <= 4 else (2 if n <= 9 else 3)
    return "fr" if score >= threshold else "en"

# ANSI colours
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"


# ── RAG helpers ───────────────────────────────────────────────────────────────

def retrieve(question: str, model: SentenceTransformer, qdrant: QdrantClient) -> list[dict]:
    vec = model.encode(question, normalize_embeddings=True).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=TOP_K,
        with_payload=True,
    )
    return [
        {
            "text": h.payload["text"],
            "page_title": h.payload.get("page_title", "?"),
            "source_file": h.payload.get("source_file", "?"),
            "score": h.score,
        }
        for h in hits
    ]


def stream_answer(question: str, chunks: list[dict], llm: OpenAI) -> tuple[str, str]:
    """Stream the LLM answer to stdout; return (full_text, lang)."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[{i}] Source: {c['page_title']}\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    lang = _detect_language(question)
    lang_instruction = (
        "Answer in French (the question is in French)."
        if lang == "fr"
        else "Answer in English (the question is in English)."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"INSTRUCTION: {lang_instruction}\n\n"
                f"QUESTION: {question}"
            ),
        },
    ]

    full = []
    print(f"\n{CYAN}{BOLD}Assistant:{RESET} ", end="", flush=True)
    for chunk in llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.1,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full.append(delta)
    print()  # newline after stream ends
    return "".join(full), lang


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Suppress HF Hub warning
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    print(f"\n{BOLD}NeuroSpin Wiki — RAG Chat{RESET}")
    print("Type your question and press Enter. '/quit' to exit, '/sources' to toggle sources.\n")
    print(f"{DIM}Logging to: {LOG_FILE}{RESET}\n")

    # Load embedding model
    print(f"{DIM}Loading embedding model '{EMBED_MODEL}'…{RESET}", end="", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"\r{DIM}Embedding model ready.{' '*30}{RESET}")

    # Connect to Qdrant
    qdrant = QdrantClient(path=QDRANT_PATH) if QDRANT_PATH else QdrantClient(url=QDRANT_URL)
    try:
        count = qdrant.count(collection_name=COLLECTION).count
        print(f"{DIM}Qdrant: {count} chunks in '{COLLECTION}'{RESET}")
    except Exception as e:
        where = QDRANT_PATH or QDRANT_URL
        print(f"{RED}ERROR: Qdrant not reachable at {where}: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Connect to vLLM
    llm = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    try:
        models = [m.id for m in llm.models.list().data]
        print(f"{DIM}vLLM:   {', '.join(models)}{RESET}")
    except Exception as e:
        print(f"{RED}ERROR: vLLM not reachable at {VLLM_BASE_URL}: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{DIM}{'─'*60}{RESET}")

    show_sources = True

    while True:
        # Prompt
        try:
            question = input(f"\n{BOLD}{YELLOW}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not question:
            continue

        # Commands
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"{DIM}Bye.{RESET}")
            break
        if question.lower() == "/clear":
            print("\033[2J\033[H", end="")  # clear screen
            continue
        if question.lower() == "/sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            print(f"{DIM}Source display: {state}{RESET}")
            continue
        if question.lower() == "/log":
            print(f"{DIM}Log file: {LOG_FILE}{RESET}")
            continue

        # Retrieve
        try:
            chunks = retrieve(question, embed_model, qdrant)
        except Exception as e:
            print(f"{RED}Retrieval error: {e}{RESET}")
            continue

        # Stream answer
        try:
            answer, lang = stream_answer(question, chunks, llm)
        except Exception as e:
            print(f"{RED}Generation error: {e}{RESET}")
            continue

        # Log conversation turn
        _write_log(question, answer, chunks, lang)

        # Sources
        if show_sources and chunks:
            print(f"\n{DIM}Sources:{RESET}")
            seen = set()
            for c in chunks:
                key = c["page_title"]
                if key not in seen:
                    seen.add(key)
                    print(f"{DIM}  • {c['page_title']}  (score={c['score']:.3f}){RESET}")


if __name__ == "__main__":
    main()
