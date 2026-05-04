#!/usr/bin/env python3
"""
NeuroSpin Wiki RAG — Terminal POC Evaluation

Tests the full RAG pipeline end-to-end: embed question → Qdrant search →
LLM answer → keyword check.

Requirements: run from inside the ingest venv (same as ingestion):
    source ingest/.venv/bin/activate
    python eval/eval.py

Override service URLs with env vars if needed:
    QDRANT_URL=http://localhost:6333
    VLLM_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    EMBED_MODEL=BAAI/bge-m3
    RAG_TOP_K=5
"""

import os
import sys
import textwrap
import time
from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL     = os.environ.get("QDRANT_URL",     "http://localhost:6333")
VLLM_BASE_URL  = os.environ.get("VLLM_BASE_URL",  "http://localhost:8000/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL    = os.environ.get("EMBED_MODEL",     "BAAI/bge-m3")
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "neurospin_wiki")
TOP_K          = int(os.environ.get("RAG_TOP_K",   "5"))

SYSTEM_PROMPT = """\
You are a helpful assistant for the NeuroSpin neuroimaging research center.
Answer questions using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Cite the source page(s) you used.
Be concise (3-5 sentences maximum).
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
    n = len(t.split())
    threshold = 1 if n <= 4 else (2 if n <= 9 else 3)
    return "fr" if score >= threshold else "en"


# ── Test cases ────────────────────────────────────────────────────────────────
# Each case: question, list of keyword strings that MUST appear in the answer
# (case-insensitive). All keywords must match to pass.

@dataclass
class TestCase:
    question: str
    must_contain: list[str] = field(default_factory=list)
    description: str = ""


TEST_CASES: list[TestCase] = [
    TestCase(
        description="PROPixx projector luminosity",
        question=(
            "What luminosity level should I use for the PROPixx video projector "
            "at NeuroSpin, and why should I avoid 100% luminosity?"
        ),
        must_contain=["12.5"],
    ),
    TestCase(
        description="Best fMRI resolution at 3T",
        question=(
            "What is the recommended voxel resolution for general-purpose fMRI "
            "at the 3T scanner at NeuroSpin?"
        ),
        must_contain=["1.75"],
    ),
    TestCase(
        description="BIDS definition",
        question="What is BIDS in the context of neuroimaging data?",
        must_contain=["brain imaging data structure"],
    ),
    TestCase(
        description="Auditory stimulation system",
        question=(
            "Which auditory stimulation system is available at both the 3T and 7T "
            "scanners at NeuroSpin?"
        ),
        must_contain=["optoacoustic", "optoactive"],  # any one of these is fine
    ),
    TestCase(
        description="FreeSurfer capabilities (EN)",
        question="What can FreeSurfer do? List its main features.",
        must_contain=["cortical", "segmentation"],
    ),
    TestCase(
        description="Qu'est-ce que BIDS (FR)",
        question=(
            "Qu'est-ce que BIDS et à quoi sert ce standard dans le domaine "
            "de la neuroimagerie ?"
        ),
        must_contain=["brain imaging data structure", "bids"],
    ),
]


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def retrieve(question: str, model: SentenceTransformer, qdrant: QdrantClient) -> list[dict]:
    """Embed question and retrieve top-K chunks from Qdrant."""
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


def generate(question: str, chunks: list[dict], llm: OpenAI) -> str:
    """Build RAG prompt and call the LLM (non-streaming for eval)."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {c['page_title']}\n{c['text']}"
        )
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
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def check_keywords(answer: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    """Return (pass, missing_keywords). Uses OR within each keyword item if / present."""
    answer_lower = answer.lower()
    missing = []
    for kw in must_contain:
        # Support OR: "optoacoustic|optoactive" → at least one must match
        variants = [v.strip() for v in kw.replace("|", "/").split("/")]
        if not any(v in answer_lower for v in variants):
            missing.append(kw)
    return (len(missing) == 0), missing


# ── Pretty print ──────────────────────────────────────────────────────────────

WIDTH = 80

def hr(char="─"):
    return char * WIDTH

def wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix,
                         subsequent_indent=prefix)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(hr("═"))
    print("  NeuroSpin Wiki RAG — Terminal Evaluation")
    print(hr("═"))
    print()

    # ── Load models & clients ──────────────────────────────────────────────
    print(f"Loading embedding model '{EMBED_MODEL}'…")
    t0 = time.time()
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"  ready in {time.time()-t0:.1f}s")
    print()

    qdrant = QdrantClient(url=QDRANT_URL)
    try:
        chunk_count = qdrant.count(collection_name=COLLECTION).count
        print(f"Qdrant: collection '{COLLECTION}' — {chunk_count} chunks")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant at {QDRANT_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    try:
        models = [m.id for m in llm.models.list().data]
        print(f"vLLM : {', '.join(models)}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM at {VLLM_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    print()

    # ── Run test cases ─────────────────────────────────────────────────────
    results: list[bool] = []

    for idx, tc in enumerate(TEST_CASES, 1):
        print(hr())
        print(f"  Test {idx}/{len(TEST_CASES)}: {tc.description}")
        print(hr())
        print()
        print("  QUESTION:")
        print(wrap(tc.question))
        print()

        # Retrieve
        t0 = time.time()
        chunks = retrieve(tc.question, embed_model, qdrant)
        retrieve_ms = (time.time() - t0) * 1000

        print(f"  RETRIEVED ({TOP_K} chunks, {retrieve_ms:.0f} ms):")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] {c['page_title']}  (score={c['score']:.3f})")
        print()

        # Generate
        t0 = time.time()
        answer = generate(tc.question, chunks, llm)
        generate_ms = (time.time() - t0) * 1000

        print(f"  ANSWER ({generate_ms:.0f} ms):")
        for line in answer.splitlines():
            print(wrap(line if line.strip() else " "))
        print()

        # Check
        passed, missing = check_keywords(answer, tc.must_contain)
        results.append(passed)

        if passed:
            print("  ✓ PASS  — all expected keywords found")
        else:
            print(f"  ✗ FAIL  — missing keywords: {missing}")
        print()

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print(hr("═"))
    print(f"  Results: {n_pass}/{n_total} tests passed")
    print(hr("═"))

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
