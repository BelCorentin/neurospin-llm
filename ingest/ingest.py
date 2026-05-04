#!/usr/bin/env python3
"""
NeuroSpin Wiki RAG — Ingestion Pipeline

Reads PmWiki flat-file pages from wiki.d/, parses and cleans them, embeds with
BAAI/bge-m3, and uploads to a local Qdrant collection called 'neurospin_wiki'.

PmWiki file format (key=value, one per line, URL-encoded values):
  name=Main.SomePage
  text=URL-encoded wiki markup (current revision only)
  diff:...:...:=...  (edit history — skipped)

Usage:
    python ingest.py                                      # ingest data/mock/*.txt
    python ingest.py --data-dir /path/to/wiki.d           # ingest real wiki
    python ingest.py --data-dir /path/to/wiki.d --reset   # reset collection first
    python ingest.py --skip-recent-changes                # default: skip *RecentChanges
"""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ── Configuration ────────────────────────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "neurospin_wiki")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
EMBED_DIM = 1024  # bge-m3 output dimension

# Tokeniser used only for counting; the actual embedder does its own tokenisation
_tokeniser = tiktoken.get_encoding("cl100k_base")

# Pages to skip (administrative / noisy)
SKIP_SUFFIXES = ("RecentChanges", "GroupAttributes", "GroupFooter", "GroupHeader")


# ── PmWiki Parser ─────────────────────────────────────────────────────────────

def parse_pmwiki(raw: str) -> tuple[str, str, str] | None:
    """
    Parse a PmWiki flat-file page.

    Returns (page_name, page_title, clean_text) or None if the page should be
    skipped (administrative pages, empty pages, or pure diff-only files).

    PmWiki stores everything URL-encoded. The `text=` key holds the current
    revision content. `diff:...:...:=` keys hold edit history — ignored.
    """
    # Extract key=value pairs (values may span to end of line)
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        # Skip diff history lines (format: diff:timestamp:timestamp:=...)
        if re.match(r"diff:", line):
            continue
        # Skip author/host history lines
        if re.match(r"(author|host):\d+", line):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        key = line[:eq]
        value = line[eq + 1 :]
        fields[key] = value

    page_name = fields.get("name", "")
    raw_text = fields.get("text", "")

    if not raw_text or not page_name:
        return None

    # Skip administrative pages
    for suffix in SKIP_SUFFIXES:
        if page_name.endswith(suffix):
            return None

    # URL-decode the text (%0a → \n, %25 → %, etc.)
    text = unquote(raw_text, encoding="utf-8", errors="replace")

    # Build a human-readable title from the page name (e.g. Main.GettingStarted)
    parts = page_name.split(".", 1)
    if len(parts) == 2:
        group, name = parts
        # CamelCase → "Camel Case"
        name_spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
        name_spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name_spaced)
        page_title = f"{group} / {name_spaced}"
    else:
        page_title = page_name

    # ── Clean PmWiki markup ────────────────────────────────────────────────
    # Remove [@...@] code-block markers but keep the code content
    text = re.sub(r"\[@\s*", "", text)
    text = re.sub(r"\s*@\]", "", text)

    # Headings: !! Heading → Heading
    text = re.sub(r"^!{1,6}\s*", "", text, flags=re.MULTILINE)

    # [[links|display]] → display; [[links]] → links
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)

    # External links [https://... display] → display; [https://...] → remove
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]?", "", text)

    # Image/attachment references: Attach:Group/file.png or %width=...%
    text = re.sub(r"Attach:[^\s]+", "", text)
    text = re.sub(r"%[a-zA-Z]+=[^%]*%", "", text)

    # Bold/italic/underline markup
    text = re.sub(r"'{2,3}", "", text)    # '' or '''
    text = re.sub(r"_{2}", "", text)      # __underline__

    # Tables: keep cell content, strip | delimiters
    text = re.sub(r"(?m)^\|\|.*$", lambda m: re.sub(r"\|+", " ", m.group()), text)

    # PmWiki directives: (:...:) → remove
    text = re.sub(r"\(:[^)]+:\)", "", text)

    # (:attachlist:) etc already removed above; remove remaining markup
    text = re.sub(r"%[A-Za-z]+(?:\s+[^%]*)%%", "", text)  # %red%...%%

    # Remove lines that are purely navigation/footer boilerplate (< 3 words)
    lines = [ln for ln in text.splitlines()
             if len(ln.split()) >= 2 or not ln.strip()]
    text = "\n".join(lines)

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < 50:  # skip near-empty pages
        return None

    return page_name, page_title, text


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping token-bounded chunks.

    Strategy:
      1. Split on double newlines (paragraph boundaries) first.
      2. If a paragraph is too long, split by single newlines.
      3. Accumulate paragraphs until we hit chunk_tokens, then emit + overlap.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: list[str] = []
    current_tokens: list[int] = []
    current_text_parts: list[str] = []

    for para in paragraphs:
        para_toks = _tokeniser.encode(para)

        # If a single paragraph is already too large, split it by line
        if len(para_toks) > chunk_tokens:
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            for line in lines:
                line_toks = _tokeniser.encode(line)
                if len(current_tokens) + len(line_toks) > chunk_tokens:
                    if current_tokens:
                        chunks.append("\n".join(current_text_parts))
                        # Keep overlap: take last `overlap_tokens` tokens worth of lines
                        overlap_text = _tokeniser.decode(current_tokens[-overlap_tokens:])
                        current_tokens = _tokeniser.encode(overlap_text)
                        current_text_parts = [overlap_text]
                current_tokens.extend(line_toks)
                current_text_parts.append(line)
        else:
            if len(current_tokens) + len(para_toks) > chunk_tokens:
                if current_tokens:
                    chunks.append("\n\n".join(current_text_parts))
                    overlap_text = _tokeniser.decode(current_tokens[-overlap_tokens:])
                    current_tokens = _tokeniser.encode(overlap_text)
                    current_text_parts = [overlap_text]
            current_tokens.extend(para_toks)
            current_text_parts.append(para)

    if current_text_parts:
        chunks.append("\n\n".join(current_text_parts))

    return [c for c in chunks if c.strip()]


# ── Main ingestion logic ──────────────────────────────────────────────────────

def ingest(data_dir: Path, reset: bool = False, pmwiki: bool = True) -> None:
    """
    Scan data_dir for pages, parse them, embed, and upsert to Qdrant.

    pmwiki=True  → parse every file (no extension filter) as a PmWiki flat-file.
    pmwiki=False → parse *.txt files as DokuWiki (legacy mock data).
    """
    if pmwiki:
        # PmWiki wiki.d — all files (no extension), one page per file
        all_files = sorted(f for f in data_dir.iterdir() if f.is_file())
    else:
        all_files = sorted(data_dir.glob("*.txt"))

    if not all_files:
        print(f"No pages found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_files)} file(s) in {data_dir}")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    client = QdrantClient(url=QDRANT_URL)

    if reset:
        print(f"Dropping collection '{COLLECTION_NAME}' (--reset)…")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Creating collection '{COLLECTION_NAME}'…")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — upserting.")

    # ── Embedding model ───────────────────────────────────────────────────────
    print(f"Loading embedding model '{EMBED_MODEL}'…")
    model = SentenceTransformer(EMBED_MODEL)

    # ── Process each page ─────────────────────────────────────────────────────
    points: list[PointStruct] = []
    point_id = 0
    skipped = 0

    for path in tqdm(all_files, desc="Pages"):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if pmwiki:
            parsed = parse_pmwiki(raw)
            if parsed is None:
                skipped += 1
                continue
            page_name, page_title, clean_text = parsed
            source_file = page_name  # use the canonical PmWiki name
        else:
            # Legacy DokuWiki — simple inline parser
            title_match = __import__("re").search(r"={6}\s*(.+?)\s*={6}", raw)
            page_title = title_match.group(1).strip() if title_match else path.stem
            clean_text = raw
            source_file = path.name

        chunks = chunk_text(clean_text)

        for chunk_idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "page_title": page_title,
                        "source_file": source_file,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    },
                )
            )
            point_id += 1

    print(f"  {len(points)} chunks from {len(all_files) - skipped} pages "
          f"({skipped} skipped)")

    if not points:
        print("Nothing to upload.", file=sys.stderr)
        sys.exit(1)

    # ── Upload in batches ─────────────────────────────────────────────────────
    batch_size = 64
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nIngestion complete. {count} chunks in collection '{COLLECTION_NAME}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NeuroSpin wiki pages into Qdrant.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory to ingest. Default: /data/shared/neurospin-wiki/pmwiki/wiki.d "
            "if it exists, else ../data/mock."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force legacy DokuWiki mock-data mode (*.txt files).",
    )
    args = parser.parse_args()

    # Resolve default data directory
    if args.data_dir is None:
        real_wiki = Path("/data/shared/neurospin-wiki/pmwiki/wiki.d")
        mock_dir = Path(__file__).parent.parent / "data" / "mock"
        args.data_dir = real_wiki if real_wiki.is_dir() else mock_dir
        print(f"Auto-selected data dir: {args.data_dir}")

    if not args.data_dir.is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    use_pmwiki = not args.mock
    ingest(args.data_dir, reset=args.reset, pmwiki=use_pmwiki)


if __name__ == "__main__":
    main()
