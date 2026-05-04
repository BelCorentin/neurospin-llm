# NeuroSpin Wiki RAG Chatbot

A fully local, GPU-accelerated Retrieval-Augmented Generation (RAG) chatbot for the [NeuroSpin wiki](https://neurospin-wiki.org/). Users ask questions in English or French; the system retrieves the most relevant wiki pages and generates a grounded, sourced answer — without hallucinating.

---

## Architecture

```
User (browser)
     │
     ▼
 Chainlit UI  (port 8080)
     │
     │  1. embed question with bge-m3 (CPU)
     ▼
 Qdrant       (port 6333) ──► top-5 matching wiki chunks
     │
     │  2. build prompt with retrieved context
     ▼
 vLLM         (port 8000) ──► Qwen2.5-7B-Instruct on GPU 0
     │
     │  3. stream answer + source citations
     ▼
 User (browser)
```

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB fp16, runs on GPU 0 (48 GB A40). Ungated, no HF token required. |
| Embeddings | `BAAI/bge-m3` | Multilingual (EN/FR), ~1.1 GB, CPU-only inside app container |
| Vector DB | Qdrant | Persistent volume at `./data/qdrant/` |
| UI | Chainlit | Streams responses, shows expandable source citations |
| Orchestration | Direct Python (no LangChain/LlamaIndex) | Simpler, easier to debug |
| Infrastructure | Docker Compose | Single `docker compose up` starts everything |

---

## Repository Layout

```
neurospin-llm/
├── docker-compose.yml       # Full service stack
├── .env.example             # Config template — copy to .env
├── .env                     # Your local config (gitignored)
├── .gitignore
│
├── data/
│   ├── mock/                # 5 mock DokuWiki pages (replace with real pages)
│   │   ├── getting_started.txt
│   │   ├── mri_protocols.txt
│   │   ├── software_install.txt
│   │   ├── data_storage.txt
│   │   └── network_access.txt
│   └── qdrant/              # Qdrant persistent storage (gitignored)
│
├── ingest/
│   ├── ingest.py            # Parse → chunk → embed → upload to Qdrant
│   └── requirements.txt
│
└── app/
    ├── app.py               # Chainlit entrypoint
    ├── rag.py               # RAG pipeline (embed → search → prompt → stream)
    ├── requirements.txt
    └── Dockerfile
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change ports or model name
```

### 2. Start Qdrant

```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz   # should return {"title":"qdrant - version x.x.x"}
```

### 3. Run the ingestion pipeline

This reads `data/mock/*.txt`, chunks them, embeds with bge-m3, and uploads to Qdrant.

```bash
# One-time setup — creates the Python env and runs ingestion
cd ingest
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
# Expected output: "Ingestion complete. X chunks uploaded."
cd ..
```

Re-run `python ingest.py --reset` any time you update the wiki pages. The `--reset` flag drops and recreates the collection.

### 4. Start vLLM

```bash
docker compose up -d vllm
# First run downloads Qwen2.5-7B-Instruct (~14 GB) — takes a few minutes.
# Watch progress:
docker compose logs -f vllm
# Ready when you see: "Application startup complete."
curl http://localhost:8000/v1/models
```

### 5. Start the app

```bash
docker compose up -d app
# Open http://localhost:8080
```

### 6. Try it

- English: *"How do I install FSL on the NeuroSpin cluster?"*
- French: *"Comment accéder au VPN NeuroSpin ?"*
- Out-of-scope: *"What is the capital of France?"* → should respond that it cannot find this in the wiki.

---

## Replacing Mock Data with Real Wiki Pages

Once you have access to the DokuWiki raw files (`.txt` format from the server), drop them in `data/real/` (or replace `data/mock/`), then re-run:

```bash
cd ingest && source .venv/bin/activate
python ingest.py --data-dir ../data/real --reset
```

The DokuWiki raw files are stored at `/var/lib/dokuwiki/data/pages/` on the wiki server — ask the admin for an rsync or tar dump of that directory.

---

## Exposing via Reverse Proxy (nginx example)

To make the chatbot available at e.g. `https://your-server.inm7.de/neurospin-chat/`:

```nginx
location /neurospin-chat/ {
    proxy_pass http://localhost:8080/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 120s;
}
```

Chainlit uses WebSockets for streaming — the `Upgrade` headers are required.

To link from the DokuWiki sidebar, add to your `sidebar.txt`:
```dokuwiki
  * [[https://your-server/neurospin-chat/|🤖 Ask the Wiki Assistant]]
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No LangChain / LlamaIndex** | Direct stack (qdrant-client + sentence-transformers + openai-client + chainlit) is more transparent and easier to debug for a POC |
| **Qwen2.5-7B over Llama-3.1-8B** | Ungated (no HF license acceptance), strong multilingual EN/FR support, comparable quality |
| **bge-m3 for embeddings** | Best-in-class multilingual dense retrieval, handles EN+FR queries against EN+FR wiki pages natively |
| **Qdrant over ChromaDB** | Docker-native, persistent storage, production-ready REST+gRPC API, easier to scale |
| **Chainlit over Open WebUI** | Python-native, trivial to add "Sources" citation elements per response, no separate backend needed |
| **CPU embeddings** | bge-m3 at ~40ms/query on CPU is fast enough for <20 concurrent users; keeps GPU 1 free for a second vLLM instance if needed |
| **500-token chunks, 100 overlap** | Balances context richness vs. precision; wiki pages are typically dense technical instructions |

---

## Key Limits of This POC

- Single-user ingestion (no live wiki sync — re-run `ingest.py` manually)
- No authentication on the Chainlit UI (add an auth callback in `app.py` when going to production)
- Conversation history is per-session only (not persisted to disk)
