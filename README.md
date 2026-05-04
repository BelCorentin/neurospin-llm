# NeuroSpin Wiki RAG Chatbot

Fully local RAG chatbot over the NeuroSpin wiki (462 pages, 988 chunks).
Ask questions in **English or French** — the system retrieves the most relevant
wiki pages and streams a grounded, cited answer. No external API calls.

```
question  →  bge-m3 (CPU)  →  Qdrant  →  Qwen2.5-7B (GPU)  →  answer + sources
```

| Component | Details |
|-----------|---------|
| LLM | `Qwen/Qwen2.5-7B-Instruct` via vLLM · port 8000 · GPU 0 (A40 48 GB) |
| Embeddings | `BAAI/bge-m3` · 1024-dim cosine · CPU-only |
| Vector DB | Qdrant 1.9.4 · port 6333 · persistent at `./data/qdrant/` |
| UI | Chainlit · port 8080 · streaming + expandable sources |
| Terminal | `eval/chat.py` · readline REPL · works over plain SSH |

---

## Quick Start

### 1 — Clone and configure

```bash
git clone <repo-url> ~/neurospin-llm && cd ~/neurospin-llm
cp .env.example .env      # defaults are fine; no HF token needed
```

### 2 — Start infrastructure

```bash
sg docker -c "docker compose up -d qdrant vllm"
# First run: vLLM downloads Qwen2.5-7B (~14 GB) — takes ~5 min.
sg docker -c "docker compose logs -f vllm"   # wait for "Application startup complete."
```

### 3 — Ingest the wiki

```bash
cd ingest && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cd ..

python ingest/ingest.py --data-dir /data/shared/neurospin-wiki/pmwiki/wiki.d --reset
# Expected: "Ingestion complete. 988 chunks in collection 'neurospin_wiki'."
```

> **Version pin:** `qdrant-client` must stay `>=1.9.0,<1.10.0` (server 1.9.4 does not
> support the `query_points()` API introduced in client ≥1.10).

### 4 — Terminal chat (SSH-friendly, no browser needed)

```bash
source ingest/.venv/bin/activate
python eval/chat.py
```

Commands: `/sources` · `/clear` · `/log` (show log path) · `/quit`

### 5 — Run automated tests

```bash
source ingest/.venv/bin/activate
python eval/eval.py
# Expected: "Results: 6/6 tests passed"
```

### 6 — Web UI

```bash
sg docker -c "docker compose up -d app"   # builds on first run (~3 min)
# Open http://localhost:8080
```

---

## Repo Layout

```
neurospin-llm/
├── docker-compose.yml      # qdrant + vllm + app
├── .env / .env.example
├── ingest/
│   ├── ingest.py           # PmWiki parser → chunker → embedder → Qdrant
│   └── requirements.txt
├── app/
│   ├── rag.py              # embed → search → prompt → stream
│   ├── app.py              # Chainlit handlers
│   └── Dockerfile
├── eval/
│   ├── eval.py             # 6 automated Q/A tests
│   └── chat.py             # terminal REPL with readline + JSONL logging
└── deploy/                 # OVH server files (see below)
    ├── docker-compose.ovh.yml
    ├── tunnel.sh / tunnel.service
    └── nginx.conf
```

---

## Docker Operations

```bash
sg docker -c "docker compose ps"                          # status
sg docker -c "docker compose logs -f vllm"               # tail logs
sg docker -c "docker compose restart vllm"               # restart one service
sg docker -c "docker compose build app && docker compose up -d app"  # rebuild after code change
sg docker -c "docker compose down"                        # stop all (data survives)
```

Nuclear wipe (re-downloads model, wipes vector DB):
```bash
sg docker -c "docker compose down"
sg docker -c "docker volume rm neurospin-llm_huggingface_cache"
rm -rf data/qdrant/
sg docker -c "docker compose up -d"
```

---

## How Services Connect

**On the cluster (all Docker):**
```
Browser / terminal
  → ns-app :8080 (Chainlit)
      ├─ qdrant:6333   (internal Docker DNS)
      └─ vllm:8000/v1  (internal Docker DNS)
```

**Terminal chat** (`eval/chat.py`) connects directly to `localhost:6333` and
`localhost:8000` — the same ports the containers expose on the host.

Environment overrides for terminal chat:
```bash
QDRANT_URL=http://localhost:6333 \
VLLM_BASE_URL=http://localhost:8000/v1 \
RAG_TOP_K=8 \
python eval/chat.py
```

---

## OVH Deployment

The OVH server runs only the lightweight Chainlit app and forwards traffic to the
cluster via a persistent SSH tunnel.

```
Internet → nginx (OVH, HTTPS) → Chainlit app (Docker)
                                       │  SSH tunnel (autossh)
                                       ├─ localhost:8000 → cluster vLLM
                                       └─ localhost:6333 → cluster Qdrant
```

**Steps (OVH server):**

```bash
# 1. Install deps
apt install -y docker.io autossh nginx certbot python3-certbot-nginx

# 2. Passwordless SSH key to cluster
ssh-keygen -t ed25519 -f ~/.ssh/cluster_key -N ""
ssh-copy-id -i ~/.ssh/cluster_key.pub cbel@ext1.idris.fr

# 3. systemd tunnel (auto-reconnects on drop)
cp deploy/tunnel.service /etc/systemd/system/neurospin-tunnel.service
# Edit: User=, CLUSTER_USER=, CLUSTER_HOST=, add -i ~/.ssh/cluster_key
systemctl enable --now neurospin-tunnel.service

# Verify tunnel:
curl http://localhost:8000/health    # → {"status":"ok"}

# 4. SSL cert (point DNS A record to OVH IP first)
certbot --nginx -d wiki-chat.neurospin.fr

# 5. nginx
cp deploy/nginx.conf /etc/nginx/sites-available/neurospin-wiki
# Edit server_name to match your domain
ln -s /etc/nginx/sites-available/neurospin-wiki /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 6. App
cp deploy/.env.ovh.example deploy/.env.ovh
docker compose -f deploy/docker-compose.ovh.yml --env-file deploy/.env.ovh up -d
```

The app container reaches the tunnelled ports via `host.docker.internal`
(configured in `docker-compose.ovh.yml` via `extra_hosts`).

---

## Conversation Logs

`eval/chat.py` appends every turn to `~/.neurospin-rag-chat.log` (JSONL).

```bash
tail -1 ~/.neurospin-rag-chat.log | python3 -m json.tool   # last turn
wc -l ~/.neurospin-rag-chat.log                            # turn count
RAG_LOG_FILE=/tmp/session.log python eval/chat.py          # custom path
```

Each record: `{ts, lang, question, answer, sources:[{page, score}]}`.

---

## Customisation

**System prompt** — edit `SYSTEM_PROMPT` in `app/rag.py` (web UI) or `eval/chat.py`
(terminal). Rebuild the container after changing `app/rag.py`:
```bash
sg docker -c "docker compose build app && docker compose up -d app"
```

**Retrieval depth** — `RAG_TOP_K` in `.env` (default 5). More = better context, slower.

**Temperature** — `LLM_TEMPERATURE` in `.env` (default 0.1). Keep between 0.05–0.2.

**Language detection** — `_detect_language()` in each file. Add French words to
`fr_words`, or lower the adaptive threshold for short questions.

**Re-ingest after wiki updates:**
```bash
source ingest/.venv/bin/activate
python ingest/ingest.py --data-dir /data/shared/neurospin-wiki/pmwiki/wiki.d --reset
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: permission denied` | `sg docker -c "..."` or `sudo usermod -aG docker $USER && newgrp docker` |
| vLLM unhealthy / CUDA OOM | `nvidia-smi` to check GPU usage; change `NVIDIA_VISIBLE_DEVICES=1` in `.env` |
| Qdrant `(unhealthy)` in `docker ps` | False alarm — healthcheck uses `wget`, not `curl`. Run `curl http://localhost:6333/healthz` to confirm it works. |
| `QdrantClient has no attribute 'search'` | `pip install "qdrant-client>=1.9.0,<1.10.0"` |
| French question → English answer | Lower threshold in `_detect_language()` or add French words to `fr_words` |
| Empty retrieval | Run ingestion with `--reset` |
| OVH: app can't reach vLLM/Qdrant | `systemctl restart neurospin-tunnel.service` |
| OVH: WebSocket disconnects | Check nginx has `Upgrade`/`Connection` headers and `proxy_buffering off` |
