# NeuroSpin Wiki RAG Chatbot

Fully local RAG chatbot over the NeuroSpin wiki (462 pages, 988 chunks).
Ask questions in **English or French** — the system retrieves the most relevant
wiki pages and streams a grounded, cited answer. No external API calls **at query
time** (models are downloaded once, then everything runs offline).

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

## ⚠️ Two things to know before you start

**1. Keep everything off `/home` — it is tiny (~66 GB free).**
The big stuff (Docker data, model weights, the Python venv, pip/HF caches) must
live on `/data`:

| What | Location | Notes |
|------|----------|-------|
| Docker data root | `/data/docker` | already configured (`docker info \| grep "Root Dir"`) |
| Model weights (vLLM) | `huggingface_cache` named volume → `/data/docker/volumes/…` | persists across restarts |
| Embedding model (host) | `/data/shared/neurospin-llm/hf` | `HF_HOME` for the venv + mounted into the app |
| Python venv | `/data/shared/neurospin-llm/venv` | **do not** create `.venv` under the repo in `/home` |
| pip cache | `/data/shared/neurospin-llm/pipcache` | `PIP_CACHE_DIR` |

**2. This host's firewall blocks Hugging Face's weight CDNs.**
`huggingface.co` (metadata/API) is reachable, but the file CDNs that actually
serve the model weights (`xethub.hf.co`, HF's CloudFront, `cdn-lfs`) are **not**.
A plain `docker compose up` will therefore hang at ~11 MB forever while "downloading"
the model.

The fix: download the models **once** through a proxy that has full internet, then
run everything **offline** (`HF_HUB_OFFLINE=1`, already set in `docker-compose.yml`).
See [Download the models](#3--download-the-models-once-via-proxy) below.

> Set `HTTPS_PROXY`/`HTTP_PROXY` to any proxy with outbound internet access
> (e.g. an SSH tunnel: `ssh -D`/`-L`, or your org proxy). The examples below use
> the shell's existing `$HTTPS_PROXY`.

---

## Shared, single-stack design

There is **one** vLLM + Qdrant + app stack on the host, shared by everyone. Don't
run a per-user stack: vLLM owns a whole A40 and binds port `:8000`, so a second
copy just clashes and wastes a GPU. All heavy assets are shared and already
downloaded:

| Asset | Location | Shared by |
|-------|----------|-----------|
| LLM weights | docker volume `neurospin-llm_huggingface_cache` | the docker daemon (any `docker` group user) |
| bge-m3 | `/data/shared/neurospin-llm/hf` | world-readable |
| Python venv | `/data/shared/neurospin-llm/venv` | world-read/exec |
| Qdrant DB (988 chunks) | `/data/shared/neurospin-llm/qdrant` | the running container |

`docker-compose.yml` pins `name: neurospin-llm`, so **every** clone (any directory)
maps to the same named volumes — no accidental empty-volume re-download.

---

## Use it (most users — nothing to download)

**Nothing runs by default.** You start the stack, use it, then **stop it** so the
GPU is free for everyone else. Three steps:

### 1. Start the stack (takes the GPU)

```bash
git clone <repo-url> ~/git/neurospin-llm && cd ~/git/neurospin-llm   # first time only
cp .env.example .env                                                 # first time only
docker compose up -d                  # starts qdrant + vllm + app
docker compose logs -f vllm           # wait for "Application startup complete." (~1 min), Ctrl-C
```

From now on vLLM holds **~15 GB of VRAM on GPU 0** until you stop it.

### 2. Ask questions

**Terminal (SSH-friendly):**
```bash
source /data/shared/neurospin-llm/venv/bin/activate   # shared venv, read-only is fine
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 python eval/chat.py
```

**Web UI:** open `http://<host>:8080`.

No model download, no pip install, no GPU of your own to set up.

### 3. ⚠️ Stop it when done — free the VRAM

```bash
docker compose down       # stops everything, frees the GPU. Data + weights survive.
```

> **Why this matters:** vLLM reserves ~15 GB of VRAM the entire time it runs, even
> when idle and no one is chatting. Leaving it up blocks other GPU users on the
> cluster. **Rule: if you're not actively using it, run `docker compose down`.**
>
> Check who's on the GPU first: `nvidia-smi`. Check if the stack is already up
> (someone else may have started it): `docker compose ps` — if `ns-vllm` is
> `Up (healthy)`, skip step 1 and **don't** run `down` when you finish (you didn't
> start it). Coordinate so you don't kill a colleague's session.

To stop only the LLM but keep the vector DB warm: `docker compose stop vllm`.

---

## First-time setup (operator, once per host)

> On **this** host all of the below is already done. These steps are for a fresh
> host or to rebuild from scratch.

### 1 — Clone and configure

```bash
git clone <repo-url> ~/git/neurospin-llm && cd ~/git/neurospin-llm
cp .env.example .env      # defaults are fine; no HF token needed
```

The repo itself is tiny, so it can live in `/home`. Only caches/venv/weights must
go to `/data` (handled by the steps below).

### 2 — Start the vector DB

```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz     # → "healthz check passed"
```

### 3 — Download the models (once, via proxy)

Both downloads go to `/data`, never `/home`. Make sure `$HTTPS_PROXY` points at a
proxy with internet access first (`echo $HTTPS_PROXY`).

**a) LLM weights → the vLLM cache volume** (~15 GB, runs in a throwaway container
on the host network so it can reach a `127.0.0.1` proxy):

```bash
docker run --rm --name ns-dl --network host \
  -e HTTPS_PROXY="$HTTPS_PROXY" -e HTTP_PROXY="$HTTP_PROXY" -e NO_PROXY=localhost \
  -v neurospin-llm_huggingface_cache:/root/.cache/huggingface \
  --entrypoint python3 vllm/vllm-openai:latest -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', max_workers=8))"
```

> The `huggingface_cache` volume is created by `docker compose`. If it doesn't exist
> yet, run `docker volume create neurospin-llm_huggingface_cache` first.

**b) Embedding model → `/data/shared/neurospin-llm/hf`** (~2.3 GB, used by the venv and the app):

```bash
HF_HOME=/data/shared/neurospin-llm/hf HTTPS_PROXY="$HTTPS_PROXY" HTTP_PROXY="$HTTP_PROXY" NO_PROXY=localhost \
  /data/shared/neurospin-llm/venv/bin/python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-m3', max_workers=8))"
```

(The venv is created in step 5 — run 3b after 5, or just use any Python with
`huggingface_hub` installed.)

### 4 — Start vLLM (offline)

```bash
docker compose up -d vllm
docker compose logs -f vllm     # wait for "Application startup complete."
curl http://localhost:8000/health      # → empty 200 when ready
```

`HF_HUB_OFFLINE=1` is set in `docker-compose.yml`, so vLLM loads from the cached
weights and never touches the network.

### 5 — Python venv on `/data` (for ingest + terminal chat)

```bash
python3 -m venv /data/shared/neurospin-llm/venv
source /data/shared/neurospin-llm/venv/bin/activate
PIP_CACHE_DIR=/data/shared/neurospin-llm/pipcache pip install -r ingest/requirements.txt
```

> **Version pin:** `qdrant-client` must stay `>=1.9.0,<1.10.0` (server 1.9.4 does not
> support the `query_points()` API introduced in client ≥1.10).

### 6 — Ingest the wiki

Embeddings run on CPU and read bge-m3 from `/data/shared/neurospin-llm/hf`:

```bash
source /data/shared/neurospin-llm/venv/bin/activate
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 \
  python ingest/ingest.py --data-dir /data/shared/neurospin-wiki/pmwiki/wiki.d --reset
# Expected: "Ingestion complete. 988 chunks in collection 'neurospin_wiki'."
```

(Skip this if the `neurospin_wiki` collection already has data —
`curl http://localhost:6333/collections/neurospin_wiki` shows the point count.)

### 7 — Terminal chat (SSH-friendly, no browser needed)

```bash
source /data/shared/neurospin-llm/venv/bin/activate
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 python eval/chat.py
```

Commands: `/sources` · `/clear` · `/log` (show log path) · `/quit`

### 8 — Run automated tests

```bash
source /data/shared/neurospin-llm/venv/bin/activate
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 python eval/eval.py
# Expected: "Results: 6/6 tests passed"
```

### 9 — Web UI

```bash
docker compose up -d app          # builds on first run (~3 min)
# Open http://localhost:8080
```

The app reads bge-m3 from `/data/shared/neurospin-llm/hf` (mounted read-only via `HF_CACHE_DIR`
in `.env`) and runs offline — no build-time or runtime model download.

---

## Repo Layout

```
neurospin-llm/
├── docker-compose.yml      # qdrant + vllm + app  (vllm/app run HF_HUB_OFFLINE=1)
├── .env / .env.example     # HF_CACHE_DIR=/data/shared/neurospin-llm/hf, ports, model names
├── ingest/
│   ├── ingest.py           # PmWiki parser → chunker → embedder → Qdrant
│   └── requirements.txt
├── app/
│   ├── rag.py              # embed → search → prompt → stream
│   ├── app.py              # Chainlit handlers
│   └── Dockerfile          # bge-m3 mounted at runtime, NOT baked in
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

Docker runs without `sudo`/`sg` (the user is in the `docker` group):

```bash
docker compose ps                              # status
docker compose logs -f vllm                    # tail logs
docker compose restart vllm                    # restart one service
docker compose build app && docker compose up -d app   # rebuild after code change
docker compose down                            # stop all (data + weights survive)
```

Re-download the LLM weights (wipes the cache volume):
```bash
docker compose down
docker volume rm neurospin-llm_huggingface_cache
# then redo step 3a (via proxy), then `docker compose up -d`
```

Wipe the vector DB (forces re-ingest):
```bash
docker compose down
rm -rf data/qdrant/
docker compose up -d qdrant
# then redo step 6
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
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 \
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
curl http://localhost:8000/health

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
docker compose build app && docker compose up -d app
```

**Retrieval depth** — `RAG_TOP_K` in `.env` (default 5). More = better context, slower.

**Temperature** — `LLM_TEMPERATURE` in `.env` (default 0.1). Keep between 0.05–0.2.

**Language detection** — `_detect_language()` in each file. Add French words to
`fr_words`, or lower the adaptive threshold for short questions.

**Re-ingest after wiki updates:**
```bash
source /data/shared/neurospin-llm/venv/bin/activate
HF_HOME=/data/shared/neurospin-llm/hf HF_HUB_OFFLINE=1 \
  python ingest/ingest.py --data-dir /data/shared/neurospin-wiki/pmwiki/wiki.d --reset
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Model "download" hangs at ~11 MB; vLLM never becomes healthy | Firewall blocks HF weight CDNs. Do the proxy download (step 3) and run offline. Confirm: `docker run --rm curlimages/curl -sI https://cas-bridge.xethub.hf.co` returns `000`. |
| `huggingface.co` works but downloads stall | Only metadata host is whitelisted. Weights need the proxy (step 3a). |
| vLLM `Errno -2 / not a local folder` while offline | Weights not in the volume yet. Run step 3a, then `docker compose up -d vllm`. |
| `/home` filling up / disk full | Something wrote to `/home`. Check `du -sh ~/.cache ~/.cache/huggingface ingest/.venv`. Use `/data/shared/neurospin-llm` (venv, `HF_HOME`, `PIP_CACHE_DIR`) instead. |
| vLLM unhealthy / CUDA OOM | `nvidia-smi` to check GPU usage; set `NVIDIA_VISIBLE_DEVICES=1` in `.env`. |
| Qdrant `(unhealthy)` in `docker ps` | False alarm — healthcheck uses `wget`, not `curl`. `curl http://localhost:6333/healthz` to confirm. |
| `QdrantClient has no attribute 'search'` | `pip install "qdrant-client>=1.9.0,<1.10.0"` |
| French question → English answer | Lower threshold in `_detect_language()` or add French words to `fr_words` |
| Empty retrieval | Run ingestion with `--reset` (step 6) |
| `docker: permission denied` | Add yourself to the group once: `sudo usermod -aG docker $USER && newgrp docker` |
| OVH: app can't reach vLLM/Qdrant | `systemctl restart neurospin-tunnel.service` |
| OVH: WebSocket disconnects | Check nginx has `Upgrade`/`Connection` headers and `proxy_buffering off` |
