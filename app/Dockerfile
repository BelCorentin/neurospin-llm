FROM python:3.11-slim

# System deps for sentence-transformers (tokenizers requires rust compiler artefacts)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: bge-m3 is NOT baked into the image. This host's firewall blocks the HF
# weight CDNs, so the model is downloaded once via the proxy into the host cache
# (/data/cbel/hf, see README) and mounted at /hf-cache via docker-compose. The app
# runs with HF_HOME=/hf-cache and HF_HUB_OFFLINE=1.

# Copy application code
COPY app.py rag.py ./

EXPOSE 8080

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
