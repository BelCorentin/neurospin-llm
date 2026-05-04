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

# Pre-download the embedding model into the image so the first request is fast.
# The model is ~1.1 GB; it will be cached in Docker layer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy application code
COPY app.py rag.py ./

EXPOSE 8080

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
