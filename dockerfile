# Stage 1: Builder - install dependencies and compile packages
FROM python:3.11.9-slim AS builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/venv/huggingface

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gcc \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------
# PRE-DOWNLOAD MACHINE LEARNING MODELS HERE
# -------------------------------
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel

print("Downloading MPNet tokenizer/model...")
AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

print("Downloading ViT tokenizer/model...")
ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

print("Model pre-download complete.")
EOF
# -------------------------------

# Cleanup unnecessary cache files
RUN find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# Stage 2: Runtime - minimal image
FROM python:3.11.9-slim

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/venv/huggingface   # Make sure HF uses the cached models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY app.py image_embed.py inference.py text_embed.py ./
COPY embedding_refiner_checkpoint.pth trained_lgbm_model.txt ./

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080

# Cloud Run will auto-check health by hitting root or /health
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
