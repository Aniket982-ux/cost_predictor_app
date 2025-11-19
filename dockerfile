# Stage 1: Builder - install dependencies and compile packages
FROM python:3.11.9-slim AS builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies (with curl, properly formatted)
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

RUN find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# Stage 2: Runtime - minimal image
FROM python:3.11.9-slim

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies (including curl, clean syntax)
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

# Removed HEALTHCHECK to avoid startup failures

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
