# QuantMind Backend — Multi-stage Dockerfile
#
# Stage 1 (builder): Install Python dependencies into a venv
# Stage 2 (runtime): Copy only the venv + source — no build tools in final image
#
# WHY python:3.11-slim (not alpine):
#   sentence-transformers and chromadb require glibc (not musl libc).
#   Alpine uses musl — causes cryptic build failures for these packages.
#   slim is ~150MB vs alpine ~50MB, but it actually works.
#
# WHY multi-stage:
#   Build tools (gcc, g++, cmake) needed to compile some packages are ~400MB.
#   We don't need them at runtime. Multi-stage keeps the final image lean.

# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — install all dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build dependencies (needed for chromadb, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment in /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements first (Docker layer cache optimization)
# If requirements.txt doesn't change, this layer is cached — fast rebuilds
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy virtual environment from builder (no gcc/g++ in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash quantmind
WORKDIR /app

# Copy application source
COPY --chown=quantmind:quantmind . .

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma && chown -R quantmind:quantmind /app/data

# Switch to non-root user
USER quantmind

# Environment defaults (override via docker-compose or Render env vars)
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV LOG_LEVEL=INFO
ENV DEBUG=false

# Expose FastAPI port
EXPOSE 8000

# Health check — Render and docker-compose use this to know when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI with uvicorn
# --workers 1: single worker (FinBERT + sentence-transformers are large, 1 worker = safer on free tier)
# --host 0.0.0.0: required for Docker networking
# $PORT: Render injects this env var; fallback to 8000 for local Docker
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
