# ──────────────────────────────────────────────────────────────────────────────
# LucidCredit Backend — Production Dockerfile
# Multi-stage: builder installs deps, final stage is lean runtime image.
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps (psycopg2 compilation, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix so we can copy them to the final image
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata
LABEL org.opencontainers.image.source="https://github.com/swarnabale/lucidcredit" \
      org.opencontainers.image.description="LucidCredit Copilot API" \
      org.opencontainers.image.licenses="MIT"

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN addgroup --system lucid && adduser --system --ingroup lucid lucid

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=lucid:lucid . .

USER lucid

EXPOSE 8090

# Health check — calls the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "2"]
