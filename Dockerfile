# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the package with all optional deps (except GPU-only extras)
RUN pip install --no-cache-dir --prefix=/install \
    ".[ocr,ocr-surya,pymupdf,nlp,nlp-gliner,ai,vision,embeddings,neon,classify]"

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps: poppler (pdf2image for Surya), libpq (psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpq5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY data/ data/

# Install the package itself (deps already present from builder)
RUN pip install --no-cache-dir --no-deps -e .

# Download spaCy model (sm for smaller image; trf available via extras)
RUN python -m spacy download en_core_web_sm || true

ENTRYPOINT ["epstein-pipeline"]
CMD ["--help"]
