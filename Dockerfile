FROM python:3.12-slim

WORKDIR /app

# Install system deps for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY data/ data/

RUN pip install --no-cache-dir ".[all]" && \
    python -m spacy download en_core_web_sm

ENTRYPOINT ["epstein-pipeline"]
CMD ["--help"]
