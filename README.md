# Epstein Pipeline

[![CI](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

World-class document processing pipeline for the [Epstein case files](https://epsteinexposed.com). Download, OCR, extract entities, deduplicate, embed, and export 140,000+ documents to Neon Postgres with pgvector semantic search.

**This is the data processing engine behind [epsteinexposed.com](https://epsteinexposed.com)** — the most comprehensive searchable database of the Epstein files.

## Architecture

```
Raw DOJ PDFs
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  OCR (multi-backend)                                     │
│  PyMuPDF → Surya → olmOCR 2 → Docling                   │
│  Per-page confidence scoring, fallback chain             │
└──────────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌────────────┐  ┌────────────┐  ┌──────────────────┐
│ NER        │  │ Dedup      │  │ Classifier       │
│ spaCy trf  │  │ Hash →     │  │ Zero-shot BART   │
│ + GLiNER   │  │ MinHash →  │  │ 12 doc categories│
│ + regex    │  │ Semantic   │  │                  │
└─────┬──────┘  └─────┬──────┘  └────────┬─────────┘
      │               │                  │
      └───────────────┼──────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Semantic Chunker → Embeddings (nomic-embed-text-v2-moe) │
│  Paragraph-aware splitting, 768-dim / 256-dim Matryoshka │
└──────────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌────────────┐  ┌────────────┐  ┌──────────────────┐
│ Neon PG    │  │ JSON/CSV   │  │ Knowledge Graph  │
│ + pgvector │  │ SQLite     │  │ GEXF + JSON      │
│ cosine ANN │  │ NDJSON     │  │ LLM extraction   │
└────────────┘  └────────────┘  └──────────────────┘
```

## Quickstart

```bash
# Install with all features
pip install "epstein-pipeline[all]"
python -m spacy download en_core_web_sm

# Download a dataset
epstein-pipeline download kaggle

# OCR with automatic backend selection
epstein-pipeline ocr ./raw-pdfs/ --output ./processed/

# Extract entities (spaCy + GLiNER)
epstein-pipeline extract-entities ./processed/ --output ./entities/

# Generate embeddings and push to Neon
epstein-pipeline embed ./processed/ --output ./embeddings/ --format neon

# Export everything to Neon Postgres
epstein-pipeline export neon --input-dir ./processed/
```

### Neon Postgres Setup

```bash
# Set your Neon connection string
export EPSTEIN_NEON_DATABASE_URL="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/epstein"

# Run schema migration (idempotent, safe to re-run)
epstein-pipeline migrate

# Semantic search from the command line
epstein-pipeline search "financial transactions offshore accounts"
```

## Processing Backends

| Component | Backend | Speed | Accuracy | GPU Required |
|-----------|---------|-------|----------|--------------|
| **OCR** | PyMuPDF | Instant | Text layers only | No |
| **OCR** | Surya | Fast | High (90+ langs) | Optional |
| **OCR** | olmOCR 2 | Slow | Highest (VLM) | Yes (8GB+) |
| **OCR** | Docling (IBM) | Medium | High | No |
| **NER** | spaCy `en_core_web_trf` | Fast | High | Optional |
| **NER** | GLiNER | Medium | High (zero-shot) | Optional |
| **Dedup** | Content hash + fuzzy | Instant | Exact only | No |
| **Dedup** | MinHash/LSH | O(n) | Near-duplicate | No |
| **Dedup** | Semantic embeddings | Slow | OCR-variant | Optional |
| **Embeddings** | nomic-embed-text-v2-moe | Fast | SOTA | Optional |
| **Classifier** | BART-large-mnli | Medium | Good | Optional |

## Installation

```bash
# Core only (no ML models)
pip install epstein-pipeline

# With OCR (CPU — Surya)
pip install "epstein-pipeline[ocr-surya]"

# With OCR (GPU — olmOCR 2, requires CUDA)
pip install "epstein-pipeline[ocr-gpu]"

# With NLP (spaCy + GLiNER)
pip install "epstein-pipeline[nlp,nlp-gliner]"

# With embeddings (sentence-transformers + torch)
pip install "epstein-pipeline[embeddings]"

# With Neon Postgres export (psycopg + pgvector)
pip install "epstein-pipeline[neon]"

# Everything (except GPU-only olmOCR)
pip install "epstein-pipeline[all]"
```

### Docker

```bash
docker compose run pipeline --help
docker compose run pipeline ocr ./raw-pdfs/ --output ./output/
docker compose run pipeline migrate
```

## CLI Commands

```bash
# ── Data Ingestion ──────────────────────────────────────────────
epstein-pipeline download doj --dataset 9       # Download DOJ dataset
epstein-pipeline download kaggle                # Download Kaggle dataset
epstein-pipeline download huggingface           # Download HuggingFace datasets

# ── Processing ──────────────────────────────────────────────────
epstein-pipeline ocr ./pdfs/ -o ./out/          # OCR (auto backend)
epstein-pipeline ocr ./pdfs/ --backend surya    # OCR with specific backend
epstein-pipeline extract-entities ./out/ -o ./e/ # NER extraction
epstein-pipeline classify --input-dir ./out/    # Document classification
epstein-pipeline dedup ./out/ --mode all        # 3-pass deduplication
epstein-pipeline embed ./out/ -o ./emb/         # Generate embeddings

# ── Export ──────────────────────────────────────────────────────
epstein-pipeline export json ./out/ -o ./site/  # JSON for website
epstein-pipeline export csv ./out/ -o docs.csv  # CSV for research
epstein-pipeline export sqlite ./out/ -o ep.db  # SQLite database
epstein-pipeline export neon --input-dir ./out/ # Push to Neon Postgres

# ── Database ────────────────────────────────────────────────────
epstein-pipeline migrate                        # Run Neon schema migration
epstein-pipeline search "query text here"       # Semantic search (pgvector)

# ── Utilities ───────────────────────────────────────────────────
epstein-pipeline validate ./out/                # Data quality checks
epstein-pipeline stats ./out/                   # Show statistics
```

## Key Features

- **Multi-backend OCR** with automatic fallback chain and per-page confidence scoring
- **Three-pass deduplication**: exact hash → MinHash/LSH (O(n)) → semantic similarity
- **GLiNER zero-shot NER** for custom legal entity types (case numbers, flight IDs, financial amounts)
- **Semantic chunking** that respects paragraph and sentence boundaries
- **pgvector embeddings** with cosine similarity search via Neon Postgres
- **Document classification** using zero-shot BART into 12 legal categories
- **Knowledge graph** with co-occurrence edges and opt-in LLM relationship extraction
- **Idempotent Neon schema migration** with pgvector, pg_trgm, and IVFFlat indexes

## Data Sources

See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for all known public data sources.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

**No coding required:** Report data quality issues, suggest sources, review processed data.

**Code contributions:** Add downloaders, improve extraction, add export formats, fix bugs.

## Related Projects

- [epsteinexposed.com](https://epsteinexposed.com) — The live website powered by this pipeline
- [Epstein-Files](https://github.com/WikiLeaksLookup/Epstein-Files) — DOJ file mirrors
- [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) — Email graph explorer
- [Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) — Community research dataset

## License

MIT License. See [LICENSE](LICENSE).
