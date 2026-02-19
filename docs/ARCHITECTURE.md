# Architecture

## Pipeline Overview

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

## Directory Structure

```
src/epstein_pipeline/
├── cli.py                          # Click CLI entry point (all commands)
├── config.py                       # Pydantic settings (env vars, paths, model names)
├── state.py                        # Pipeline state tracking (processed files, hashes)
│
├── downloaders/                    # Data source fetchers
│   ├── doj.py                      # DOJ EFTA dataset downloads (DS1-DS12)
│   ├── kaggle.py                   # Kaggle Epstein Ranker dataset
│   ├── huggingface.py              # HuggingFace datasets (emails, filings)
│   └── archive.py                  # Archive.org media collections
│
├── processors/                     # Core processing pipeline
│   ├── ocr.py                      # Multi-backend OCR (Docling, Surya, olmOCR, PyMuPDF)
│   ├── pymupdf_extractor.py        # PyMuPDF-specific text/image extraction
│   ├── entities.py                 # spaCy + GLiNER NER with person registry matching
│   ├── person_linker.py            # Fast substring person linking (no spaCy dep)
│   ├── confidence.py               # Numeric confidence scores for entity matches
│   ├── dedup.py                    # Three-pass dedup (hash → MinHash → semantic)
│   ├── classifier.py               # Zero-shot BART document classification
│   ├── chunker.py                  # Semantic text chunking (paragraph-aware)
│   ├── embeddings.py               # nomic-embed-text-v2-moe vector generation
│   ├── knowledge_graph.py          # Entity relationship graph (JSON + GEXF)
│   ├── redaction.py                # Redaction detection + recovery analysis
│   ├── image_extractor.py          # PDF image extraction + optional AI description
│   ├── plist_forensics.py          # Apple Mail PLIST metadata extraction
│   ├── summarizer.py               # AI document summarization (Ollama / OpenAI)
│   └── transcriber.py              # Audio/video transcription (faster-whisper)
│
├── exporters/                      # Output format converters
│   ├── json_export.py              # JSON export (site-compatible camelCase)
│   ├── csv_export.py               # CSV export for researchers
│   ├── sqlite_export.py            # SQLite with FTS5 full-text search
│   ├── neon_export.py              # Neon Postgres with pgvector embeddings
│   ├── neon_schema.py              # Idempotent Neon schema migration SQL
│   └── site_sync.py                # Direct sync to epstein-index site data/
│
├── importers/                      # External data importers
│   └── sea_doughnut.py             # Import Sea_Doughnut research databases
│
├── models/                         # Pydantic data models
│   ├── document.py                 # Document, Page, Entity, Embedding models
│   ├── registry.py                 # Person registry (names, aliases, IDs)
│   └── forensics.py                # Redaction, PLIST, image analysis models
│
├── validators/                     # Data quality enforcement
│   ├── schema.py                   # JSON schema validation
│   └── integrity.py                # Cross-reference integrity checks
│
└── utils/                          # Shared utilities
    ├── hashing.py                  # Content hashing (SHA-256, SimHash)
    ├── parallel.py                 # ProcessPoolExecutor wrapper
    └── progress.py                 # Rich progress bars
```

## Key Design Decisions

### Multi-Backend OCR with Fallback Chain

The pipeline supports four OCR backends because no single engine handles all document types well:

| Backend | Strengths | Weaknesses |
|---|---|---|
| **PyMuPDF** | Instant, extracts existing text layers | Cannot OCR scanned images |
| **Surya** | Fast, 90+ languages, good accuracy | Misses some layouts |
| **olmOCR 2** | Highest accuracy (VLM-based) | Requires 8GB+ GPU |
| **Docling (IBM)** | Understands tables/layout, no GPU | Slower than Surya |

The default strategy (`--backend both`) tries PyMuPDF first (for documents with text layers), then falls back to Docling for scanned pages. Per-page confidence scoring tracks which backend produced each extraction.

### Three-Pass Deduplication

Duplicate detection uses three complementary approaches:

1. **Content hash** — SHA-256 of normalized text catches exact duplicates (O(1) per doc)
2. **MinHash/LSH** — Locality-sensitive hashing finds near-duplicates (O(n) total, sublinear per query)
3. **Semantic embeddings** — Cosine similarity catches OCR-variant duplicates where the same document was scanned differently

Results are stored in `data/known-duplicates.json` with human-reviewable match explanations.

### Pydantic Models with camelCase Fields

Pydantic v2 models use `camelCase` field names (via `alias_generator`) to directly match the TypeScript interfaces on epsteinexposed.com. This means data flows from pipeline → JSON → site with zero transformation:

```python
class Document(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    source: str
    personIds: list[str] = Field(alias="person_ids")  # camelCase in JSON
```

### Person Registry

The person registry (`data/persons-registry.json`) contains 1,500+ known persons with:
- Canonical names, aliases, and spelling variations
- Unique IDs matching the site's person pages
- Categories (associate, legal, political, victim, etc.)

Entity extraction matches against this registry using rapidfuzz fuzzy matching with configurable confidence thresholds.

### Neon Postgres with pgvector

The Neon exporter creates a production-ready database with:
- **pgvector** for semantic search (cosine similarity on 768-dim embeddings)
- **pg_trgm** for fuzzy text search (trigram similarity)
- **FTS** via tsvector/GIN indexes for full-text search
- **IVFFlat** indexes for approximate nearest neighbor queries
- Idempotent schema migration (`epstein-pipeline migrate`)

### Knowledge Graph

The knowledge graph processor builds weighted entity-relationship graphs:
- **Co-occurrence edges** from documents mentioning multiple persons
- **Co-passenger edges** from flight log data
- **Correspondence edges** from email sender/recipient pairs
- Optional **LLM relationship extraction** for relationship labeling

Output formats: JSON (for D3.js visualization) and GEXF (for Gephi analysis).

## Data Flow

```
1. Download     Raw PDFs from DOJ, Kaggle, HuggingFace, Archive.org
                ↓
2. OCR          Extract text (PyMuPDF → Surya → Docling fallback chain)
                ↓
3. Entities     spaCy NER + GLiNER zero-shot + regex patterns
                ↓
4. Person Link  Match entity names → canonical person IDs (rapidfuzz)
                ↓
5. Classify     Zero-shot BART → 12 legal document categories
                ↓
6. Dedup        Hash → MinHash/LSH → semantic similarity
                ↓
7. Chunk        Semantic paragraph-aware text splitting
                ↓
8. Embed        nomic-embed-text-v2-moe → 768-dim vectors
                ↓
9. Validate     Schema checks, cross-reference integrity
                ↓
10. Export      JSON, CSV, SQLite, or Neon Postgres
```

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | What It Does |
|---|---|---|
| `ci.yml` | Push/PR to main | Lint, test (3.10-3.13), typecheck, schema validation |
| `publish.yml` | Release tag | Build and publish to PyPI |
| `validate-data.yml` | PR with data changes | Validate contributed data files |

### Docker

Multi-stage build for smaller images:
- **Builder stage:** Installs all dependencies with build tools
- **Runtime stage:** Copies only installed packages + runtime deps
- Includes spaCy `en_core_web_sm` model
- Entry point: `epstein-pipeline` CLI

```bash
docker compose run pipeline ocr ./pdfs/ --output ./output/
docker compose run pipeline export neon --input-dir ./output/
```
