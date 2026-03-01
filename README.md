# Epstein Pipeline

[![CI](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source document processing pipeline for the Jeffrey Epstein case files. Downloads, OCRs, extracts entities, deduplicates, embeds, and exports **2.1 million+ documents** to Neon Postgres with pgvector semantic search.

**This is the data engine behind [epsteinexposed.com](https://epsteinexposed.com)** -- the most comprehensive searchable database of the Epstein files.

## What It Does

```
DOJ EFTA Releases (DS1-DS12)  ─┐
Kaggle Datasets                ─┤
HuggingFace Collections        ─┼──► Download
Archive.org Mirrors            ─┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  OCR (multi-backend fallback chain)                      │
│  PyMuPDF → Surya → olmOCR 2 → Docling                   │
│  Per-page confidence scoring, automatic backend selection│
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
      ▼               ▼                  ▼
┌────────────┐  ┌────────────┐  ┌──────────────────┐
│ Summarizer │  │ Redaction  │  │ Image Extractor  │
│ LLM-based  │  │ Analysis   │  │ + AI description │
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

## Current Scale

| Metric | Count |
|--------|-------|
| Documents ingested | 2,145,000+ |
| OCR texts extracted | 2,014,000+ |
| Persons identified | 1,723 |
| Document-person links | 2,443,000+ |
| SHA-256 integrity hashes | 1,380,000+ |
| DOJ datasets processed | 12 of 12 (DS1-DS12) |

## Quickstart

```bash
# Install with all features
pip install "epstein-pipeline[all]"
python -m spacy download en_core_web_sm

# Download a DOJ dataset
epstein-pipeline download doj --dataset 9

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

# With OCR (CPU -- Surya)
pip install "epstein-pipeline[ocr-surya]"

# With OCR (GPU -- olmOCR 2, requires CUDA)
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

## CLI Reference

```bash
# -- Data Ingestion ------------------------------------------------
epstein-pipeline download doj --dataset 9       # Download DOJ EFTA dataset (1-12)
epstein-pipeline download kaggle                # Download Kaggle dataset
epstein-pipeline download huggingface           # Download HuggingFace datasets
epstein-pipeline download archive               # Download from Archive.org mirrors

# -- Processing ----------------------------------------------------
epstein-pipeline ocr ./pdfs/ -o ./out/          # OCR (auto backend selection)
epstein-pipeline ocr ./pdfs/ --backend surya    # OCR with specific backend
epstein-pipeline extract-entities ./out/ -o ./e/ # NER extraction (spaCy + GLiNER)
epstein-pipeline classify --input-dir ./out/    # Zero-shot document classification
epstein-pipeline dedup ./out/ --mode all        # 3-pass deduplication
epstein-pipeline embed ./out/ -o ./emb/         # Generate embeddings

# -- Export --------------------------------------------------------
epstein-pipeline export json ./out/ -o ./site/  # JSON for website
epstein-pipeline export csv ./out/ -o docs.csv  # CSV for researchers
epstein-pipeline export sqlite ./out/ -o ep.db  # SQLite database
epstein-pipeline export neon --input-dir ./out/ # Push to Neon Postgres

# -- Database ------------------------------------------------------
epstein-pipeline migrate                        # Run Neon schema migration
epstein-pipeline search "query text here"       # Semantic search (pgvector)

# -- Quality -------------------------------------------------------
epstein-pipeline validate ./out/                # Data quality checks
epstein-pipeline stats ./out/                   # Show processing statistics

# -- Sanctions & PEP Cross-Reference --------------------------------
epstein-pipeline check-sanctions               # Cross-check all persons vs OpenSanctions
epstein-pipeline check-sanctions --threshold 0.3 --use-search  # Lower threshold, search API
epstein-pipeline import-sanctions ./output/sanctions/opensanctions-results.json

# -- Person Integrity Auditor -------------------------------------
epstein-pipeline audit-persons                  # Full 5-phase audit
epstein-pipeline audit-persons --phases dedup   # Single phase only
epstein-pipeline audit-persons --person bill-clinton --dry-run
epstein-pipeline audit-persons --min-severity 40 -o report.json
```

## Processors

### OCR (`processors/ocr.py`)
Multi-backend OCR with automatic fallback. Tries PyMuPDF (text extraction) first, falls back through Surya, olmOCR 2, and Docling based on per-page confidence scores. Handles scanned PDFs, image-only pages, and mixed documents.

### Entity Extraction (`processors/entities.py`)
Hybrid NER using spaCy transformer models + GLiNER zero-shot extraction + regex patterns. Extracts people, organizations, locations, dates, case numbers, flight IDs, financial amounts, and Bates numbers from legal documents.

### Deduplication (`processors/dedup.py`)
Three-pass deduplication pipeline:
1. **Exact hash** -- SHA-256 content hash for identical files
2. **MinHash/LSH** -- O(n) near-duplicate detection for OCR variants
3. **Semantic similarity** -- Embedding cosine similarity for reformatted duplicates

### Document Classification (`processors/classifier.py`)
Zero-shot classification using BART-large-mnli into 12 legal document categories (court filings, depositions, correspondence, financial records, flight logs, etc.).

### Semantic Chunking (`processors/chunker.py`)
Paragraph-aware text splitting with OCR noise cleaning. Respects sentence and paragraph boundaries, targets 450 tokens per chunk with 50-token overlap. Includes contextual prefixes (document title + source) per chunk.

### Embeddings (`processors/embeddings.py`)
Generates vector embeddings using nomic-embed-text-v2-moe (768-dim, Matryoshka to 256-dim). Used for semantic deduplication and search indexing.

### Redaction Analysis (`processors/redaction.py`)
Detects redacted regions in PDFs and attempts text recovery where redactions are improperly applied (transparent overlays, recoverable text layers).

### Image Extraction (`processors/image_extractor.py`)
Extracts embedded images from PDFs with optional AI-powered description via vision models.

### Summarization (`processors/summarizer.py`)
LLM-based document summarization for generating concise descriptions of legal documents.

### Person Linking (`processors/person_linker.py`)
Links extracted entity mentions to known persons in the database using fuzzy name matching with word boundary safety (multi-word names only to prevent false positives).

### Knowledge Graph (`processors/knowledge_graph.py`)
Builds entity relationship graphs from co-occurrence analysis and optional LLM-based relationship extraction. Exports to GEXF and JSON formats.

### Plist Forensics (`processors/plist_forensics.py`)
Parses Apple plist files found in the Epstein device data for contact and metadata extraction.

## OpenSanctions Cross-Reference

Cross-references all 1,538 persons against 100+ global sanctions, PEP, and watchlist datasets via the [OpenSanctions API](https://opensanctions.org/).

**Datasets checked:** OFAC SDN, EU Financial Sanctions, UN Security Council, UK HMT, Interpol Red Notices, PEP registries (Every Politician), ICIJ Offshore Leaks (mirrored), and 100+ more.

```bash
# Cross-check all persons (takes ~13 min at 0.5s/request rate limit)
export EPSTEIN_OPENSANCTIONS_API_KEY="your-api-key"
epstein-pipeline check-sanctions

# Import results into Neon Postgres
epstein-pipeline import-sanctions ./output/sanctions/opensanctions-results.json
```

**What it does:**
1. Loads all persons from `data/persons-registry.json`
2. Queries OpenSanctions `/match` endpoint for each person (fuzzy name matching)
3. Flags persons as `is_sanctioned` (on any sanctions list) or `is_pep` (politically exposed)
4. Saves detailed results to `output/sanctions/opensanctions-results.json`
5. `import-sanctions` writes flags to the `persons` table and creates a `sanctions_matches` table in Neon

**Output:** Each person gets: best match score, sanctions/PEP flags, matched datasets, and individual match details. Results are displayed in a Rich summary table with top matches ranked by score.

Requires: `EPSTEIN_OPENSANCTIONS_API_KEY` (free for non-commercial use at [opensanctions.org](https://opensanctions.org/))

## Person Integrity Auditor

Automated 5-phase data quality pipeline that scans all person records against the Neon database, Wikidata, Wikipedia, and Claude AI to detect issues before they reach users.

| Phase | What It Does | Cost |
|-------|-------------|------|
| **Dedup** | rapidfuzz name similarity + alias cross-check for duplicate entries | Free |
| **Wikidata** | Cross-reference occupation, dates, nationality against Wikidata + Wikipedia | Free |
| **Fact-Check** | Decompose bios into atomic claims, verify against 2M+ documents via FTS | ~$1-2 |
| **Coherence** | Sample linked documents, detect merged identities (one record = two people) | ~$0.50 |
| **Score** | Calculate composite severity (0-100), create ai_leads for admin review | Free |

**Severity Tiers**: Critical (70-100), High (40-69), Medium (20-39), Low (0-19)

Issues detected: duplicate entries, merged identities, wrong categories, bio contradictions, ungrounded claims, stale data, external contradictions with Wikidata/Wikipedia.

Requires: `EPSTEIN_AUDITOR_ANTHROPIC_API_KEY` + `EPSTEIN_NEON_DATABASE_URL`

Optional: `EPSTEIN_AUDITOR_VOYAGE_API_KEY` (semantic search), `EPSTEIN_AUDITOR_COHERE_API_KEY` (reranking)

## Environment Variables

All configuration is via environment variables prefixed with `EPSTEIN_`. No credentials are ever stored in code or config files.

| Variable | Required | Purpose |
|----------|----------|---------|
| `EPSTEIN_NEON_DATABASE_URL` | For DB export/search | Neon Postgres connection string |
| `EPSTEIN_OPENSANCTIONS_API_KEY` | For sanctions check | OpenSanctions API key (free for non-commercial) |
| `EPSTEIN_AUDITOR_ANTHROPIC_API_KEY` | For person audit | Claude API key (fact-checking) |
| `EPSTEIN_AUDITOR_VOYAGE_API_KEY` | Optional | Voyage AI (semantic search in auditor) |
| `EPSTEIN_AUDITOR_COHERE_API_KEY` | Optional | Cohere (reranking in auditor) |

## Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| **Neon Postgres** | Production website, semantic search | `export neon` |
| **JSON** | Static site generation, API consumption | `export json` |
| **CSV** | Research, spreadsheet analysis | `export csv` |
| **SQLite** | Local querying, offline research | `export sqlite` |
| **NDJSON** | Streaming, log-style processing | `export json --format ndjson` |

## Data Sources

All source data comes from publicly released government records and court documents:

| Source | URL | Content |
|--------|-----|---------|
| DOJ EFTA Library | https://www.justice.gov/epstein | 12 datasets, 2M+ files |
| FBI Vault | https://vault.fbi.gov/jeffrey-epstein | FBI records |
| CourtListener | https://www.courtlistener.com/docket/4355835/giuffre-v-maxwell/ | Court filings |
| House Oversight | https://oversight.house.gov | Congressional releases |
| DocumentCloud | https://www.documentcloud.org | Searchable court docs |
| Archive.org | https://archive.org/details/epstein-flight-logs-unredacted_202304 | Flight logs, mirrors |
| Kaggle | Various | Community-compiled datasets |

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for the complete list.

## Documentation

| Document | Description |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute (setup, workflow, standards) |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards and expectations |
| [SECURITY.md](SECURITY.md) | Security policy and data handling guidelines |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design decisions |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | All known public data sources |
| [docs/PROCESSORS.md](docs/PROCESSORS.md) | Processor reference (OCR, NER, dedup, etc.) |
| [docs/SITE_SYNC.md](docs/SITE_SYNC.md) | Syncing processed data to epsteinexposed.com |
| [docs/SEA_DOUGHNUT.md](docs/SEA_DOUGHNUT.md) | Sea_Doughnut research data integration |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

**No coding required:** Report data quality issues, suggest new data sources, review processed data.

**Code contributions:** Add downloaders, improve extraction accuracy, add export formats, fix bugs.

## Related Projects

- [epsteinexposed.com](https://epsteinexposed.com) -- The live website powered by this pipeline
- [rodrigopolo/epstein-doj-library-sha256](https://github.com/rodrigopolo/epstein-doj-library-sha256) -- SHA-256 integrity hashes for DOJ files
- [Epstein-Files](https://github.com/WikiLeaksLookup/Epstein-Files) -- DOJ file mirrors
- [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) -- Email graph explorer
- [Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) -- Community research dataset

## License

MIT License. See [LICENSE](LICENSE).
