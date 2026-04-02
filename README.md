# Epstein Pipeline

[![CI](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source document processing pipeline for the Jeffrey Epstein case files. Downloads, OCRs, transcribes video depositions, extracts entities, deduplicates, classifies, embeds, and exports **2.1 million+ documents** to Neon Postgres with pgvector semantic search.

**This is the data engine behind [epsteinexposed.com](https://epsteinexposed.com)** --the most comprehensive searchable database of the Epstein files.

## What It Does

```
 ┌─ DOJ EFTA Releases (DS1-DS12)
 ├─ Kaggle / HuggingFace / Archive
 ├─ Video Depositions (justice.gov)          Each stage is a CLI command.
 └─ DS10 Seized Media (826 files)            Chain them together or run
         │                                   individually as needed.
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  download                                                   │
  │  Fetch raw PDFs, media, and metadata from public sources    │
  └────────────────────────────┬────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
  ┌──────────────────┐  ┌───────────┐  ┌──────────────────────┐
  │  ocr             │  │ transcribe│  │ extract-entities     │
  │  PyMuPDF →       │  │ WhisperX /│  │ spaCy trf + GLiNER   │
  │  SmolDocling →   │  │ faster-   │  │ + regex patterns     │
  │  Surya → olmOCR  │  │ whisper   │  │ Persons, orgs, dates │
  │  → Docling       │  │ + pyannote│  │ money, case numbers  │
  │  (auto fallback) │  │ diarize   │  │                      │
  └────────┬─────────┘  └─────┬─────┘  └──────────┬───────────┘
           │                  │                    │
           ▼                  ▼                    ▼
  ┌──────────────────┐  ┌───────────┐  ┌──────────────────────┐
  │  classify        │  │ dedup     │  │ extract-structured   │
  │  GLiClass-       │  │ Hash →    │  │ Instructor + Pydantic│
  │  ModernBERT      │  │ MinHash → │  │ Case refs, amounts,  │
  │  12 categories   │  │ Semantic  │  │ persons, dates, locs │
  │  (50x faster)    │  │           │  │ (Ollama / OpenAI)    │
  └────────┬─────────┘  └─────┬─────┘  └──────────┬───────────┘
           │                  │                    │
           └──────────────────┼────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  embed                                                      │
  │  Semantic Chunker → nomic-embed-text-v2-moe embeddings      │
  │  Paragraph-aware splitting, 768-dim / 256-dim Matryoshka    │
  └────────────────────────────┬────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
  ┌──────────────────┐  ┌───────────┐  ┌──────────────────────┐
  │  export-neon     │  │ export    │  │ export-kg / neo4j    │
  │  Neon Postgres   │  │ json/csv  │  │ Knowledge Graph      │
  │  + pgvector      │  │ sqlite    │  │ GEXF + JSON + Neo4j  │
  │  cosine ANN      │  │ ndjson    │  │ LLM extraction       │
  └──────────────────┘  └───────────┘  └──────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
  ┌──────────────────┐  ┌───────────┐  ┌──────────────────────┐
  │  extract-events  │  │ resolve-  │  │ timeline-search()    │
  │  Temporal event  │  │ entities  │  │ Neon SQL function    │
  │  extraction via  │  │ Splink 4  │  │ for date-range +     │
  │  LLM + Pydantic  │  │ + DuckDB  │  │ participant queries  │
  └──────────────────┘  └───────────┘  └──────────────────────┘
```

> **How it works:** Each box is a CLI command (`epstein-pipeline <command>`). Run them in sequence or individually. Each stage reads the previous stage's JSON output. No stage is fully automatic -- you trigger each step and can inspect the output before proceeding.

## Current Scale

| Metric | Count |
|--------|-------|
| Documents ingested | 2,146,000+ |
| OCR texts extracted | 2,013,000+ |
| Persons identified | 1,570+ |
| Document-person links | 2,286,000+ |
| SHA-256 integrity hashes | 1,380,000+ |
| DOJ datasets processed | 12 of 12 (DS1-DS12) |
| Video/audio files cataloged | 826 (DS10 seized media) |
| Deposition audio transcribed | 6.1 hours (Maxwell DOJ interview) |
| Deposition transcript segments | 4,510 |
| House Oversight docs indexed | 38,000+ |
| Embedding chunks (pgvector) | 2,670,000+ |

## Quickstart

```bash
# Install with all features
pip install "epstein-pipeline[all]"
python -m spacy download en_core_web_sm

# Download from a source
epstein-pipeline download doj

# OCR with automatic backend selection
epstein-pipeline ocr ./raw-pdfs/ --output ./processed/

# Transcribe video/audio with GPU (optional speaker diarization)
epstein-pipeline transcribe ./media/ --output ./transcripts/ --model large-v3-turbo
epstein-pipeline transcribe ./media/ --diarize --hf-token $HF_TOKEN

# Download and transcribe video depositions
epstein-pipeline download-depositions --list
epstein-pipeline download-depositions --source archive
epstein-pipeline download-depositions --catalog-ds10 E:/epstein-ds10/extracted

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
| **OCR** | SmolDocling-256M | 0.35s/page | High (tables, charts, forms) | Optional (500MB) |
| **OCR** | Surya | Fast | High (90+ langs) | Optional |
| **OCR** | olmOCR 2 | Slow | Highest (VLM) | Yes (8GB+) |
| **OCR** | Docling (IBM) | Medium | High (complex layouts) | No |
| **Transcription** | faster-whisper | 10-15x realtime | High | Optional |
| **Transcription** | WhisperX + pyannote | 70x realtime | High + speaker IDs | Yes |
| **NER** | spaCy `en_core_web_trf` | Fast | High | Optional |
| **NER** | GLiNER v1 | Medium | High (zero-shot) | Optional |
| **NER** | GLiNER2 | Medium | Higher (unified) | Optional |
| **Coref** | fastcoref (FCoref/LingMess) | Fast | High | Optional |
| **Dedup** | Content hash + fuzzy | Instant | Exact only | No |
| **Dedup** | MinHash/LSH | O(n) | Near-duplicate | No |
| **Dedup** | Semantic embeddings | Slow | OCR-variant | Optional |
| **Embeddings** | nomic-embed-text-v2-moe | Fast | SOTA | Optional |
| **Classifier** | GLiClass-ModernBERT | Very fast (50x) | High (8K context) | Optional |
| **Classifier** | BART-large-mnli | Medium | Good (legacy) | Optional |
| **Extraction** | Instructor + Pydantic | LLM-dependent | High | No (uses Ollama/API) |
| **Temporal** | Instructor + Pydantic | LLM-dependent | High | No (uses Ollama/API) |
| **Entity Resolution** | Splink 4 + DuckDB | Fast | Probabilistic | No |
| **Graph Export** | Neo4j async driver | Fast | N/A | No |

## Video Deposition Indexing

The pipeline can download, transcribe, and index video depositions and audio interviews from the Epstein case.

### Known Deposition Sources

| Source | Content | Status |
|--------|---------|--------|
| Maxwell DOJ Interview | 11 WAV files, 2-day prison interview (July 2025) | ✅ Transcribed (6.1h, 52K words) |
| Maxwell House Oversight | Virtual deposition, Feb 2026 (invoked Fifth) | Available on C-SPAN |
| Clinton Depositions | Bill & Hillary Clinton, March 2026 | Available |
| Indyke/Kahn Depositions | Epstein estate co-executors, March 2026 | Available |
| DS10 Seized Media | 826 video/audio files from Epstein's devices | ✅ Cataloged (13GB) |
| Epstein SEC Deposition | 2010 SEC deposition (pleaded Fifth) | Archive.org |

### Transcription Features

- **GPU-accelerated** via faster-whisper with CUDA (tested on GTX 1660 SUPER, 6GB VRAM)
- **Auto INT8 quantization** for GPUs with ≤8GB VRAM --large-v3-turbo quality at medium-model memory
- **Speaker diarization** via WhisperX + pyannote-audio 3.1 (requires HuggingFace token)
- **Timestamped segments** with speaker labels, confidence scores
- **Searchable transcripts** exported to Neon with full-text search (tsvector/GIN indexes)

```bash
# List known deposition sources
epstein-pipeline download-depositions --list

# Download Maxwell prison interview (11 WAV files from justice.gov)
epstein-pipeline download-depositions --id vd-maxwell-interview-2025

# Catalog DS10 media files
epstein-pipeline download-depositions --catalog-ds10 E:/epstein-ds10/extracted

# Transcribe with GPU (auto-selects INT8 on ≤8GB VRAM)
epstein-pipeline transcribe ./media/ --model large-v3-turbo

# Transcribe with speaker diarization
epstein-pipeline transcribe ./media/ --diarize --hf-token $HF_TOKEN --min-speakers 2
```

### Database Schema (v4)

```sql
-- Deposition metadata
video_depositions (id, title, deponent, case_name, deposition_date,
                   duration_seconds, source_url, video_url, speaker_count,
                   segment_count, word_count, description)

-- Timestamped transcript segments with full-text search
deposition_segments (deposition_id, segment_index, start_time, end_time,
                     speaker, speaker_person_id, text, embedding,
                     tsv [generated tsvector for FTS])

-- HNSW index on embeddings for semantic search
-- GIN index on tsv for full-text keyword search
```

## Structured Extraction

LLM-powered extraction of structured fields from legal documents using [Instructor](https://github.com/567-labs/instructor) + Pydantic schemas. Works with Ollama (free, local), OpenAI, or Anthropic.

Extracts:
- **Case references** --case number, court, parties
- **Financial amounts** --amount, currency, context, from/to entities
- **Persons with roles** --name, role (attorney, witness, defendant), organization
- **Dates with events** --date, what happened, location
- **Locations** --name, type (address, property, city), context

```bash
# Uses Ollama by default (free, runs locally)
epstein-pipeline extract-structured ./documents/ --backend ollama --model llama3.2

# Or use OpenAI for higher accuracy
epstein-pipeline extract-structured ./documents/ --backend openai --model gpt-4o-mini
```

## Temporal Event Extraction

LLM-powered timeline extraction from depositions, legal documents, and correspondence. Extracts structured events with dates, participants, locations, and event types. Handles explicit dates, relative dates, date ranges, and imprecise references.

Event types: meeting, flight, transaction, communication, legal_proceeding, arrest, testimony, deposition, court_filing, property_transaction, employment, travel, social_event, abuse_allegation, investigation, media_report.

```bash
# Extract timeline events from processed documents
epstein-pipeline extract-events ./output/ocr --backend openai --confidence 0.5

# Use Ollama for free local extraction
epstein-pipeline extract-events ./output/ocr --backend ollama -o ./output/events
```

Events are stored in Neon with FTS and a `timeline_search()` SQL function for date-range + participant queries.

## Neo4j Knowledge Graph Export

Export the knowledge graph (persons, organizations, locations, documents, and all relationships) to a Neo4j graph database for Cypher queries, community detection, centrality analysis, and shortest-path traversal.

```bash
# Export to Neo4j (local or Aura cloud)
epstein-pipeline export-neo4j ./output/entities --neo4j-uri bolt://localhost:7687

# Or via environment variables
export EPSTEIN_NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
export EPSTEIN_NEO4J_PASSWORD=your-password
epstein-pipeline export-neo4j ./output/entities
```

## Entity Resolution (Splink)

Probabilistic person deduplication using [Splink 4](https://github.com/moj-analytical-services/splink) with DuckDB. Replaces naive fuzzy string matching with Fellegi-Sunter probabilistic record linkage across multiple fields (name, first/last name, aliases, category).

```bash
# Resolve duplicate persons in the registry
epstein-pipeline resolve-entities -r ./data/persons-registry.json

# With custom match threshold
epstein-pipeline resolve-entities -r ./data/persons-registry.json --threshold 0.9
```

Outputs a merge map and cluster report showing which person records refer to the same real-world entity.

## Installation

```bash
# Core only (no ML models)
pip install epstein-pipeline

# With OCR (CPU --Surya + SmolDocling)
pip install "epstein-pipeline[ocr-surya,pymupdf]"

# With OCR (GPU --olmOCR 2, requires CUDA)
pip install "epstein-pipeline[ocr-gpu]"

# With NLP (spaCy + GLiNER + coreference)
pip install "epstein-pipeline[nlp,nlp-gliner,nlp-coref]"

# With GLiNER2 (unified NER + classification + relations)
pip install "epstein-pipeline[nlp-gliner2]"

# With transcription (faster-whisper, CPU or GPU)
pip install "epstein-pipeline[transcription]"

# With transcription + speaker diarization (WhisperX + pyannote)
pip install "epstein-pipeline[transcription-diarize]"

# With fast classification (GLiClass-ModernBERT, 50x faster than BART)
pip install "epstein-pipeline[classify-fast]"

# With structured extraction (Instructor + Pydantic)
pip install "epstein-pipeline[structured]"

# With embeddings (sentence-transformers + torch)
pip install "epstein-pipeline[embeddings]"

# With Neon Postgres export (psycopg + pgvector)
pip install "epstein-pipeline[neon]"

# With Neo4j graph database export
pip install "epstein-pipeline[neo4j]"

# With Splink entity resolution (probabilistic dedup)
pip install "epstein-pipeline[splink]"

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
epstein-pipeline download depositions           # Download video depositions

# -- Video Depositions ---------------------------------------------
epstein-pipeline download-depositions --list    # List all known deposition sources
epstein-pipeline download-depositions --source archive  # Download from Archive.org
epstein-pipeline download-depositions --id vd-maxwell-interview-2025  # Specific deposition
epstein-pipeline download-depositions --catalog-ds10 ./ds10/extracted  # Catalog DS10 media

# -- Processing ----------------------------------------------------
epstein-pipeline ocr ./pdfs/ -o ./out/          # OCR (auto backend: PyMuPDF → SmolDocling → Surya → Docling)
epstein-pipeline ocr ./pdfs/ --backend surya    # OCR with specific backend
epstein-pipeline ocr ./pdfs/ --backend smoldocling  # SmolDocling-256M VLM OCR
epstein-pipeline transcribe ./media/ -o ./tx/   # Transcribe audio/video (faster-whisper)
epstein-pipeline transcribe ./media/ --diarize  # With speaker diarization (WhisperX)
epstein-pipeline extract-entities ./out/ -o ./e/ # NER extraction (spaCy + GLiNER)
epstein-pipeline classify --input-dir ./out/    # Document classification (GLiClass-ModernBERT)
epstein-pipeline dedup ./out/ --mode all        # 3-pass deduplication
epstein-pipeline embed ./out/ -o ./emb/         # Generate embeddings

# -- Export --------------------------------------------------------
epstein-pipeline export json ./out/ -o ./site/  # JSON for website
epstein-pipeline export csv ./out/ -o docs.csv  # CSV for researchers
epstein-pipeline export sqlite ./out/ -o ep.db  # SQLite database
epstein-pipeline export neon --input-dir ./out/ # Push to Neon Postgres

# -- Timeline & Entity Resolution ----------------------------------
epstein-pipeline extract-events ./out/ --backend openai  # Temporal event extraction
epstein-pipeline resolve-entities -r ./data/persons-registry.json  # Splink entity resolution
epstein-pipeline export-neo4j ./out/ --neo4j-uri bolt://localhost:7687  # Neo4j graph export

# -- Database ------------------------------------------------------
epstein-pipeline migrate                        # Run Neon schema migration (v4)
epstein-pipeline search "query text here"       # Semantic search (pgvector)

# -- Quality -------------------------------------------------------
epstein-pipeline validate ./out/                # Data quality checks
epstein-pipeline stats ./out/                   # Show processing statistics

# -- Sanctions & PEP Cross-Reference --------------------------------
epstein-pipeline check-sanctions               # Cross-check all persons vs OpenSanctions
epstein-pipeline import-sanctions ./output/sanctions/opensanctions-results.json

# -- Person Integrity Auditor -------------------------------------
epstein-pipeline audit-persons                  # Full 5-phase audit
epstein-pipeline audit-persons --person bill-clinton --dry-run
```

## Processors

### OCR (`processors/ocr.py`)
Multi-backend OCR with automatic fallback chain: PyMuPDF (text extraction) → SmolDocling-256M (fast VLM, 0.35s/page, tables/charts/forms) → Surya (90+ languages) → Docling (complex layouts). Per-page confidence scoring triggers fallback when quality is low. olmOCR available for explicit selection (GPU-heavy, best for handwriting).

### Transcription (`processors/transcriber.py`)
Dual-backend audio/video transcription:
- **faster-whisper** (default): GPU-accelerated, `large-v3-turbo` model with auto INT8 quantization for ≤8GB VRAM GPUs
- **WhisperX** (`--diarize`): Word-level timestamp alignment + pyannote-audio 3.1 speaker diarization

Supports: `.mp3, .mp4, .wav, .m4a, .avi, .wmv, .flac, .ogg, .webm, .mov`

### Entity Extraction (`processors/entities.py`)
Hybrid NER using spaCy transformer models + GLiNER/GLiNER2 zero-shot extraction + regex patterns. Extracts people, organizations, locations, dates, case numbers, flight IDs, financial amounts, and Bates numbers from legal documents. Optional coreference resolution (fastcoref) resolves pronouns before NER for 30-50% more entity mentions.

### Document Classification (`processors/classifier.py`)
Dual-backend zero-shot classification:
- **GLiClass-ModernBERT** (default): 50x faster than BART, 8K token context, state-of-the-art accuracy
- **BART-large-mnli** (legacy fallback): Proven but slower

Classifies into 12 legal document categories: legal, financial, travel, communications, investigation, media, government, personal, medical, property, corporate, intelligence.

### Structured Extraction (`processors/structured_extractor.py`)
LLM-powered extraction using Instructor + Pydantic schemas. Extracts case references, financial amounts, persons with roles, dated events, and locations. Works with Ollama (free), OpenAI, or Anthropic backends.

### Deduplication (`processors/dedup.py`)
Three-pass deduplication pipeline:
1. **Exact hash** --SHA-256 content hash for identical files
2. **MinHash/LSH** --O(n) near-duplicate detection for OCR variants
3. **Semantic similarity** --Embedding cosine similarity for reformatted duplicates

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
Links extracted entity mentions to known persons in the database using fuzzy name matching (rapidfuzz) with word boundary safety (multi-word names only to prevent false positives).

### Knowledge Graph (`processors/knowledge_graph.py`)
Builds entity relationship graphs from co-occurrence analysis and optional LLM-based relationship extraction. Exports to GEXF, JSON, and Neo4j formats.

### Temporal Event Extraction (`processors/temporal_extractor.py`)
LLM-powered timeline extraction using Instructor + Pydantic. Chunks documents with overlap, extracts structured events (date, participants, locations, event type), normalizes dates, and deduplicates across chunks. Supports Ollama, OpenAI, and Anthropic backends.

### Entity Resolution (`processors/entity_resolution.py`)
Probabilistic person deduplication using Splink 4 with DuckDB backend. Fellegi-Sunter model with JaroWinkler comparisons on name, first/last name, and aliases. Produces match clusters and a merge map for consolidating duplicate person records.

### Neo4j Export (`exporters/neo4j_export.py`)
Async Neo4j exporter using batch MERGE operations. Maps GraphNode types to Neo4j labels (Person, Organization, Location, Document) and GraphEdge types to Neo4j relationship types. Includes schema constraints, retry logic, and idempotent upserts.

### Plist Forensics (`processors/plist_forensics.py`)
Parses Apple plist files found in the Epstein device data for contact and metadata extraction.

## OpenSanctions Cross-Reference

Cross-references all 1,723+ persons against 100+ global sanctions, PEP, and watchlist datasets via the [OpenSanctions API](https://opensanctions.org/).

**Datasets checked:** OFAC SDN, EU Financial Sanctions, UN Security Council, UK HMT, Interpol Red Notices, PEP registries (Every Politician), ICIJ Offshore Leaks (mirrored), and 100+ more.

```bash
# Cross-check all persons
export EPSTEIN_OPENSANCTIONS_API_KEY="your-api-key"
epstein-pipeline check-sanctions

# Import results into Neon Postgres
epstein-pipeline import-sanctions ./output/sanctions/opensanctions-results.json
```

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

Requires: `EPSTEIN_AUDITOR_ANTHROPIC_API_KEY` + `EPSTEIN_NEON_DATABASE_URL`

## Environment Variables

All configuration is via environment variables prefixed with `EPSTEIN_`. No credentials are ever stored in code or config files.

| Variable | Required | Purpose |
|----------|----------|---------|
| `EPSTEIN_NEON_DATABASE_URL` | For DB export/search | Neon Postgres connection string |
| `EPSTEIN_OPENSANCTIONS_API_KEY` | For sanctions check | OpenSanctions API key |
| `EPSTEIN_AUDITOR_ANTHROPIC_API_KEY` | For person audit | Claude API key (fact-checking) |
| `EPSTEIN_NEO4J_URI` | For Neo4j export | Neo4j connection URI (bolt:// or neo4j+s://) |
| `EPSTEIN_NEO4J_PASSWORD` | For Neo4j export | Neo4j password |
| `EPSTEIN_WHISPER_MODEL` | Optional | Whisper model (default: `large-v3-turbo`) |
| `EPSTEIN_CLASSIFIER_MODEL` | Optional | Classifier model (default: `knowledgator/gliclass-modern-base-v3.0`) |
| `HF_TOKEN` | For diarization | HuggingFace token (pyannote model access) |
| `OPENAI_API_KEY` | For extraction/summary | OpenAI API key |
| `ANTHROPIC_API_KEY` | For extraction | Anthropic API key |

## Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| **Neon Postgres** | Production website, semantic search | `export neon` |
| **JSON** | Static site generation, API consumption | `export json` |
| **CSV** | Research, spreadsheet analysis | `export csv` |
| **SQLite** | Local querying, offline research | `export sqlite` |
| **NDJSON** | Streaming, log-style processing | `export json --format ndjson` |
| **Neo4j** | Graph traversal, community detection, Cypher queries | `export-neo4j` |

## Data Sources

All source data comes from publicly released government records and court documents:

| Source | URL | Content |
|--------|-----|---------|
| DOJ EFTA Library | https://www.justice.gov/epstein | 12 datasets, 2M+ files |
| DOJ Maxwell Interview | https://www.justice.gov/maxwell-interview | 16 WAV files, transcripts |
| FBI Vault | https://vault.fbi.gov/jeffrey-epstein | FBI records |
| CourtListener | https://www.courtlistener.com | Court filings |
| House Oversight | https://oversight.house.gov | Congressional depositions |
| Archive.org | https://archive.org | Flight logs, video mirrors |
| C-SPAN | https://www.c-span.org | Deposition recordings |
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

- [epsteinexposed.com](https://epsteinexposed.com) --The live website powered by this pipeline
- [rhowardstone/Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) --Community research dataset (1,530 transcripts, entity registry)
- [rodrigopolo/epstein-doj-library-sha256](https://github.com/rodrigopolo/epstein-doj-library-sha256) --SHA-256 integrity hashes for DOJ files
- [freelawproject/courtlistener](https://github.com/freelawproject/courtlistener) --Court data infrastructure
- [freelawproject/juriscraper](https://github.com/freelawproject/juriscraper) --PACER scraper
- [Epstein-Files](https://github.com/WikiLeaksLookup/Epstein-Files) --DOJ file mirrors
- [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) --Email graph explorer
