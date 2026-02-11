# Architecture

## Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────┐
│  Download    │────>│   Process    │────>│    Validate    │────>│  Export   │
│             │     │              │     │                │     │           │
│ DOJ EFTA    │     │ OCR (Docling)│     │ Schema check   │     │ JSON      │
│ Kaggle      │     │ NER (spaCy)  │     │ Integrity      │     │ CSV       │
│ HuggingFace │     │ Dedup        │     │ Dedup report   │     │ SQLite    │
│ Archive.org │     │ Person link  │     │                │     │           │
└─────────────┘     └──────────────┘     └────────────────┘     └───────────┘
                           │
                    ┌──────┴──────┐
                    │  Persons    │
                    │  Registry   │
                    │ (1,400+     │
                    │  known      │
                    │  persons)   │
                    └─────────────┘
```

## Directory Structure

```
src/epstein_pipeline/
├── cli.py              # Click CLI entry point
├── config.py           # Pydantic settings
├── downloaders/        # Source-specific data fetchers
├── processors/         # Core processing pipeline
│   ├── ocr.py          # PDF → text via IBM Docling
│   ├── entities.py     # Named entity recognition
│   ├── dedup.py        # Duplicate detection
│   ├── person_linker.py # Name → person ID matching
│   └── summarizer.py   # Optional AI summarization
├── validators/         # Data quality enforcement
├── exporters/          # Output format converters
├── models/             # Pydantic data models
└── utils/              # Shared utilities
```

## Key Design Decisions

### Why Docling for OCR?
IBM's Docling (52K GitHub stars) provides the best open-source PDF extraction available. It understands page layout, reading order, tables, and formulas. It runs on a standard laptop without GPU. MIT licensed.

### Why spaCy for NER?
spaCy is the industry standard for production NLP. Fast, accurate English NER out of the box. The `en_core_web_sm` model is only 12MB.

### Why Pydantic for Models?
Pydantic v2 gives us validation, JSON Schema generation, and serialization in one package. Models directly match the TypeScript interfaces on epsteinexposed.com, making data exchange seamless.

### Why Click for CLI?
Click produces clean, composable CLI interfaces with good --help text. It's the standard for Python CLI tools.

## Data Flow

1. **Download** — Fetch raw source files (PDFs, CSVs, JSON) from public sources
2. **OCR** — Extract text from PDFs using Docling. Skip already-processed files via content hash.
3. **Entity Extraction** — Run spaCy NER on extracted text, match against persons registry
4. **Person Linking** — Map extracted names to canonical person IDs using fuzzy matching
5. **Deduplication** — Find duplicate documents via title similarity, Bates range overlap, content hash
6. **Validation** — Verify schema compliance, data integrity, cross-references
7. **Export** — Output in JSON (site-compatible), CSV (researchers), or SQLite (advanced queries)
