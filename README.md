# Epstein Pipeline

[![CI](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/stonesalltheway1/Epstein-Pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open source document processing pipeline for the [Epstein case files](https://epsteinexposed.com). Download, OCR, extract entities, deduplicate, and export 140,000+ documents from the DOJ releases.

**This is the data processing companion to [epsteinexposed.com](https://epsteinexposed.com)** — the most comprehensive searchable database of the Epstein files.

## Quickstart

```bash
# Install
pip install epstein-pipeline

# Download a dataset
epstein-pipeline download kaggle

# OCR documents
epstein-pipeline ocr ./raw-pdfs/ --output ./processed/

# Extract entities and link to known persons
epstein-pipeline extract-entities ./processed/ --output ./entities/

# Export for the website
epstein-pipeline export json ./processed/ --output ./export/
```

## What This Does

```
Raw DOJ PDFs ──> OCR ──> Entity Extraction ──> Deduplication ──> Export
                  │              │                    │              │
              Docling (IBM)   spaCy NER        rapidfuzz         JSON/CSV/SQLite
                              + 1,400+         fuzzy match
                              known persons
```

| Step | Tool | Description |
|------|------|-------------|
| **Download** | Built-in | Fetch from DOJ, Kaggle, HuggingFace, Archive.org |
| **OCR** | [Docling](https://github.com/docling-project/docling) (IBM) | Extract text from PDFs with layout understanding |
| **Entity Extraction** | [spaCy](https://spacy.io/) | Find person names, organizations, locations |
| **Person Linking** | [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) | Match names to 1,400+ known persons |
| **Deduplication** | rapidfuzz + content hashing | Find duplicate documents across sources |
| **Validation** | Pydantic | Schema validation, integrity checks |
| **Export** | Built-in | JSON (website), CSV (research), SQLite (queries) |

## Installation

### Basic (no OCR)

```bash
pip install epstein-pipeline
```

### With OCR support

```bash
pip install "epstein-pipeline[ocr]"
```

### With NLP (entity extraction)

```bash
pip install "epstein-pipeline[nlp]"
python -m spacy download en_core_web_sm
```

### Everything

```bash
pip install "epstein-pipeline[all]"
python -m spacy download en_core_web_sm
```

### Docker (includes all dependencies)

```bash
docker compose run pipeline --help
docker compose run pipeline ocr ./raw-pdfs/ --output ./output/
```

## CLI Commands

```bash
epstein-pipeline --help                          # Show all commands
epstein-pipeline download doj --dataset 9        # Download DOJ dataset 9
epstein-pipeline download kaggle                 # Download Kaggle dataset
epstein-pipeline download huggingface            # Download HuggingFace datasets
epstein-pipeline ocr ./pdfs/ -o ./out/           # OCR PDF files
epstein-pipeline extract-entities ./out/ -o ./e/  # Extract entities
epstein-pipeline dedup ./out/ -o report.json     # Find duplicates
epstein-pipeline validate ./out/                 # Validate data quality
epstein-pipeline export json ./out/ -o ./site/   # Export for website
epstein-pipeline export csv ./out/ -o docs.csv   # Export as CSV
epstein-pipeline export sqlite ./out/ -o ep.db   # Export as SQLite
epstein-pipeline stats ./out/                    # Show statistics
```

## Contributing

We welcome contributions from everyone! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

**No coding required:**
- Report data quality issues (wrong person matches, duplicates)
- Suggest new data sources
- Review and verify processed data
- Improve documentation

**Code contributions:**
- Add new data source downloaders
- Improve entity extraction accuracy
- Add export formats
- Fix bugs

## Data Sources

See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for all known public data sources.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline design details.

## Related Projects

- [epsteinexposed.com](https://epsteinexposed.com) — The live website powered by this pipeline
- [Epstein-Files](https://github.com/WikiLeaksLookup/Epstein-Files) — DOJ file mirrors and torrents
- [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) — Email graph explorer
- [Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) — Community research dataset

## License

MIT License. See [LICENSE](LICENSE).
