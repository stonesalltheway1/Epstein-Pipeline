# Data Sources

All publicly available data sources for the Epstein case files.

## DOJ EFTA Releases (Datasets 1-12)

The U.S. Department of Justice released documents from the Jeffrey Epstein estate through the EFTA (Electronic File Transfer Agreement) process via the SDNY U.S. Attorney's Office.

| Dataset | EFTA Range | Approx. Size | Documents | Content |
|---------|-----------|--------------|-----------|---------|
| 1 | 00000001–00003158 | ~2 GB | ~3,158 | Initial prosecution files |
| 2 | 00003159–00003857 | ~0.5 GB | ~699 | Additional prosecution docs |
| 3 | 00003858–00005586 | ~1 GB | ~1,729 | Investigation records |
| 4 | 00005705–00008320 | ~2 GB | ~2,616 | Financial/legal documents |
| 5 | 00008409–00008528 | ~0.1 GB | ~120 | Supplemental filings |
| 6 | 00008529–00008998 | ~0.3 GB | ~470 | Court records |
| 7 | 00009016–00009664 | ~0.5 GB | ~649 | Additional court records |
| 8 | 00009676–00039023 | ~15 GB | ~29,348 | Media, spreadsheets, devices |
| 9 | 00039025–01262781 | ~103 GB | ~1,223,757 | Largest release (prosecution working files) |
| 10 | 01262782–02212882 | ~65 GB | ~950,101 | Financial subpoenas (DB, JPMC), telecom |
| 11 | 02212883–02730262 | ~25 GB | ~517,380 | Additional financial/device data |
| 12 | 02730265–02731783 | ~2 GB | ~1,519 | Final supplemental release |

**Total:** ~218 GB, ~2.73M documents, ~1.38M unique EFTA numbers

**Access:** [DOJ EFTA Releases](https://www.justice.gov/usao-sdny/united-states-v-jeffrey-epstein)
**Mirrors:** [Archive.org collections](https://archive.org/search?query=Epstein+Dataset) (bulk downloads)

### Three Bates Numbering Systems

1. **EFTA########** — Public DOJ identifier (e.g., EFTA00039025)
2. **SDNY_GM_########** — SDNY prosecution internal number (visible in OCR text of many documents)
3. **DB-SDNY-########** — Deutsche Bank internal production number (in DS10 financial subpoena returns)

## Sea_Doughnut Processed Databases

Pre-processed research databases covering the complete DOJ release. See [SEA_DOUGHNUT.md](SEA_DOUGHNUT.md) for full schema documentation.

| Database | Size | Content |
|----------|------|---------|
| full_text_corpus.db | 6.1 GB | 1,380,941 docs, 2,731,825 pages, FTS5 index |
| transcripts.db | ~50 MB | 1,530 transcripts (375 with speech) |
| redaction_analysis_v2.db | ~2 GB | 849,655 docs, 2,587,102 redactions |
| concordance_metadata.db | ~580 MB | OPT/DAT concordance, SDNY bridge, provenance |
| persons_registry.json | ~1 MB | 1,538 persons with aliases |

- **Source:** [rhowardstone/Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data)
- **Import:** `epstein-pipeline import sea-doughnut --data-dir /path/to/data`

## Kaggle: Epstein Ranker Dataset

AI-analyzed documents with summaries and person matching.

- **URL:** https://www.kaggle.com/datasets/jamesgrantz/epstein-ranker
- **Size:** ~23,700 documents
- **Format:** CSV with AI-generated summaries
- **Download:** `epstein-pipeline download kaggle`

## HuggingFace Datasets

Structured email and document datasets.

| Dataset | Content | Records |
|---------|---------|---------|
| Epstein Emails | Structured email metadata | ~4,700 |
| EFTA Documents | Additional EFTA filings | ~50 |

- **Download:** `epstein-pipeline download huggingface`

## Archive.org Collections

Media files (photos, videos, audio) from various Epstein-related collections.

- **Collections:** FBI raid photos, court proceedings, property images
- **Format:** Mixed (JPEG, MP4, MP3)
- **Download:** `epstein-pipeline download archive`

## Community Sources

| Source | Description |
|--------|-------------|
| [rhowardstone/Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) | 1.38M processed documents, full-text search, provenance mapping |
| [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) | Graph explorer of emails |
| [epstein-docs.github.io](https://epstein-docs.github.io) | 8,186 analyzed documents with AI summaries |

## How to Add a New Source

1. Open an issue using the "New Data Source" template
2. Or implement a downloader in `src/epstein_pipeline/downloaders/` and submit a PR
