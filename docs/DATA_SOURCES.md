# Data Sources

All publicly available data sources for the Epstein case files.

## DOJ EFTA Releases (Datasets 1-12)

The U.S. Department of Justice released documents from the Jeffrey Epstein estate through the EFTA (Electronic File Transfer Agreement) process.

| Dataset | Description | Approx. Size | Documents |
|---------|-------------|--------------|-----------|
| 1-8 | Initial releases (2023-2024) | Varies | ~4,000 |
| 9 (VOL00009) | Largest single release | 57 GB | ~107,000 |
| 10-12 | Subsequent releases | Varies | TBD |

**Access:** [DOJ EFTA Releases](https://www.justice.gov/usao-sdny/united-states-v-jeffrey-epstein)
**Mirrors:** [Epstein-Files repo](https://github.com/WikiLeaksLookup/Epstein-Files) (torrent magnets, checksums)

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

| Source | Description | Contact |
|--------|-------------|---------|
| [rhowardstone/Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) | 519K processed PDFs, 107K entities, knowledge graph | u/Sea_Doughnut_8853 |
| [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) | Graph explorer of emails | 497 stars |
| [epstein-docs.github.io](https://epstein-docs.github.io) | 8,186 analyzed documents with AI summaries | Community |

## How to Add a New Source

1. Open an issue using the "New Data Source" template
2. Or implement a downloader in `src/epstein_pipeline/downloaders/` and submit a PR
