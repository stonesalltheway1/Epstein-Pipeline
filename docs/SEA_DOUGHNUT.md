# Sea_Doughnut Data Integration

## Overview

[Sea_Doughnut](https://github.com/rhowardstone/Epstein-research-data) (u/Sea_Doughnut_8853) is an independent research project that processed the **complete** DOJ Epstein document releases — all 12 datasets, 218 GB of source material. The v2 dataset includes:

- **1,380,941 documents** with full OCR text across all 12 DOJ datasets
- **2,731,825 pages** with per-page text content and FTS5 full-text search
- **849,655 redaction analyses** with proper/improper classification
- **1,530 audio/video transcripts** (375 with speech, 92,153 words)
- **1,538 persons** in a unified registry with fuzzy matching
- **106,514 SDNY_GM↔EFTA bridge mappings** linking prosecution Bates numbers
- **26-range provenance map** covering 99.999% of the corpus

## Database Schemas

### full_text_corpus.db (~6.1 GB)

The core database with all document text.

```sql
-- Document metadata (one row per EFTA PDF)
documents(efta_number TEXT PK, dataset INTEGER, total_pages INTEGER, file_size INTEGER)

-- Per-page OCR text content
pages(efta_number TEXT, page_number INTEGER, text_content TEXT, char_count INTEGER)
  -- PK: (efta_number, page_number)

-- Full-text search (FTS5)
pages_fts(efta_number, page_number, text_content)
```

### transcripts.db (~50 MB)

Audio/video transcription results (separate database).

```sql
transcripts(
    efta_number TEXT PK,
    file_path TEXT,
    file_type TEXT,        -- m4a, mp4, wav, etc.
    duration_secs REAL,
    language TEXT,          -- detected language
    transcript TEXT,        -- full transcript text
    word_count INTEGER,
    dataset_source TEXT     -- ds1, ds8, ds9, etc.
)
```

### redaction_analysis_v2.db

Redaction detection results.

```sql
-- Per-document summary
document_summary(
    efta_number TEXT PK,
    total_redactions INTEGER,
    bad_redactions INTEGER,     -- bad_overlay (see note below)
    proper_redactions INTEGER,
    has_recoverable_text BOOLEAN
)

-- Individual redaction regions
redactions(
    efta_number TEXT,
    page_number INTEGER,
    hidden_text TEXT,       -- recovered text if any
    confidence REAL,
    redaction_type TEXT     -- proper, bad_overlay
)
```

> **Note on "bad_overlay" redactions:** For DS9-11, the DOJ files are scanned documents with
> redaction bars baked into the JPEG pixels. The OCR layer was generated *over* the black bars,
> producing garbled text. The redaction detector flags these as "bad_overlay" with "recoverable text",
> but this is a **false positive** — the "recovered" text is garbled OCR of black bars, not the
> original redacted content. Genuinely useful hidden text comes from documents where OCR was done
> *before* redaction was applied (e.g., PLIST emails, EFTA00001932 victim letter).

### concordance_metadata.db (optional)

Cross-reference and provenance data.

```sql
-- Provenance map: what each EFTA range contains
provenance_map(
    dataset INTEGER, efta_start TEXT, efta_end TEXT,
    efta_start_num INTEGER, efta_end_num INTEGER,
    sdny_gm_start TEXT, sdny_gm_end TEXT,
    source_description TEXT, source_category TEXT,
    doc_count INTEGER, page_count INTEGER, confidence TEXT
)

-- Direct EFTA↔SDNY_GM Bates number mappings
sdny_efta_bridge(efta_number TEXT, sdny_gm_number TEXT)

-- Discovery production index (from cover letters)
productions(id INTEGER PK, description TEXT, ...)

-- OPT load file document index
opt_documents(efta_number TEXT PK, page_count INTEGER, ...)
```

### persons_registry.json

```json
[
    {
        "name": "Jeffrey Epstein",
        "slug": "jeffrey-epstein",
        "aliases": ["JE"],
        "category": "key-figure",
        "description": "Convicted sex trafficker",
        "search_terms": ["Jeffrey Epstein"],
        "sources": ["epstein-pipeline", "la-rana-chicana"]
    }
]
```

## Directory Layout

The importer expects this layout:

```
data-dir/
    full_text_corpus.db           # REQUIRED - 6.1 GB
    transcripts.db                # separate database (NOT inside corpus)
    redaction_analysis_v2.db      # redaction scores + recovered text
    concordance_metadata.db       # optional — provenance + SDNY bridge
    persons_registry.json         # optional — 1,538 persons
```

## Usage

```bash
# Import all data
epstein-pipeline import sea-doughnut --data-dir /path/to/sea-doughnut-data

# Import with output directory
epstein-pipeline import sea-doughnut -d ./sea-doughnut -o ./output/sea-doughnut

# Import with document limit (for testing)
epstein-pipeline import sea-doughnut -d ./sea-doughnut -l 1000
```

## What Gets Imported

| Data Type | Source DB | Source Table | Pipeline Model |
|-----------|----------|-------------|----------------|
| Documents | full_text_corpus.db | `documents` + `pages` | `Document` |
| Redaction scores | redaction_analysis_v2.db | `document_summary` | `RedactionScore` |
| Recovered text | redaction_analysis_v2.db | `redactions` (hidden_text) | `RecoveredText` |
| Transcripts | transcripts.db | `transcripts` | `Transcript` |
| Persons | persons_registry.json | — | `Person` |
| Concordance | concordance_metadata.db | `provenance_map` etc. | `ConcordanceSummary` |
| Images | — | — | not in DB (on-disk only) |
| Entities | — | — | run Pipeline's extract-entities after import |

## DOJ PDF URLs

Every imported document gets a `pdfUrl` linking to the DOJ source:

```
https://www.justice.gov/epstein/files/DataSet%20{N}/EFTA{XXXXXXXX}.pdf
```

The EFTA-to-dataset mapping covers all 12 releases:

| Dataset | EFTA Range | Documents |
|---------|-----------|-----------|
| 1 | 00000001–00003158 | ~3,158 |
| 2 | 00003159–00003857 | ~699 |
| 3 | 00003858–00005586 | ~1,729 |
| 4 | 00005705–00008320 | ~2,616 |
| 5 | 00008409–00008528 | ~120 |
| 6 | 00008529–00008998 | ~470 |
| 7 | 00009016–00009664 | ~649 |
| 8 | 00009676–00039023 | ~29,348 |
| 9 | 00039025–01262781 | ~1,223,757 |
| 10 | 01262782–02212882 | ~950,101 |
| 11 | 02212883–02730262 | ~517,380 |
| 12 | 02730265–02731783 | ~1,519 |

## Provenance Categories

Documents are categorized based on provenance:

| Source Category | Pipeline Category | Description |
|----------------|-------------------|-------------|
| `prosecution` | `legal` | SDNY prosecution working files |
| `prosecution_admin` | `legal` | Administrative prosecution docs |
| `financial_subpoena` | `financial` | Deutsche Bank, JPMorgan Chase subpoena returns |
| `telecom_subpoena` | `communications` | AT&T, pen registers, phone records |
| `device_extraction` | `communications` | Phone/computer forensic extractions |
| `investigation` | `investigation` | FBI/DOJ investigative files |
| `mixed_prosecution` | `legal` | Mixed prosecution materials |
