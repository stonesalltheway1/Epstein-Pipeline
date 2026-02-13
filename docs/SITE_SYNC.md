# Site Sync

The `sync-site` command exports pipeline output into the format expected by the [Epstein Exposed](https://epsteinexposed.com) Next.js site.

## Usage

```bash
epstein-pipeline sync-site \
  --input ./output \
  --site-dir ../epstein-index \
  --output-json \
  --output-sqlite
```

### Options

| Flag | Description |
|------|-------------|
| `--input` | Directory containing pipeline output (JSON results) |
| `--site-dir` | Path to the `epstein-index` Next.js project |
| `--output-json` | Export documents as JSON files to `data/` |
| `--output-sqlite` | Seed the SQLite database at `data/epstein.sqlite` |

## What Gets Exported

### JSON Export (`--output-json`)

Writes `documents.json` to the site's `data/` directory. Each document includes:
- `id`, `title`, `source`, `category`
- `personIds` (linked persons)
- `tags`, `summary`, `ocrText`
- `sourceUrl`, `archiveUrl`

### SQLite Export (`--output-sqlite`)

Seeds the site's SQLite database with:
- **documents** table + FTS5 virtual table for full-text search
- **persons** table
- **document_persons** junction table for fast JOINs
- **redaction_scores** — density and classification per document
- **recovered_text** — text extracted from under redactions
- **transcripts** — audio/video transcriptions
- **extracted_entities** — NER results (PERSON, ORG, GPE, etc.)
- **extracted_images** — image metadata from PDFs

## Registry Sync

To sync the person registry from the site back to the pipeline:

```bash
epstein-pipeline sync-registry \
  --from-site-path ../epstein-index/data/persons.ts
```

This reads the TypeScript `persons.ts` file via regex extraction and writes
`data/persons-registry.json` with all person IDs, slugs, names, aliases,
categories, and bios.

## Architecture

```
Pipeline Output          Site (epstein-index)
================         ====================
output/*.json    --->    data/documents.json
output/entities  --->    data/epstein.sqlite (extracted_entities)
output/redactions --->   data/epstein.sqlite (redaction_scores, recovered_text)
output/transcripts --->  data/epstein.sqlite (transcripts)
output/images    --->    data/epstein.sqlite (extracted_images)
```

The `SiteSyncer` class in `exporters/site_sync.py` handles the format
conversion. Documents are serialized with camelCase field names to match
the site's TypeScript interfaces.
