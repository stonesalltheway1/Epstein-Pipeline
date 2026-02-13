# Sea_Doughnut Data Integration

## Overview

[Sea_Doughnut](https://github.com/rhowardstone/Epstein-research-data) (u/Sea_Doughnut_8853) is an independent research project by a PhD CS researcher that processed the full DOJ Epstein document releases. Their v2 dataset includes:

- **1.38M documents** from full_text_corpus.db across all 12 DOJ datasets
- **638K redaction scores** with proper/improper classification
- **39.5K recovered text pages** from under redactions
- **107K extracted entities** (persons, organizations, locations)
- **38.9K extracted images** from PDFs
- **1,530 audio/video transcripts**

## Database Layout

The importer expects the following directory structure:

```
data-dir/
  full_text_corpus.db       # 6.1 GB - documents, transcripts, entities
  redaction_analysis_v2.db  # redaction scores + recovered text
  image_analysis.db         # extracted image metadata
  ocr_database.db           # OCR text fallback
  persons_registry.json     # 1,536 persons (optional)
```

## Usage

```bash
# Import all data
epstein-pipeline import sea-doughnut --data-dir E:/epstein-data/sea-doughnut-v2

# Import with output directory
epstein-pipeline import sea-doughnut -d ./sea-doughnut -o ./output/sea-doughnut

# Import with document limit (for testing)
epstein-pipeline import sea-doughnut -d ./sea-doughnut -l 1000
```

## What Gets Imported

| Data Type | Source Table | Pipeline Model |
|-----------|-------------|----------------|
| Documents | `documents` / `corpus` | `Document` |
| Redaction scores | `redaction_scores` | `RedactionScore` |
| Recovered text | `recovered_text` | `RecoveredText` |
| Images | `images` / `image_catalog` | `ExtractedImage` |
| Transcripts | `transcripts` | `Transcript` |
| Entities | `entities` | `ExtractedEntity` |

## Source Mapping

Sea_Doughnut documents are mapped to DOJ dataset-specific source types:

| Dataset | Source Type |
|---------|------------|
| Data Set 1 | `efta-ds1` |
| Data Set 2 | `efta-ds2` |
| ... | ... |
| Data Set 12 | `efta-ds12` |
