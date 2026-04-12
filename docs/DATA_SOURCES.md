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

**Primary role as of Apr 2026**: Archive.org is the canonical resilient mirror for DOJ DS1-DS12 and House Oversight Committee productions. The site's `pdfUrl` columns migrated 1.4M+ records away from `justice.gov` (Akamai WAF fragility) to `archive.org` (served via IA's zip-extraction URLs plus `/api/pdf-proxy`).

Key items:
| Identifier | Content | Size |
|------------|---------|------|
| `epstein_library_transparency_act_hr_4405_dataset1_20260204` | DOJ DS1 (HR 4405) | 1.3 GB |
| `epstein_library_transparency_act_hr_4405_dataset8` | DOJ DS8 | 10.7 GB |
| `epstein_library_transparency_act_hr_4405_dataset9_202602` | DOJ DS9 | 107 GB |
| `epstein_library_transparency_act_hr_4405_dataset10_202605` | DOJ DS10 | 84 GB |
| `epstein_library_transparency_act_hr_4405_dataset11_202602` | DOJ DS11 | 27 GB |
| `data-set-12_20260131` | DOJ DS12 | 120 MB |
| `oversight-committee-additional-epstein-files` | House Oversight Nov 12, 2025 (23K page JPGs + .dat metadata) | 25 GB |
| `Epstein_Estate_Documents_-_Seventh_Production` | Oversight Seventh Production | 39 GB |
| `ds-9-efta-gap-repair` | DS9 gap-fill PDFs | 70 MB |
| `efta-19-dec-2025` | Dec 19, 2025 HR 4405 batch | 5.1 GB |

Downloader: `downloaders/archive_org.py`, `scripts/link-doj-datasets-to-ia.py`, `scripts/link-oversight-to-ia.py`.
Metadata parsing for Relativity `.dat` load files: `scripts/parse-oversight-relativity.py`.

Also: media files (photos, videos, audio) from older collections via `epstein-pipeline download archive`.

## CourtListener / RECAP

Federal court filings mirrored from PACER by the Free Law Project.

- **Downloader:** `downloaders/courtlistener.py` — free-tier search API (token required, registration at courtlistener.com/help/api/)
- **Coverage:** 342 documents across 251 Epstein-related dockets (Giuffre v. Maxwell, USVI v. JPMorgan, Doe 1 v. JPMorgan, U.S. v. Maxwell, Edwards v. Dershowitz, Deutsche Bank settlements, etc.)
- **Note:** Free tier blocks `/docket-entries/` and `/recap-documents/` list endpoints with 403. Use `/search/?type=r&q=docket_id:{id}` instead.

## SEC EDGAR

Public company disclosures for entities in the Epstein financial network.

- **Downloader:** `downloaders/sec_edgar.py` — free EDGAR API (compliant User-Agent required)
- **Targets:** JPMorgan Chase (CIK 0000019617), Deutsche Bank (0001159508), Bath & Body Works / L Brands (0000701985), Victoria's Secret (0001856437)
- **Filings pulled:** 10-K, 13D, 13F, 8-K, Form 4 — currently 80 documents in the database

## ProPublica Nonprofit Explorer

Form 990 filings for Epstein-controlled and associate foundations.

- **Downloader:** `downloaders/propublica_nonprofits.py`
- **Curated list:** 16 EINs — J Epstein Foundation, C.O.U.Q., Enhanced Education, Wexner Foundation, Wexner Heritage Foundation, Mark Epstein Foundation, Clinton Foundation, Harvard, MIT, Melanoma Research Alliance, etc.
- **Note:** JSON metadata is rich (officers, grants, revenues); raw 990 PDFs are gated with 403 from ProPublica's proxy even with `Referer` header — metadata-only ingest works well.
- **Coverage:** 33 orgs + 463 Form 990 filings

## Senate Finance & Oversight

Congressional investigation letters and disclosures — ingested individually via `scripts/ingest-featured-releases.py`.

| Release | Date | Source URL |
|---------|------|-----------|
| Wyden SARs (Treasury Suspicious Activity Reports) | 2025-11-21 | whitehouse.senate.gov |
| Wyden letter to DEA re. mystery Epstein investigation | 2026-02-25 | finance.senate.gov |
| Merkley / Murkowski / Luján / Durbin GAO audit referral | 2026-03-11 | merkley.senate.gov |

## SDNY Court (Direct)

Orders / opinions published directly by SDNY clerk's office — ingested via featured-releases:

| Release | Docket | Date |
|---------|--------|------|
| Maxwell grand jury unsealing opinion | 20-cr-00330 | 2026-01-21 |

## DOJ Maxwell Interview Transcripts (Aug 2025)

- `doj-maxwell-blanche-interview-2025-07-24` — Day 1 (~263pp redacted)
- `doj-maxwell-blanche-interview-2025-07-25` — Day 2 (~66pp)
- **Source:** `justice.gov/storage/audio-files/Interview%20Transcript/` (works without auth but serves 403 on landing page to automated UAs)

## Community Sources

| Source | Description |
|--------|-------------|
| [rhowardstone/Epstein-research-data](https://github.com/rhowardstone/Epstein-research-data) | 1.38M processed documents, full-text search, provenance mapping |
| [Epstein-doc-explorer](https://github.com/nicholasgasior/Epstein-doc-explorer) | Graph explorer of emails |
| [epstein-docs.github.io](https://epstein-docs.github.io) | 8,186 analyzed documents with AI summaries |

## How to Add a New Source

1. Open an issue using the "New Data Source" template
2. Or implement a downloader in `src/epstein_pipeline/downloaders/` and submit a PR
