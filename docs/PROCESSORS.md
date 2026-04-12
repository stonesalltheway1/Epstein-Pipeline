# Pipeline Processors

## OCR (`processors/ocr.py`)

Multi-backend PDF text extraction with automatic fallback chain and per-page confidence scoring.

**Auto mode chain:** PyMuPDF → PaddleOCR → Granite-Docling-258M → Surya → Docling

| Backend | Speed | GPU | Best For |
|---------|-------|-----|----------|
| `pymupdf` | Instant | No | Digital PDFs with text layers |
| `paddleocr` | ~12s/page | No | Scanned docs (94.5% OmniDocBench); primary fallback |
| `granite-docling` | VLM | Optional (500MB) | Structure-aware (tables, forms, charts) |
| `smoldocling` | 0.35s/page | Optional (500MB) | Legacy — superseded by Granite-Docling |
| `surya` | Fast | Optional | 90+ languages, structured output |
| `olmocr` | Slow | Yes (8GB+) | Handwriting, degraded scans |
| `docling` | Medium | No | Complex layouts, table extraction |

```bash
epstein-pipeline ocr ./pdfs --backend auto              # Automatic fallback chain
epstein-pipeline ocr ./pdfs --backend paddleocr         # PaddleOCR PP-OCRv5
epstein-pipeline ocr ./pdfs --backend granite-docling   # Granite-Docling-258M VLM
epstein-pipeline ocr ./pdfs --backend surya             # Surya (90+ languages)
epstein-pipeline ocr ./pdfs --workers 8                 # Parallel processing
```

### PaddleOCR CLI workaround (Windows)

`paddlepaddle 3.2.0` + `paddleocr 3.4.0` on Windows exhibits a silent-exit bug
after the first successful Python-API OCR call: the native oneDNN layer calls
`std::abort()` / `ExitProcess(0)` and bypasses Python's exception machinery
entirely (no traceback, no error message). Known matches: Paddle issues
#61724, #60251, PaddleOCR #14654/#14892.

The `paddleocr` CLI runs in a fresh subprocess per invocation and does not
hit this bug. For reliable batch work, use the wrapper:

```bash
python scripts/ocr-via-cli.py path/to/a.pdf path/to/b.pdf
```

This spawns the CLI per doc, collects the per-page JSON outputs, and writes
the `.txt` + `.meta.json` sidecar cache files that
`scripts/ingest-featured-releases.py` reads on resume. Env flags tried but
insufficient: `FLAGS_use_mkldnn=0`, `FLAGS_call_stack_level=2`,
`PYTHONFAULTHANDLER=1`.

## Transcription (`processors/transcriber.py`)

Audio/video transcription with optional speaker diarization.

**Dual backend:**
- **faster-whisper** (default): GPU-accelerated, `large-v3-turbo` model. Auto INT8 quantization on ≤8GB VRAM GPUs.
- **WhisperX** (`--diarize`): Word-level timestamps + pyannote-audio 3.1 speaker diarization.

**Supported formats:** `.mp3, .mp4, .wav, .m4a, .avi, .wmv, .flac, .ogg, .webm, .mov`

**Output:** JSON with timestamped segments, speaker labels, confidence scores.

```bash
# Basic transcription (GPU auto-detected)
epstein-pipeline transcribe ./media --model large-v3-turbo

# With speaker diarization
epstein-pipeline transcribe ./media --diarize --hf-token $HF_TOKEN

# Control speaker count
epstein-pipeline transcribe ./media --diarize --min-speakers 2 --max-speakers 5
```

## Document Classification (`processors/classifier.py`)

Zero-shot classification into 12 legal document categories.

**Dual backend:**
- **GLiClass-ModernBERT** (default): 50x faster than BART, 8K token context. Model: `knowledgator/gliclass-modern-base-v3.0`
- **BART-large-mnli** (legacy fallback): Slower but well-tested.

**Categories:** legal, financial, travel, communications, investigation, media, government, personal, medical, property, corporate, intelligence

```bash
epstein-pipeline classify --input-dir ./output/
```

## Structured Extraction (`processors/structured_extractor.py`)

LLM-powered extraction of structured fields using Instructor + Pydantic schemas.

**Extracts:**
- Case references (case number, court, parties)
- Financial amounts (amount, currency, context, from/to entities)
- Persons with roles (name, role, organization)
- Dated events (date, event description, location)
- Locations (name, type, context)

**Backends:** Ollama (free, local), OpenAI, Anthropic

```bash
epstein-pipeline extract-structured ./docs/ --backend ollama --model llama3.2
epstein-pipeline extract-structured ./docs/ --backend openai --model gpt-4o-mini
```

## Entity Extraction (`processors/entities.py`)

Hybrid NER pipeline: spaCy transformers + GLiNER/GLiNER2 zero-shot + regex patterns.

**Entity types:** PERSON, ORG, GPE, DATE, MONEY, LOC, PHONE, EMAIL_ADDR, ACCOUNT, ADDRESS, CASE_NUMBER, FLIGHT_ID, BATES_NUMBER

**Four NER backends** (controlled via `EPSTEIN_NER_BACKEND`):
- **spacy** — `en_core_web_trf` transformer NER
- **gliner** — GLiNER v1 zero-shot (`urchade/gliner_multi_pii-v1`)
- **gliner2** — GLiNER2 unified NER (`fastino/gliner2-base-v1`) with entity descriptions
- **both** — Union merge from spaCy + GLiNER (default)

**Optional coreference resolution** (`--enable-coref`):
- Pre-NER pronoun resolution using fastcoref (FCoref or LingMessCoref)
- Resolves "he", "she", "they" to named entities for 30-50% more mentions
- Install: `pip install 'epstein-pipeline[nlp-coref]'`

```bash
epstein-pipeline extract-entities ./output/ocr --entity-types PERSON,ORG,GPE
epstein-pipeline extract-entities ./output/ocr --enable-coref
```

## Person Linker (`processors/person_linker.py`)

Links extracted entity mentions to the 1,723-person registry using rapidfuzz fuzzy matching (token_sort_ratio, threshold 85%). Multi-word names only — single-word names are never auto-linked to prevent false positives.

## Deduplication (`processors/dedup.py`)

Three-pass deduplication pipeline:
1. **Exact hash** — SHA-256 content hash for identical files
2. **MinHash/LSH** — O(n) near-duplicate detection for OCR variants
3. **Semantic similarity** — Embedding cosine similarity for reformatted duplicates

```bash
epstein-pipeline dedup ./output/ --mode all         # All three passes
epstein-pipeline dedup ./output/ --mode exact        # Hash only (fast)
epstein-pipeline dedup ./output/ --mode minhash      # Near-duplicate only
```

## Embeddings (`processors/embeddings.py`)

Vector embeddings using nomic-embed-text-v2-moe (768-dim, Matryoshka to 256-dim). MoE architecture activates 305M of 475M params for efficiency.

```bash
epstein-pipeline embed ./output/ -o ./embeddings/ --format neon
```

## Semantic Chunking (`processors/chunker.py`)

Paragraph-aware text splitting. Targets 450 tokens per chunk with 50-token overlap. Includes contextual prefixes (document title + source) per chunk for retrieval quality.

## Redaction Analysis (`processors/redaction.py`)

Detects redaction regions in PDFs and classifies them:
- **proper** — No text found under the redaction
- **bad_overlay** — Text accessible in the PDF stream
- **recoverable** — Text extractable from under the redaction

```bash
epstein-pipeline analyze-redactions ./pdfs --output ./output/redactions
```

## Image Extraction (`processors/image_extractor.py`)

Extracts embedded images from PDFs using PyMuPDF. Optionally describes them using AI vision models (Ollama llava or OpenAI gpt-4o-mini).

```bash
epstein-pipeline extract-images ./pdfs --output ./output/images --describe
```

## Summarization (`processors/summarizer.py`)

LLM-based document summarization via Ollama (local, free) or OpenAI (cloud). Generates concise descriptions of legal documents for search results and person profiles.

## Knowledge Graph (`processors/knowledge_graph.py`)

Builds weighted entity-relationship graphs from documents, flights, and emails.

**Edge types:** co-occurrence, co-passenger, correspondence

**Export formats:** JSON (D3.js), GEXF (Gephi)

```bash
epstein-pipeline build-graph ./output/entities --format both
```

## PLIST Forensics (`processors/plist_forensics.py`)

Scans PDFs for embedded Apple Mail PLIST metadata. Some DOJ documents contain hidden email headers, sender/recipient data, and timestamps.

```bash
epstein-pipeline forensics plist ./pdfs --output ./output/plist
```

## Temporal Event Extraction (`processors/temporal_extractor.py`)

LLM-powered timeline extraction from depositions, legal documents, and correspondence.

**Features:**
- Chunks long documents with overlap, extracts events per chunk, deduplicates across chunks
- Date normalization (natural language → YYYY-MM-DD via `python-dateutil`)
- 17 event types: meeting, flight, transaction, communication, legal_proceeding, arrest, testimony, deposition, court_filing, property_transaction, employment, travel, social_event, abuse_allegation, investigation, media_report, other
- Confidence scoring: 0.9+ for explicit dates, 0.5-0.8 for approximate, 0.3-0.5 for vague

**Backends:** Ollama (free, local), OpenAI, Anthropic — same Instructor + Pydantic pattern as structured extraction.

```bash
epstein-pipeline extract-events ./output/ocr --backend openai --confidence 0.5
epstein-pipeline extract-events ./output/ocr --backend ollama -o ./output/events
```

Events stored in Neon `temporal_events` table with FTS, GIN indexes, and `timeline_search()` SQL function.

## Entity Resolution (`processors/entity_resolution.py`)

Probabilistic person deduplication using [Splink 4](https://github.com/moj-analytical-services/splink) with DuckDB backend.

**How it works:**
- Fellegi-Sunter probabilistic model — no training data required
- JaroWinkler comparisons on name, first name, last name, and aliases
- ExactMatch on category
- Blocking rules to avoid O(n²) comparisons
- EM training for m/u probability estimation
- Outputs: entity clusters + merge map (old_id → canonical_id)

```bash
epstein-pipeline resolve-entities -r ./data/persons-registry.json
epstein-pipeline resolve-entities -r ./data/persons-registry.json --threshold 0.9
```

## Neo4j Knowledge Graph Export (`exporters/neo4j_export.py`)

Exports the in-memory knowledge graph to a Neo4j graph database using async batch MERGE operations.

**Node labels:** Person, Organization, Location, Document, Entity (fallback)
**Relationship types:** CO_OCCURRENCE, CO_PASSENGER, CORRESPONDENCE, FLEW_WITH, EMPLOYED_BY, ASSOCIATED_WITH, PARTY_TO, WITNESS_IN, DEFENDANT_IN, FINANCIAL_LINK, FAMILY_MEMBER, LEGAL_COUNSEL

```bash
epstein-pipeline export-neo4j ./output/entities --neo4j-uri bolt://localhost:7687
```

Includes uniqueness constraints, retry with exponential backoff, and `clear_all()` for full reloads.

## Confidence Scoring

Numeric confidence values for entity-person matches:

| Match Type | Confidence |
|-----------|-----------|
| Exact canonical name | 1.00 |
| Exact alias | 0.95 |
| Fuzzy > 95% | 0.85 |
| Fuzzy > 90% | 0.75 |
| Substring | 0.60 |

## Person Integrity Auditor (`audit/person_auditor.py`)

5-phase automated data quality pipeline:
1. **Dedup** — rapidfuzz similarity + alias cross-check
2. **Wikidata** — Cross-reference against Wikidata/Wikipedia
3. **Fact-Check** — Decompose bios into claims, verify via FTS
4. **Coherence** — Detect merged identities
5. **Score** — Composite severity (0-100)

```bash
epstein-pipeline audit-persons --min-severity 40 -o report.json
```
