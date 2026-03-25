# Pipeline Processors

## OCR (`processors/ocr.py`)

Multi-backend PDF text extraction with automatic fallback chain and per-page confidence scoring.

**Auto mode chain:** PyMuPDF → SmolDocling-256M → Surya → Docling

| Backend | Speed | GPU | Best For |
|---------|-------|-----|----------|
| `pymupdf` | Instant | No | Digital PDFs with text layers |
| `smoldocling` | 0.35s/page | Optional (500MB) | Scanned docs, tables, charts, forms |
| `surya` | Fast | Optional | 90+ languages, structured output |
| `olmocr` | Slow | Yes (8GB+) | Handwriting, degraded scans |
| `docling` | Medium | No | Complex layouts, table extraction |

```bash
epstein-pipeline ocr ./pdfs --backend auto          # Automatic fallback chain
epstein-pipeline ocr ./pdfs --backend smoldocling   # SmolDocling-256M VLM
epstein-pipeline ocr ./pdfs --backend surya         # Surya (90+ languages)
epstein-pipeline ocr ./pdfs --workers 8             # Parallel processing
```

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

Hybrid NER pipeline: spaCy transformers + GLiNER zero-shot + regex patterns.

**Entity types:** PERSON, ORG, GPE, DATE, MONEY, LOC, PHONE, EMAIL_ADDR, ACCOUNT, ADDRESS, CASE_NUMBER, FLIGHT_ID, BATES_NUMBER

**Three-pass extraction:**
1. spaCy `en_core_web_trf` NER for standard entity types
2. GLiNER zero-shot for custom legal entities
3. Regex patterns for case numbers, flight IDs, Bates numbers, phone numbers

```bash
epstein-pipeline extract-entities ./output/ocr --entity-types PERSON,ORG,GPE
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
