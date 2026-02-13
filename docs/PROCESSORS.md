# Pipeline Processors

## OCR (`processors/ocr.py`)

PDF text extraction with dual backend support.

**Backends:**
- `docling` (default) - IBM Docling for structured document conversion
- `pymupdf` - PyMuPDF for fast text extraction (better for invisible OCR layers)
- `both` - Try PyMuPDF first, fall back to Docling

**Parallel processing:** Uses `ProcessPoolExecutor` for CPU-bound OCR when `--workers > 1`.

```bash
epstein-pipeline ocr ./pdfs --backend both --workers 8
```

## Entity Extraction (`processors/entities.py`)

Named entity recognition using spaCy with person registry matching.

**Entity types:** PERSON, ORG, GPE, DATE, MONEY, LOC, PHONE, EMAIL_ADDR, ACCOUNT, ADDRESS

**Two-pass extraction:**
1. spaCy NER detects entities, matches PERSON entities against registry
2. Direct name scan finds names spaCy missed

```bash
epstein-pipeline extract-entities ./output/ocr --entity-types PERSON,ORG,GPE
```

## Redaction Analysis (`processors/redaction.py`)

Detects redaction regions (filled black rectangles) in PDFs and classifies them:

- **proper** - No text found under the redaction
- **bad_overlay** - Text is accessible in the PDF stream
- **recoverable** - Text can be extracted from under the redaction

```bash
epstein-pipeline analyze-redactions ./pdfs --output ./output/redactions
```

## Image Extraction (`processors/image_extractor.py`)

Extracts embedded images from PDFs using PyMuPDF. Optionally describes them using AI vision models (Ollama llava or OpenAI gpt-4o-mini).

```bash
epstein-pipeline extract-images ./pdfs --output ./output/images --describe
```

## Transcription (`processors/transcriber.py`)

Audio/video transcription using faster-whisper. Supports: mp3, mp4, wav, m4a, avi, wmv, flac.

```bash
epstein-pipeline transcribe ./media --model large-v3
```

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

## Confidence Scoring (`processors/confidence.py`)

Numeric confidence values for entity-person matches:

| Match Type | Confidence |
|-----------|-----------|
| Exact canonical name | 1.00 |
| Exact alias | 0.95 |
| Fuzzy > 95% | 0.85 |
| Fuzzy > 90% | 0.75 |
| Substring | 0.60 |

## Person Linker (`processors/person_linker.py`)

Fast substring-based person linking (no spaCy dependency). Links document text fields (title, summary, OCR) to person IDs.

## Summarizer (`processors/summarizer.py`)

AI document summarization via Ollama (local) or OpenAI (cloud).
