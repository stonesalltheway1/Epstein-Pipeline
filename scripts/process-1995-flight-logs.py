"""Process 1995 DOJ flight log PDFs through OCR and export to Neon.

Downloads 4 flight-log PDFs from the DOJ Epstein Library, runs them through
the Epstein-Pipeline OCR processor, saves JSON output, and inserts results
into the Neon Postgres database (ocr_text + documents tables).

Usage:
    cd E:\\Epstein-Pipeline
    python scripts/process-1995-flight-logs.py              # download + OCR + DB
    python scripts/process-1995-flight-logs.py --skip-download  # OCR from local PDFs
    python scripts/process-1995-flight-logs.py --backend surya  # force specific OCR backend
    python scripts/process-1995-flight-logs.py --dry-run        # OCR only, no DB writes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INGEST_DIR = Path("E:/Epstein-Pipeline/ingest/doj-flight-logs")
OUTPUT_DIR = Path("E:/Epstein-Pipeline/output/ocr/flight-logs-1995")
ENV_LOCAL = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")

# The 4 target documents: (doc_id, bates_number, description, expected_flights)
DOCUMENTS = [
    ("d-8934", "DOJ-OGR-00015834", "David Rodgers 1995 log (13 flights)", 13),
    ("d-8935", "DOJ-OGR-00015835", "David Hodge log Sep-Nov 1995 (41 flights)", 41),
    ("d-9026", "DOJ-OGR-00015953", "David Hodge log with passengers", None),
    ("d-13854", "DOJ-OGR-00015952", "David Rodgers 1995 with Trump flights", None),
]

# URL patterns to try -- DOJ has changed their URL scheme multiple times
URL_PATTERNS = [
    "https://www.justice.gov/d9/2025-02/{bates}.pdf",
    "https://www.justice.gov/d9/2026-01/{bates}.pdf",
    "https://www.justice.gov/d9/2025-01/{bates}.pdf",
    "https://www.justice.gov/d9/2024-12/{bates}.pdf",
    "https://www.justice.gov/epstein/files/{bates}.pdf",
    "https://www.justice.gov/media/{bates}/dl",
    "https://www.justice.gov/d9/2025-02/{bates_lower}.pdf",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_database_url() -> str:
    """Read DATABASE_URL from .env.local, stripping sslnegotiation param.

    psycopg doesn't support the Neon-specific sslnegotiation=direct param,
    so we strip it for compatibility.
    """
    if not ENV_LOCAL.exists():
        print(f"ERROR: {ENV_LOCAL} not found")
        sys.exit(1)

    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL=") and "UNPOOLED" not in line and "READONLY" not in line:
            url = line.split("=", 1)[1].strip()
            # psycopg doesn't support sslnegotiation param
            url = url.replace("&sslnegotiation=direct", "")
            return url

    print("ERROR: DATABASE_URL not found in .env.local")
    sys.exit(1)


def download_pdf(bates: str, dest: Path) -> str | None:
    """Try multiple URL patterns to download a DOJ PDF.

    Returns the successful URL string if downloaded, None if all patterns failed.
    Prints each attempt so user can manually locate the PDF.
    """
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Already exists: {dest.name} ({dest.stat().st_size:,} bytes)")
        return "(already cached)"

    tried_urls: list[str] = []

    for pattern in URL_PATTERNS:
        url = pattern.format(bates=bates, bates_lower=bates.lower())
        tried_urls.append(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 1000:
                content_type = resp.headers.get("Content-Type", "")
                # Accept PDF or octet-stream
                if "pdf" in content_type.lower() or "octet-stream" in content_type.lower() or resp.content[:5] == b"%PDF-":
                    dest.write_bytes(resp.content)
                    print(f"  Downloaded: {dest.name} ({len(resp.content):,} bytes) from {url}")
                    return url
                else:
                    print(f"  {url} -> 200 but not PDF (Content-Type: {content_type})")
            else:
                print(f"  {url} -> {resp.status_code}")
        except requests.RequestException as exc:
            print(f"  {url} -> error: {exc}")

        # Brief pause between attempts to be polite
        time.sleep(0.5)

    print(f"\n  FAILED: Could not download {bates}")
    print(f"  Tried {len(tried_urls)} URLs:")
    for u in tried_urls:
        print(f"    - {u}")
    print(f"  Manual fix: download the PDF and place it at {dest}")
    return None


def run_ocr(pdf_path: Path, backend: str = "auto") -> dict | None:
    """Run OCR on a single PDF using the pipeline's OcrProcessor.

    Returns a dict with keys: text, confidence, backend_used, page_texts,
    page_confidences, or None on failure.
    """
    try:
        from epstein_pipeline.config import Settings
        from epstein_pipeline.processors.ocr import OcrProcessor
    except ImportError:
        print("ERROR: epstein_pipeline not installed. Run: pip install -e .")
        print("  (from E:\\Epstein-Pipeline)")
        return None

    settings = Settings()
    processor = OcrProcessor(settings, backend=backend)

    print(f"  Running OCR (backend={backend})...")
    result = processor.process_file(pdf_path)

    if result.errors:
        for err in result.errors:
            print(f"  OCR ERROR: {err}")

    if result.warnings:
        for warn in result.warnings:
            print(f"  OCR WARNING: {warn}")

    if result.document and result.document.ocrText:
        text = result.document.ocrText

        # Split text back into pages using the double-newline separator
        # that all backends use to join pages
        page_texts = text.split("\n\n")

        return {
            "text": text,
            "confidence": result.ocr_confidence or 0.0,
            "backend_used": (
                result.warnings[0].split(":")[0]
                if result.warnings
                else backend
            ),
            "page_texts": page_texts,
            "page_confidences": [],
            "processing_time_ms": result.processing_time_ms,
        }
    else:
        print("  OCR produced no text")
        return None


def save_ocr_json(doc_id: str, bates: str, description: str, ocr_data: dict, output_dir: Path) -> Path:
    """Save OCR result as a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}.json"

    payload = {
        "id": doc_id,
        "bates": bates,
        "description": description,
        "text": ocr_data["text"],
        "confidence": ocr_data["confidence"],
        "backend_used": ocr_data["backend_used"],
        "page_count": len(ocr_data["page_texts"]),
        "char_count": len(ocr_data["text"]),
        "processing_time_ms": ocr_data["processing_time_ms"],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved JSON: {out_path}")
    return out_path


def insert_ocr_to_neon(
    doc_id: str,
    bates: str,
    ocr_data: dict,
    pdf_url: str | None,
    db_url: str,
    dry_run: bool = False,
) -> None:
    """Insert OCR text into Neon ocr_text table and update document pdfUrl.

    Uses per-page inserts so each page is independently searchable.
    """
    if dry_run:
        print(f"  [DRY RUN] Would insert {len(ocr_data['page_texts'])} pages into ocr_text")
        print(f"  [DRY RUN] Would update documents.pdfUrl for {doc_id}")
        return

    try:
        import psycopg
    except ImportError:
        print("ERROR: psycopg not installed. Run: pip install psycopg[binary]")
        return

    conn = psycopg.connect(db_url)
    try:
        cur = conn.cursor()

        # Insert per-page OCR text
        pages_inserted = 0
        for page_num, page_text in enumerate(ocr_data["page_texts"], start=1):
            if not page_text.strip():
                continue
            cur.execute(
                """
                INSERT INTO ocr_text ("docId", page, text)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (doc_id, page_num, page_text.strip()),
            )
            pages_inserted += cur.rowcount

        # Also insert full combined text as page 0 for whole-document search
        cur.execute(
            """
            INSERT INTO ocr_text ("docId", page, text)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (doc_id, 0, ocr_data["text"]),
        )
        pages_inserted += cur.rowcount

        print(f"  Inserted {pages_inserted} page(s) into ocr_text for {doc_id}")

        # Update document pdfUrl if we have one
        if pdf_url:
            cur.execute(
                """
                UPDATE documents SET "pdfUrl" = %s WHERE id = %s
                """,
                (pdf_url, doc_id),
            )
            if cur.rowcount:
                print(f"  Updated pdfUrl for {doc_id}")
            else:
                print(f"  WARNING: No document row found for {doc_id} (pdfUrl not updated)")

        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"  DB ERROR: {exc}")
    finally:
        conn.close()


def run_chunker_and_embeddings(doc_id: str, ocr_text: str) -> None:
    """Optionally chunk and embed the OCR text for vector search.

    This is best-effort -- will skip gracefully if dependencies are missing
    or GPU is unavailable.
    """
    try:
        from epstein_pipeline.config import Settings
        from epstein_pipeline.models.document import Document
        from epstein_pipeline.processors.chunker import Chunker
        from epstein_pipeline.processors.embeddings import EmbeddingProcessor
    except ImportError:
        print("  Skipping chunker/embeddings (pipeline not fully installed)")
        return

    settings = Settings()

    # Chunk the text
    chunker = Chunker(
        mode=settings.chunker_mode,
        target_tokens=settings.chunker_target_tokens,
        min_tokens=settings.chunker_min_tokens,
        max_tokens=settings.chunker_max_tokens,
    )
    chunks = chunker.chunk_document(doc_id, ocr_text)
    print(f"  Chunked into {len(chunks)} chunks")

    if not chunks:
        return

    # Generate embeddings
    try:
        doc = Document(
            id=doc_id,
            title=f"1995 Flight Log {doc_id}",
            source="travel",
            category="travel",
            ocrText=ocr_text,
        )
        embedder = EmbeddingProcessor(settings)
        embed_result = embedder.embed_document(doc)
        print(
            f"  Generated {len(embed_result.embeddings)} embeddings "
            f"({embed_result.processing_time_ms}ms, model={embed_result.model_name})"
        )
    except Exception as exc:
        print(f"  Embedding skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process 1995 DOJ flight log PDFs through OCR and export to Neon."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step; use PDFs already in the ingest directory.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "pymupdf", "surya", "olmocr", "docling", "smoldocling"],
        help="OCR backend (default: auto = pymupdf -> smoldocling -> surya -> docling).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run OCR and save JSON but do not write to Neon database.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the chunker + embeddings step.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("1995 Flight Log Processor")
    print("=" * 70)
    print(f"  Ingest dir:  {INGEST_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  OCR backend: {args.backend}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Documents:   {len(DOCUMENTS)}")
    print()

    # Create directories
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load DB URL (even if dry-run, validate it exists early)
    db_url = None
    if not args.dry_run:
        db_url = load_database_url()
        print(f"  DB: ...{db_url[-40:]}")
        print()

    # ── Step 1: Download PDFs ────────────────────────────────────────────
    print("STEP 1: Download PDFs")
    print("-" * 40)

    download_results: dict[str, Path | None] = {}
    download_urls: dict[str, str | None] = {}

    for doc_id, bates, desc, _ in DOCUMENTS:
        pdf_path = INGEST_DIR / f"{bates}.pdf"
        print(f"\n[{doc_id}] {desc}")
        print(f"  Bates: {bates}")

        if args.skip_download:
            if pdf_path.exists():
                print(f"  Found local: {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")
                download_results[doc_id] = pdf_path
                download_urls[doc_id] = None
            else:
                # Also check for doc_id-named files
                alt_path = INGEST_DIR / f"{doc_id}.pdf"
                if alt_path.exists():
                    print(f"  Found local: {alt_path.name} ({alt_path.stat().st_size:,} bytes)")
                    download_results[doc_id] = alt_path
                    download_urls[doc_id] = None
                else:
                    print(f"  NOT FOUND: Place PDF at {pdf_path}")
                    download_results[doc_id] = None
                    download_urls[doc_id] = None
        else:
            result_url = download_pdf(bates, pdf_path)
            if result_url is not None:
                download_results[doc_id] = pdf_path
                download_urls[doc_id] = result_url if result_url != "(already cached)" else None
            else:
                download_results[doc_id] = None
                download_urls[doc_id] = None

    available = {k: v for k, v in download_results.items() if v is not None}
    missing = {k: v for k, v in download_results.items() if v is None}

    print(f"\n  Available: {len(available)}/{len(DOCUMENTS)}")
    if missing:
        print(f"  Missing:   {', '.join(missing.keys())}")
        print(f"\n  To fix missing downloads, manually place PDFs in:\n    {INGEST_DIR}")
        print(f"  Expected filenames: {', '.join(bates + '.pdf' for _, bates, _, _ in DOCUMENTS)}")
    print()

    if not available:
        print("ERROR: No PDFs available. Nothing to process.")
        print("  Run with --skip-download after placing PDFs manually.")
        sys.exit(1)

    # ── Step 2: Run OCR ──────────────────────────────────────────────────
    print("STEP 2: Run OCR")
    print("-" * 40)

    ocr_results: dict[str, dict] = {}

    for doc_id, bates, desc, expected_flights in DOCUMENTS:
        pdf_path = download_results.get(doc_id)
        if pdf_path is None:
            print(f"\n[{doc_id}] SKIPPED (no PDF)")
            continue

        print(f"\n[{doc_id}] {desc}")
        print(f"  File: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

        ocr_data = run_ocr(pdf_path, backend=args.backend)
        if ocr_data:
            char_count = len(ocr_data["text"])
            page_count = len(ocr_data["page_texts"])
            confidence = ocr_data["confidence"]

            print(f"  Result: {char_count:,} chars, {page_count} pages, confidence={confidence:.3f}")

            # Quick content preview (first 200 chars)
            preview = ocr_data["text"][:200].replace("\n", " ").strip()
            print(f"  Preview: {preview}...")

            # Save JSON
            save_ocr_json(doc_id, bates, desc, ocr_data, OUTPUT_DIR)

            ocr_results[doc_id] = ocr_data
        else:
            print(f"  FAILED: No text extracted")

    print(f"\n  OCR completed: {len(ocr_results)}/{len(available)} documents")
    print()

    if not ocr_results:
        print("ERROR: OCR produced no results. Check backend availability.")
        sys.exit(1)

    # ── Step 3: Insert into Neon ─────────────────────────────────────────
    print("STEP 3: Insert OCR text into Neon")
    print("-" * 40)

    for doc_id, bates, desc, _ in DOCUMENTS:
        ocr_data = ocr_results.get(doc_id)
        if ocr_data is None:
            continue

        print(f"\n[{doc_id}] {desc}")

        # Use the download URL as pdfUrl if we have one
        pdf_url = download_urls.get(doc_id)

        insert_ocr_to_neon(
            doc_id=doc_id,
            bates=bates,
            ocr_data=ocr_data,
            pdf_url=pdf_url,
            db_url=db_url or "",
            dry_run=args.dry_run,
        )

    print()

    # ── Step 4: Chunker + Embeddings (optional) ─────────────────────────
    if not args.skip_embeddings:
        print("STEP 4: Chunker + Embeddings")
        print("-" * 40)

        for doc_id, bates, desc, _ in DOCUMENTS:
            ocr_data = ocr_results.get(doc_id)
            if ocr_data is None:
                continue

            print(f"\n[{doc_id}] {desc}")
            run_chunker_and_embeddings(doc_id, ocr_data["text"])

        print()
    else:
        print("STEP 4: Chunker + Embeddings [SKIPPED]")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Doc ID':<12} {'Bates':<22} {'Pages':>6} {'Chars':>10} {'Conf':>7} {'Status'}")
    print("-" * 70)

    total_chars = 0
    total_pages = 0

    for doc_id, bates, desc, expected_flights in DOCUMENTS:
        ocr_data = ocr_results.get(doc_id)
        if ocr_data:
            pages = len(ocr_data["page_texts"])
            chars = len(ocr_data["text"])
            conf = ocr_data["confidence"]
            total_chars += chars
            total_pages += pages
            status = "OK"
            if conf < 0.5:
                status = "LOW CONF"
            print(f"{doc_id:<12} {bates:<22} {pages:>6} {chars:>10,} {conf:>7.3f} {status}")
        else:
            pdf_exists = download_results.get(doc_id) is not None
            status = "OCR FAIL" if pdf_exists else "NO PDF"
            print(f"{doc_id:<12} {bates:<22} {'--':>6} {'--':>10} {'--':>7} {status}")

    print("-" * 70)
    print(f"{'TOTAL':<12} {'':<22} {total_pages:>6} {total_chars:>10,}")
    print()

    if missing:
        print("NEXT STEPS:")
        print(f"  1. Manually download missing PDFs to {INGEST_DIR}")
        print(f"  2. Re-run with: python scripts/process-1995-flight-logs.py --skip-download")
    elif len(ocr_results) == len(DOCUMENTS):
        print("All 4 flight logs processed successfully.")
    else:
        print("Some documents had OCR failures. Check backend availability.")
        print("  Try: python scripts/process-1995-flight-logs.py --backend surya")


if __name__ == "__main__":
    main()
