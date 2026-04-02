"""Bulk ingest DS10 PDFs into Neon: documents table + ocr_text table.

Scans all 503K DS10 PDFs, extracts text with PyMuPDF, and batch-inserts
document records + OCR text into the production Neon database.

Usage:
    python scripts/ingest-ds10-to-neon.py --workers 8
    python scripts/ingest-ds10-to-neon.py --workers 8 --limit 1000
    python scripts/ingest-ds10-to-neon.py --dry-run --limit 100

Prerequisites:
    pip install pymupdf psycopg2-binary python-dotenv
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────
INPUT_DIR = Path(r"E:\epstein-ds10\extracted\VOL00010\IMAGES")
SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ENV_FILE = SITE_DIR / ".env.local"
COMPLETED_LOG = Path(r"E:\epstein-ds10\neon-ingested.txt")

BATCH_SIZE = 200  # Rows per INSERT batch
DOC_SOURCE = "doj-ds10"
DOC_CATEGORY = "other"


def get_db_url() -> str:
    """Read DATABASE_URL from the site's .env.local."""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"').strip("'")
            return url
    raise RuntimeError(f"DATABASE_URL not found in {ENV_FILE}")


def extract_text_from_pdf(pdf_path_str: str) -> tuple[str, str, int, str]:
    """Extract text + title from a single PDF. Returns (efta_id, text, page_count, title)."""
    import fitz

    pdf_path = Path(pdf_path_str)
    efta_id = pdf_path.stem  # e.g. EFTA01262782

    try:
        doc = fitz.open(pdf_path_str)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()

        full_text = "\n\n---PAGE BREAK---\n\n".join(pages) if pages else ""

        # Derive title from first non-empty line of text
        title = efta_id
        for line in full_text.split("\n"):
            clean = line.strip()
            if clean and len(clean) > 3 and len(clean) < 200:
                title = clean[:200]
                break

        return (efta_id, full_text, len(pages) or 1, title)

    except Exception as e:
        return (efta_id, "", 1, efta_id)


def make_doc_id(efta_id: str) -> str:
    """Convert EFTA01262782 -> efta-efta01262782 (matching existing DS10 format)."""
    return f"efta-{efta_id.lower()}"


def batch_insert_docs(conn, batch: list[tuple]):
    """Insert a batch of documents. Skip conflicts (already exists)."""
    if not batch:
        return 0

    cur = conn.cursor()
    inserted = 0

    # Use execute_values for efficient batch insert
    from psycopg2.extras import execute_values

    # documents: (id, title, summary, date, source, category, "pageCount", "pdfUrl")
    query = """
        INSERT INTO documents (id, title, summary, date, source, category, "pageCount", "pdfUrl")
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    values = [
        (
            make_doc_id(efta_id),
            title[:500],  # truncate long titles
            f"DOJ EFTA Data Set 10 document {efta_id}",
            None,
            DOC_SOURCE,
            DOC_CATEGORY,
            page_count,
            f"https://efts.fbi.gov/file-repository/{efta_id}.pdf",
        )
        for efta_id, _, page_count, title in batch
    ]

    execute_values(cur, query, values, page_size=BATCH_SIZE)
    inserted = cur.rowcount
    conn.commit()
    cur.close()
    return inserted


def batch_insert_ocr(conn, batch: list[tuple]):
    """Insert a batch of OCR text records. Skip conflicts."""
    if not batch:
        return 0

    cur = conn.cursor()

    from psycopg2.extras import execute_values

    # ocr_text: ("docId", text)  — note camelCase column name
    query = """
        INSERT INTO ocr_text ("docId", text)
        VALUES %s
        ON CONFLICT ("docId") DO NOTHING
    """
    values = [
        (make_doc_id(efta_id), text[:500000])  # cap at 500K chars
        for efta_id, text, _, _ in batch
        if text.strip()  # only insert if there's actual text
    ]

    if values:
        execute_values(cur, query, values, page_size=BATCH_SIZE)

    conn.commit()
    cur.close()
    return len(values)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest DS10 PDFs into Neon")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()), help="PDF text extraction workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit PDFs to process")
    parser.add_argument("--dry-run", action="store_true", help="Extract text but don't insert into DB")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already-ingested docs")
    args = parser.parse_args()

    print("DS10 → Neon Ingestion Pipeline", flush=True)
    print(f"  Workers: {args.workers}", flush=True)

    # Connect to Neon
    db_url = get_db_url()
    print(f"  DB: ...{db_url[-30:]}", flush=True)

    if not args.dry_run:
        import psycopg2
        conn = psycopg2.connect(db_url)
        # Check current count
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents WHERE source = %s", (DOC_SOURCE,))
        existing_count = cur.fetchone()[0]
        cur.close()
        print(f"  Existing DS10 docs in DB: {existing_count:,}", flush=True)

    # Scan for PDFs
    print("\nScanning for PDFs...", flush=True)
    pdfs = []
    for subdir_entry in sorted(os.scandir(INPUT_DIR), key=lambda e: e.name):
        if not subdir_entry.is_dir():
            continue
        for entry in os.scandir(subdir_entry.path):
            if entry.name.endswith(".pdf"):
                pdfs.append(entry.path)
    print(f"Found {len(pdfs):,} PDFs", flush=True)

    # Skip already-ingested
    completed = set()
    if args.skip_existing and COMPLETED_LOG.exists():
        completed = set(COMPLETED_LOG.read_text().splitlines())
        before = len(pdfs)
        pdfs = [p for p in pdfs if Path(p).stem not in completed]
        if before - len(pdfs) > 0:
            print(f"Skipping {before - len(pdfs):,} already ingested", flush=True)

    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print("Nothing to ingest!", flush=True)
        return

    print(f"\nExtracting text from {len(pdfs):,} PDFs with {args.workers} workers...", flush=True)

    total_docs = 0
    total_ocr = 0
    total_chars = 0
    start = time.time()
    batch = []
    newly_completed = []
    chunksize = max(1, min(200, len(pdfs) // (args.workers * 2)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(extract_text_from_pdf, pdfs, chunksize=chunksize)):
            efta_id, text, page_count, title = result
            batch.append(result)
            newly_completed.append(efta_id)

            # Process batch
            if len(batch) >= BATCH_SIZE:
                if not args.dry_run:
                    docs_inserted = batch_insert_docs(conn, batch)
                    ocr_inserted = batch_insert_ocr(conn, batch)
                    total_docs += docs_inserted
                    total_ocr += ocr_inserted
                else:
                    total_docs += len(batch)
                    total_ocr += sum(1 for _, t, _, _ in batch if t.strip())

                total_chars += sum(len(t) for _, t, _, _ in batch)
                batch = []

            # Progress
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(pdfs) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i+1:,}/{len(pdfs):,} | {total_docs:,} docs, {total_ocr:,} OCR | "
                    f"{total_chars/1e6:.0f}M chars | {rate:.0f}/s | ~{remaining/60:.0f}m left",
                    flush=True,
                )

                # Flush completed log
                if newly_completed:
                    with open(COMPLETED_LOG, "a") as f:
                        f.write("\n".join(newly_completed) + "\n")
                    newly_completed = []

    # Final batch
    if batch:
        if not args.dry_run:
            total_docs += batch_insert_docs(conn, batch)
            total_ocr += batch_insert_ocr(conn, batch)
        else:
            total_docs += len(batch)
            total_ocr += sum(1 for _, t, _, _ in batch if t.strip())
        total_chars += sum(len(t) for _, t, _, _ in batch)

    # Final completed log flush
    if newly_completed:
        with open(COMPLETED_LOG, "a") as f:
            f.write("\n".join(newly_completed) + "\n")

    if not args.dry_run:
        conn.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Documents inserted: {total_docs:,}", flush=True)
    print(f"  OCR records inserted: {total_ocr:,}", flush=True)
    print(f"  Total text: {total_chars/1e6:.1f}M characters", flush=True)
    print(f"  Rate: {len(pdfs)/elapsed:.1f} PDFs/s", flush=True)


if __name__ == "__main__":
    main()
