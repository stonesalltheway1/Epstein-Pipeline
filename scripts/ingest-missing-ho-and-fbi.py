"""Ingest missing House Oversight docs (5) from HuggingFace + FBI Vault FOIA (22 parts)."""

import csv
import sys
import json
import hashlib
import re
import os
import time
import httpx
from pathlib import Path

# Bump CSV field limit for large OCR text
csv.field_size_limit(sys.maxsize)

_raw_url = os.environ.get("EPSTEIN_NEON_DATABASE_URL", "")
if not _raw_url:
    raise RuntimeError("EPSTEIN_NEON_DATABASE_URL environment variable is required")
# psycopg doesn't support sslnegotiation param — strip it
DB_URL = re.sub(r'[&?]sslnegotiation=[^&]*', '', _raw_url)

# ---------------------------------------------------------------------------
# Part 1: Ingest 5 missing House Oversight docs from HuggingFace
# ---------------------------------------------------------------------------

def ingest_missing_house_oversight():
    """Extract the 5 missing HO docs from the HuggingFace dataset and insert into Neon."""
    import psycopg

    HF_FILE = Path.home() / ".cache/huggingface/hub/datasets--teyler--epstein-files-20k/snapshots/1e669c107a8351eed3f28e99e727249d40b393ea/EPS_FILES_20K_NOV2025.txt"
    MISSING_IDS = {16552, 16694, 16695, 16696, 16697}

    if not HF_FILE.exists():
        print("HuggingFace dataset not found. Skipping HO ingest.")
        return

    print(f"\n=== Ingesting {len(MISSING_IDS)} missing House Oversight docs ===")

    # Extract text for missing docs
    docs = {}
    with open(HF_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            fn = row[0]
            m = re.search(r'HOUSE_OVERSIGHT_(\d+)', fn)
            if m and int(m.group(1)) in MISSING_IDS:
                doc_id = int(m.group(1))
                text = row[1] if len(row) > 1 else ""
                # Accumulate text (multi-line docs)
                if doc_id in docs:
                    docs[doc_id] += "\n" + text
                else:
                    docs[doc_id] = text

    print(f"Found text for {len(docs)} docs: {sorted(docs.keys())}")

    with psycopg.connect(DB_URL) as conn:
        for doc_num, text in docs.items():
            doc_id = f"kaggle-ho-{doc_num:06d}"
            title = f"HOUSE_OVERSIGHT_{doc_num:06d}"
            summary = text[:500].strip() if text else ""

            conn.execute(
                """INSERT INTO documents (id, title, summary, source, category)
                   VALUES (%s, %s, %s, 'house-oversight', 'other')
                   ON CONFLICT (id) DO NOTHING""",
                (doc_id, title, summary),
            )

            # tsv is a generated column — no manual update needed
            print(f"  Inserted {doc_id}: {len(text)} chars")

        conn.commit()

    print(f"Done: {len(docs)} House Oversight docs inserted.")


# ---------------------------------------------------------------------------
# Part 2: Download FBI Vault FOIA (22 parts)
# ---------------------------------------------------------------------------

def download_fbi_vault():
    """Download FBI Vault FOIA releases with browser-like headers."""
    out_dir = Path(__file__).parent.parent / "ingest" / "fbi-vault"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading FBI Vault FOIA (22 parts) ===")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://vault.fbi.gov/jeffrey-epstein",
    }

    downloaded = []
    for part in range(1, 23):
        part_str = f"{part:02d}"
        outfile = out_dir / f"fbi-epstein-part-{part_str}.pdf"

        if outfile.exists() and outfile.stat().st_size > 10000:
            print(f"  Part {part_str}: already downloaded ({outfile.stat().st_size:,} bytes)")
            downloaded.append(outfile)
            continue

        # Try multiple URL patterns
        urls = [
            f"https://vault.fbi.gov/jeffrey-epstein/Jeffrey%20Epstein%20Part%20{part_str}%20of%2022/at_download/file",
            f"https://vault.fbi.gov/jeffrey-epstein/jeffrey-epstein-part-{part_str}-of-22/at_download/file",
            f"https://vault.fbi.gov/jeffrey-epstein/Jeffrey Epstein Part {part_str} of 22/at_download/file",
        ]

        success = False
        for url in urls:
            try:
                with httpx.Client(follow_redirects=True, timeout=60) as client:
                    resp = client.get(url, headers=headers)
                    if resp.status_code == 200 and len(resp.content) > 10000:
                        outfile.write_bytes(resp.content)
                        print(f"  Part {part_str}: downloaded {len(resp.content):,} bytes")
                        downloaded.append(outfile)
                        success = True
                        break
                    else:
                        print(f"  Part {part_str}: {resp.status_code} ({len(resp.content)} bytes) from {url[:80]}")
            except Exception as e:
                print(f"  Part {part_str}: error from {url[:60]}: {e}")

        if not success:
            print(f"  Part {part_str}: FAILED all URLs")

        time.sleep(1)  # Be polite

    print(f"\nDownloaded {len(downloaded)} of 22 FBI Vault parts")
    return downloaded


# ---------------------------------------------------------------------------
# Part 3: OCR + ingest FBI Vault PDFs
# ---------------------------------------------------------------------------

def ingest_fbi_vault_pdfs(pdf_files: list[Path]):
    """OCR and ingest FBI Vault PDFs into Neon."""
    if not pdf_files:
        print("No FBI Vault PDFs to ingest.")
        return

    try:
        import pymupdf
    except ImportError:
        print("pymupdf not installed. Run: pip install pymupdf")
        return

    import psycopg

    print(f"\n=== OCR + Ingest {len(pdf_files)} FBI Vault PDFs ===")

    with psycopg.connect(DB_URL) as conn:
        for pdf_path in pdf_files:
            part_num = re.search(r'part-(\d+)', pdf_path.name)
            if not part_num:
                continue
            part = int(part_num.group(1))
            doc_id = f"fbi-vault-{part:02d}"

            # Check if already in DB
            r = conn.execute("SELECT id FROM documents WHERE id = %s", (doc_id,)).fetchone()
            if r:
                print(f"  {doc_id}: already in DB, skipping")
                continue

            # Extract text with PyMuPDF
            try:
                doc = pymupdf.open(str(pdf_path))
                pages = []
                for page in doc:
                    pages.append(page.get_text())
                full_text = "\n\n".join(pages)
                page_count = len(doc)
                doc.close()
            except Exception as e:
                print(f"  {doc_id}: OCR failed: {e}")
                continue

            title = f"FBI FOIA - Jeffrey Epstein Part {part:02d} of 22"
            summary = full_text[:1000].strip() if full_text else ""

            conn.execute(
                """INSERT INTO documents (id, title, summary, source, category, "pageCount", "pdfUrl")
                   VALUES (%s, %s, %s, 'fbi', 'fbi-foia', %s, %s)
                   ON CONFLICT (id) DO UPDATE SET summary = EXCLUDED.summary, "pageCount" = EXCLUDED."pageCount"
                """,
                (doc_id, title, summary, page_count,
                 f"https://vault.fbi.gov/jeffrey-epstein/Jeffrey%20Epstein%20Part%20{part:02d}%20of%2022/view"),
            )

            # Update tsvector
            conn.execute(
                """UPDATE documents SET tsv = to_tsvector('english', coalesce(title,'') || ' ' || coalesce(summary,''))
                   WHERE id = %s""",
                (doc_id,),
            )

            print(f"  {doc_id}: {page_count} pages, {len(full_text):,} chars")

        conn.commit()

    print("FBI Vault ingest complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Epstein Pipeline - Ingest Missing Docs")
    print("=" * 60)

    # Step 1: House Oversight missing docs
    ingest_missing_house_oversight()

    # Step 2: Download FBI Vault
    fbi_pdfs = download_fbi_vault()

    # Step 3: OCR + ingest FBI Vault
    ingest_fbi_vault_pdfs(fbi_pdfs)

    print("\n=== All done ===")
