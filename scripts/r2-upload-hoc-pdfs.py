"""Upload assembled HOC PDFs to Cloudflare R2 and update pdfUrl in Neon.

Uploads to: epstein-exposed-media/hoc-pdfs/{doc_id}.pdf
Sets pdfUrl to: https://media.epsteinexposed.com/hoc-pdfs/{doc_id}.pdf

Uses concurrent uploads (10 workers) with retry logic.
"""

import os
import re
import sys
import time
import hashlib
import asyncio
import aiohttp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Fix Windows asyncio event loop for aiodns
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PDF_DIR = Path("E:/Epstein-Pipeline/output/hoc-pdfs")
ENV_PATH = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")

CF_ACCOUNT_ID = None
CF_API_TOKEN = None
BUCKET = "epstein-exposed-media"
R2_BASE_URL = "https://media.epsteinexposed.com"
R2_PREFIX = "hoc-pdfs"

CONCURRENCY = 10  # parallel uploads
MAX_RETRIES = 3


def load_env():
    global CF_ACCOUNT_ID, CF_API_TOKEN
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("R2_ACCOUNT_ID="):
            CF_ACCOUNT_ID = line.split("=", 1)[1].strip()
        elif line.startswith("R2_API_TOKEN="):
            CF_API_TOKEN = line.split("=", 1)[1].strip()


def get_db_url() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            raw = line.split("=", 1)[1].strip().strip('"')
            return re.sub(r"[&?]sslnegotiation=[^&]*", "", raw)
    raise RuntimeError("DATABASE_URL not found")


def fmt_size(b: int) -> str:
    if b > 1_000_000_000:
        return f"{b / 1_000_000_000:.1f} GB"
    if b > 1_000_000:
        return f"{b / 1_000_000:.1f} MB"
    return f"{b / 1_000:.1f} KB"


def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

async def upload_one(session: aiohttp.ClientSession, pdf_path: Path, sem: asyncio.Semaphore) -> tuple[str, bool]:
    """Upload a single PDF to R2. Returns (doc_id, success)."""
    doc_id = pdf_path.stem  # e.g. "kaggle-ho-010477"
    key = f"{R2_PREFIX}/{pdf_path.name}"
    api_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/r2/buckets/{BUCKET}/objects/{key}"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                data = pdf_path.read_bytes()
                async with session.put(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {CF_API_TOKEN}",
                        "Content-Type": "application/pdf",
                        "Cache-Control": "public, max-age=31536000, immutable",
                    },
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status in (200, 201):
                        return (doc_id, True)
                    else:
                        text = await resp.text()
                        if attempt == MAX_RETRIES - 1:
                            print(f"  FAIL {doc_id}: {resp.status} {text[:100]}")
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  FAIL {doc_id}: {e}")
                await asyncio.sleep(2 ** attempt)

    return (doc_id, False)


async def upload_all(pdf_files: list[Path]) -> list[str]:
    """Upload all PDFs concurrently. Returns list of successfully uploaded doc_ids."""
    sem = asyncio.Semaphore(CONCURRENCY)
    uploaded = []
    failed = 0
    total = len(pdf_files)
    total_bytes = sum(f.stat().st_size for f in pdf_files)

    print(f"\nUploading {total:,} PDFs ({fmt_size(total_bytes)}) to R2")
    print(f"  Bucket: {BUCKET}/{R2_PREFIX}/")
    print(f"  Concurrency: {CONCURRENCY}")

    t0 = time.time()
    bytes_uploaded = 0

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches for progress reporting
        batch_size = 50
        for batch_start in range(0, total, batch_size):
            batch = pdf_files[batch_start:batch_start + batch_size]
            tasks = [upload_one(session, f, sem) for f in batch]
            results = await asyncio.gather(*tasks)

            for doc_id, success in results:
                if success:
                    uploaded.append(doc_id)
                    bytes_uploaded += pdf_files[batch_start + results.index((doc_id, success))].stat().st_size if success else 0
                else:
                    failed += 1

            elapsed = time.time() - t0
            done = len(uploaded) + failed
            rate = done / elapsed if elapsed > 0 else 1
            remaining = (total - done) / rate if rate > 0 else 0
            pct = done / total * 100
            print(f"  {done:>5} / {total} ({pct:.0f}%)  ok={len(uploaded)} fail={failed}  ~{fmt_time(remaining)} remaining")

    elapsed = time.time() - t0
    print(f"\n  Upload done in {fmt_time(elapsed)}: {len(uploaded)} uploaded, {failed} failed")
    return uploaded


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

def update_pdf_urls(doc_ids: list[str]):
    """Update pdfUrl on documents for all uploaded PDFs."""
    import psycopg

    db_url = get_db_url()
    print(f"\nUpdating pdfUrl for {len(doc_ids):,} documents...")

    t0 = time.time()
    updated = 0

    with psycopg.connect(db_url) as conn:
        for i, doc_id in enumerate(doc_ids):
            pdf_url = f"{R2_BASE_URL}/{R2_PREFIX}/{doc_id}.pdf"
            conn.execute(
                """UPDATE documents SET "pdfUrl" = %s WHERE id = %s AND ("pdfUrl" IS NULL OR "pdfUrl" = '')""",
                (pdf_url, doc_id),
            )
            updated += 1

            if (i + 1) % 500 == 0:
                conn.commit()
                print(f"  {i+1}/{len(doc_ids)} updated")

        conn.commit()

    elapsed = time.time() - t0
    print(f"  Done in {int(elapsed)}s: {updated} pdfUrl fields set")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERROR: Missing R2_ACCOUNT_ID or R2_API_TOKEN")
        sys.exit(1)

    # Get list of PDFs
    pdf_files = sorted(PDF_DIR.glob("kaggle-ho-*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files):,} PDFs in {PDF_DIR}")

    # Upload to R2
    uploaded_ids = asyncio.run(upload_all(pdf_files))

    # Update DB
    if uploaded_ids:
        update_pdf_urls(uploaded_ids)

    print(f"\nAll done! {len(uploaded_ids)} HOC documents now have PDF links on R2.")


if __name__ == "__main__":
    main()
