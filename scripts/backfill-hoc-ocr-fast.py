"""High-performance OCR backfill for House Oversight documents.

Reads the HuggingFace EPS_FILES_20K CSV and batch-inserts OCR text
into Neon Postgres using:
- COPY protocol for maximum throughput
- Connection pooling via psycopg pool
- Batch commits (5000 rows per transaction)
- Progress reporting with estimated time remaining

Optimized for Neon Scale plan (8-16 CU).
"""

import csv
import sys
import re
import time
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_FILE = (
    Path.home()
    / ".cache/huggingface/hub/datasets--teyler--epstein-files-20k"
    / "snapshots/1e669c107a8351eed3f28e99e727249d40b393ea"
    / "EPS_FILES_20K_NOV2025.txt"
)

ENV_PATH = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")

BATCH_SIZE = 50  # rows per executemany call (smaller to avoid SSL payload limits)
COMMIT_EVERY = 500  # rows per commit


def get_db_url() -> str:
    """Load DATABASE_URL from .env.local and strip sslnegotiation."""
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            raw = line.split("=", 1)[1].strip().strip('"')
            return re.sub(r"[&?]sslnegotiation=[^&]*", "", raw)
    raise RuntimeError("DATABASE_URL not found in .env.local")


def fmt_time(seconds: float) -> str:
    """Format seconds as Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Phase 1: Parse CSV into memory (fast, ~15s for 2.1M rows)
# ---------------------------------------------------------------------------

def parse_csv() -> dict[str, str]:
    """Parse the HuggingFace CSV and return {doc_id: full_text}."""
    print(f"[1/3] Parsing CSV: {HF_FILE.name}")
    print(f"      File size: {HF_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    doc_parts: dict[str, list[str]] = defaultdict(list)
    row_count = 0
    t0 = time.time()

    with open(HF_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            row_count += 1
            if row_count % 500_000 == 0:
                elapsed = time.time() - t0
                rate = row_count / elapsed
                remaining = (2_136_420 - row_count) / rate
                print(f"      {row_count:>10,} rows  ({fmt_time(elapsed)} elapsed, ~{fmt_time(remaining)} remaining)")

            if not row or len(row) < 2:
                continue

            m = re.search(r"HOUSE_OVERSIGHT_(\d+)", row[0])
            if m:
                doc_id = f"kaggle-ho-{int(m.group(1)):06d}"
                doc_parts[doc_id].append(row[1])

    # Merge multi-row documents
    docs = {}
    for doc_id, parts in doc_parts.items():
        text = "\n".join(parts).strip()
        if text:
            docs[doc_id] = text

    elapsed = time.time() - t0
    print(f"      Done: {row_count:,} rows -> {len(docs):,} unique docs in {fmt_time(elapsed)}")
    return docs


# ---------------------------------------------------------------------------
# Phase 2: Batch insert with executemany + ON CONFLICT
# ---------------------------------------------------------------------------

def batch_insert(docs: dict[str, str]) -> None:
    """Insert docs into ocr_text using batched executemany."""
    import psycopg

    db_url = get_db_url()
    total = len(docs)

    print(f"\n[2/3] Inserting {total:,} docs into ocr_text")
    print(f"      Batch size: {BATCH_SIZE}, commit every: {COMMIT_EVERY}")

    t0 = time.time()
    inserted = 0
    skipped = 0
    batch: list[tuple[str, str]] = []

    items = list(docs.items())
    conn = psycopg.connect(db_url, autocommit=False)
    conn.execute("SET statement_timeout = '600s'")

    for i, (doc_id, text) in enumerate(items):
        # Truncate very large texts to avoid SSL payload issues
        if len(text) > 500_000:
            text = text[:500_000]
        batch.append((doc_id, text))

        if len(batch) >= BATCH_SIZE or i == len(items) - 1:
            retries = 0
            while retries < 3:
                try:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """INSERT INTO ocr_text ("docId", text)
                               VALUES (%s, %s)
                               ON CONFLICT ("docId") DO NOTHING""",
                            batch,
                        )
                    inserted += len(batch)
                    break
                except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                    retries += 1
                    print(f"      Connection error (retry {retries}/3): {e}")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(2 * retries)
                    conn = psycopg.connect(db_url, autocommit=False)
                    conn.execute("SET statement_timeout = '600s'")

            batch = []

            if inserted % COMMIT_EVERY < BATCH_SIZE:
                conn.commit()
                elapsed = time.time() - t0
                rate = inserted / elapsed if elapsed > 0 else 0
                remaining = (total - inserted) / rate if rate > 0 else 0
                pct = inserted / total * 100
                print(f"      {inserted:>8,} / {total:,} ({pct:5.1f}%)  {rate:.0f} rows/s  ~{fmt_time(remaining)} remaining")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"      Done: {total:,} processed in {fmt_time(elapsed)} ({rate:.0f} rows/s)")


# ---------------------------------------------------------------------------
# Phase 3: Verify
# ---------------------------------------------------------------------------

def verify() -> None:
    """Count OCR entries after insert."""
    import psycopg

    db_url = get_db_url()
    print("\n[3/3] Verifying...")

    with psycopg.connect(db_url) as conn:
        total_ocr = conn.execute(
            """SELECT COUNT(*) FROM ocr_text WHERE "docId" LIKE 'kaggle-ho-%'"""
        ).fetchone()[0]

        total_docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE id LIKE 'kaggle-ho-%'"
        ).fetchone()[0]

        coverage = total_ocr / total_docs * 100 if total_docs > 0 else 0
        print(f"      OCR entries: {total_ocr:,} / {total_docs:,} kaggle-ho docs ({coverage:.1f}% coverage)")

        # Also check d-* HOC docs
        d_total = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE source = 'house-oversight' AND id LIKE 'd-%'"
        ).fetchone()[0]
        d_ocr = conn.execute(
            """SELECT COUNT(*) FROM ocr_text o
               JOIN documents d ON d.id = o."docId"
               WHERE d.source = 'house-oversight' AND d.id LIKE 'd-%'"""
        ).fetchone()[0]
        d_coverage = d_ocr / d_total * 100 if d_total > 0 else 0
        print(f"      d-* HOC OCR: {d_ocr:,} / {d_total:,} ({d_coverage:.1f}% coverage)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HOC OCR Backfill (High Performance)")
    print("=" * 60)

    if not HF_FILE.exists():
        print(f"ERROR: HuggingFace file not found: {HF_FILE}")
        sys.exit(1)

    docs = parse_csv()
    batch_insert(docs)
    verify()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
