"""Backfill OCR text for House Oversight documents from HuggingFace text dataset.

Reads the EPS_FILES_20K_NOV2025.txt CSV (2.1M rows) and inserts OCR text
for any kaggle-ho-* document that's missing from the ocr_text table.
"""

import csv
import sys
import re
import os
import time
from pathlib import Path
from collections import defaultdict

# Bump CSV field limit for large OCR text
csv.field_size_limit(sys.maxsize)

_raw_url = os.environ.get(
    "EPSTEIN_NEON_DATABASE_URL",
    ""  # Will be loaded from .env.local
)

# Load from .env.local if not set
if not _raw_url:
    env_path = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                _raw_url = line.split("=", 1)[1].strip().strip('"')
                break

DB_URL = re.sub(r'[&?]sslnegotiation=[^&]*', '', _raw_url)

HF_FILE = Path.home() / ".cache/huggingface/hub/datasets--teyler--epstein-files-20k/snapshots/1e669c107a8351eed3f28e99e727249d40b393ea/EPS_FILES_20K_NOV2025.txt"


def main():
    import psycopg

    if not HF_FILE.exists():
        print(f"HuggingFace file not found: {HF_FILE}")
        sys.exit(1)

    print(f"Reading HuggingFace text file: {HF_FILE}")
    print(f"File size: {HF_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 1: Get existing ocr_text doc IDs for HOC docs
    print("\nChecking existing OCR coverage...")
    with psycopg.connect(DB_URL) as conn:
        existing_ocr = set()
        rows = conn.execute(
            """SELECT "docId" FROM ocr_text WHERE "docId" LIKE 'kaggle-ho-%'"""
        ).fetchall()
        existing_ocr = {r[0] for r in rows}
        print(f"  Existing OCR entries for kaggle-ho-*: {len(existing_ocr)}")

        # Get all kaggle-ho-* doc IDs
        all_ho = conn.execute(
            "SELECT id FROM documents WHERE id LIKE 'kaggle-ho-%'"
        ).fetchall()
        all_ho_ids = {r[0] for r in all_ho}
        print(f"  Total kaggle-ho-* documents: {len(all_ho_ids)}")

        missing = all_ho_ids - existing_ocr
        print(f"  Missing OCR: {len(missing)}")

    if not missing:
        print("All kaggle-ho-* docs already have OCR text!")
        return

    # Step 2: Parse HF CSV and collect text for missing docs
    print(f"\nParsing HF CSV for {len(missing)} missing docs...")
    doc_texts = defaultdict(list)
    row_count = 0
    matched = 0

    with open(HF_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        print(f"  Header: {header}")

        for row in reader:
            row_count += 1
            if row_count % 500000 == 0:
                print(f"  Processed {row_count:,} rows, matched {matched} docs...")

            if not row or len(row) < 2:
                continue

            fn = row[0]
            m = re.search(r'HOUSE_OVERSIGHT_(\d+)', fn)
            if not m:
                continue

            doc_num = int(m.group(1))
            doc_id = f"kaggle-ho-{doc_num:06d}"

            if doc_id in missing:
                text = row[1] if len(row) > 1 else ""
                doc_texts[doc_id].append(text)
                if len(doc_texts[doc_id]) == 1:
                    matched += 1

    print(f"\nParsed {row_count:,} rows total")
    print(f"Matched text for {len(doc_texts)} missing docs")

    # Step 3: Insert into ocr_text
    print(f"\nInserting OCR text into database...")
    inserted = 0
    errors = 0

    with psycopg.connect(DB_URL) as conn:
        for doc_id, text_parts in doc_texts.items():
            full_text = "\n".join(text_parts).strip()
            if not full_text:
                continue

            try:
                conn.execute(
                    """INSERT INTO ocr_text ("docId", text)
                       VALUES (%s, %s)
                       ON CONFLICT ("docId") DO NOTHING""",
                    (doc_id, full_text),
                )
                inserted += 1

                if inserted % 1000 == 0:
                    conn.commit()
                    print(f"  Inserted {inserted}...")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error on {doc_id}: {e}")

        conn.commit()

    print(f"\nDone!")
    print(f"  Inserted: {inserted}")
    print(f"  Errors: {errors}")
    print(f"  Still missing: {len(missing) - len(doc_texts)} (no text in HF file)")


if __name__ == "__main__":
    main()
