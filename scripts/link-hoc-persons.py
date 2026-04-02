"""Link HOC documents to known persons via substring matching on OCR text.

Scans all kaggle-ho-* OCR text against the 1,570+ persons registry.
Uses Aho-Corasick-style matching (multi-pattern substring search) for speed.

For each match, inserts into document_persons and updates documents.personIds.
"""

import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_PATH = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")
BATCH_SIZE = 100
MIN_NAME_LEN = 6  # Skip very short names to avoid false positives


def get_db_url() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            raw = line.split("=", 1)[1].strip().strip('"')
            return re.sub(r"[&?]sslnegotiation=[^&]*", "", raw)
    raise RuntimeError("DATABASE_URL not found")


def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def main():
    import psycopg

    db_url = get_db_url()
    conn = psycopg.connect(db_url)

    # Step 1: Load all persons (id, name, aliases)
    print("Loading persons registry...")
    persons_rows = conn.execute(
        """SELECT id, name, COALESCE(aliases, '[]'::jsonb) as aliases FROM persons"""
    ).fetchall()

    # Build search patterns: lowercase name -> person_id
    patterns = {}  # lowercase_name -> person_id
    for pid, name, aliases in persons_rows:
        if len(name) >= MIN_NAME_LEN:
            patterns[name.lower()] = pid
        if aliases:
            import json
            alias_list = aliases if isinstance(aliases, list) else json.loads(aliases)
            for alias in alias_list:
                if isinstance(alias, str) and len(alias) >= MIN_NAME_LEN:
                    patterns[alias.lower()] = pid

    print(f"  {len(persons_rows)} persons, {len(patterns)} search patterns")

    # Step 2: Load HOC docs that need linking (no existing person links)
    print("\nLoading HOC documents needing person links...")
    docs = conn.execute(
        """SELECT o."docId", substring(o.text, 1, 50000) as text
           FROM ocr_text o
           WHERE o."docId" LIKE 'kaggle-ho-%'
           AND NOT EXISTS (
               SELECT 1 FROM document_persons dp WHERE dp.doc_id = o."docId"
           )
           ORDER BY o."docId"
        """
    ).fetchall()
    print(f"  {len(docs)} docs to process")

    if not docs:
        print("All HOC docs already have person links!")
        return

    # Step 3: Match persons in OCR text
    print(f"\nScanning {len(docs)} docs against {len(patterns)} name patterns...")
    t0 = time.time()
    total_links = 0
    docs_linked = 0
    errors = 0

    # Sort patterns by length descending to match longer names first
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)

    for i, (doc_id, text) in enumerate(docs):
        if not text:
            continue

        text_lower = text.lower()
        found_persons = set()

        for name_lower, person_id in sorted_patterns:
            if name_lower in text_lower:
                # Word boundary check: ensure we're matching whole words
                # Skip if it's a substring of a longer word
                idx = text_lower.find(name_lower)
                if idx > 0 and text_lower[idx - 1].isalpha():
                    continue
                end = idx + len(name_lower)
                if end < len(text_lower) and text_lower[end].isalpha():
                    continue
                found_persons.add(person_id)

        if found_persons:
            docs_linked += 1
            for pid in found_persons:
                try:
                    conn.execute(
                        """INSERT INTO document_persons (doc_id, person_id)
                           VALUES (%s, %s) ON CONFLICT DO NOTHING""",
                        (doc_id, pid),
                    )
                    conn.execute(
                        """UPDATE documents SET "personIds" = "personIds" || %s::jsonb
                           WHERE id = %s AND NOT "personIds" @> %s::jsonb""",
                        (f'"{pid}"', doc_id, f'"{pid}"'),
                    )
                    total_links += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error on {doc_id}/{pid}: {e}")
                    conn.rollback()
                    conn = psycopg.connect(db_url)

        if (i + 1) % 1000 == 0:
            conn.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(docs) - i - 1) / rate
            pct = (i + 1) / len(docs) * 100
            print(
                f"  {i+1:>6}/{len(docs)} ({pct:.0f}%) "
                f"docs_linked={docs_linked} total_links={total_links} "
                f"errors={errors} ~{fmt_time(remaining)}"
            )

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {fmt_time(elapsed)}:")
    print(f"  Docs scanned:  {len(docs):,}")
    print(f"  Docs linked:   {docs_linked:,}")
    print(f"  Total links:   {total_links:,}")
    print(f"  Errors:        {errors}")
    print(f"  Rate:          {len(docs) / elapsed:.0f} docs/sec")


if __name__ == "__main__":
    main()
