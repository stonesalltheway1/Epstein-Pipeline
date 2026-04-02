"""NER-based person linking for DS10 documents.

Loads all person names from Neon, scans OCR text for matches,
and inserts into document_persons table with relevance scores.

Uses multi-strategy matching:
1. Exact full-name match (highest confidence)
2. Last-name + first-initial match (medium confidence)
3. Last-name-only match with context validation (lower confidence)

Usage:
    python scripts/ner-ds10-persons.py --batch-size 500
    python scripts/ner-ds10-persons.py --limit 10000 --dry-run
    python scripts/ner-ds10-persons.py --min-confidence 0.5

Prerequisites:
    pip install psycopg2-binary python-dotenv
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ENV_FILE = SITE_DIR / ".env.local"

# Names too common to match on last-name alone
COMMON_LAST_NAMES = {
    "smith", "johnson", "williams", "brown", "jones", "davis", "miller",
    "wilson", "moore", "taylor", "anderson", "thomas", "jackson", "white",
    "harris", "martin", "thompson", "garcia", "martinez", "robinson",
    "clark", "rodriguez", "lewis", "lee", "walker", "hall", "allen",
    "young", "king", "wright", "scott", "green", "baker", "adams",
    "nelson", "hill", "campbell", "mitchell", "roberts", "carter",
    "phillips", "evans", "turner", "torres", "parker", "collins",
    "edwards", "stewart", "flores", "morris", "murphy", "cook",
    "ross", "rogers", "morgan", "bell", "fisher", "black", "ford",
}

# Skip matching against these ambiguous / single-word names
SKIP_NAMES = {"epstein", "maxwell", "doe", "unknown"}


def get_db_url() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"DATABASE_URL not found in {ENV_FILE}")


def build_person_matchers(persons: list[dict]) -> dict:
    """Build lookup structures for fast name matching.

    Returns dict with:
        full_name -> {person_id, confidence: 0.95}
        last_first_init -> [{person_id, full_name, confidence: 0.7}]
        last_name -> [{person_id, full_name, confidence: 0.3}]
    """
    full_names = {}       # "jeffrey epstein" -> person_id
    last_first = {}       # "epstein j" -> [person_id, ...]
    last_names = {}       # "epstein" -> [person_id, ...]
    name_patterns = []    # compiled regex patterns for each person

    for p in persons:
        pid = p["id"]
        name = p["name"].strip()
        if not name or len(name) < 3:
            continue

        name_lower = name.lower()
        parts = name_lower.split()
        if len(parts) < 2:
            continue

        # Skip ambiguous names
        if name_lower in SKIP_NAMES:
            continue

        # Full name match
        full_names[name_lower] = pid

        # Also match "Last, First" format
        last = parts[-1]
        first = parts[0]
        full_names[f"{last}, {first}"] = pid
        full_names[f"{last} {first}"] = pid  # reversed order

        # Last + first initial
        fi = first[0] if first else ""
        key = f"{last} {fi}"
        if key not in last_first:
            last_first[key] = []
        last_first[key].append({"person_id": pid, "name": name})

        # Last name only (skip if too common)
        if last not in COMMON_LAST_NAMES and len(last) > 2:
            if last not in last_names:
                last_names[last] = []
            last_names[last].append({"person_id": pid, "name": name})

        # Regex pattern for the full name (word-boundary aware)
        escaped = re.escape(name)
        # Allow flexible whitespace/punctuation between name parts
        pattern = r"\b" + re.sub(r"\s+", r"[\\s,.-]+", escaped) + r"\b"
        name_patterns.append((re.compile(pattern, re.IGNORECASE), pid, name))

    return {
        "full_names": full_names,
        "last_first": last_first,
        "last_names": last_names,
        "patterns": name_patterns,
    }


def find_persons_in_text(text: str, matchers: dict, min_confidence: float = 0.3) -> list[dict]:
    """Find person mentions in text. Returns list of {person_id, confidence, context}."""
    if not text or len(text) < 10:
        return []

    text_lower = text.lower()
    found = {}  # person_id -> {confidence, context}

    # Strategy 1: Regex pattern matching (best — handles formatting variations)
    for pattern, pid, name in matchers["patterns"]:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            confidence = 0.9

            if pid not in found or found[pid]["confidence"] < confidence:
                found[pid] = {"confidence": confidence, "context": context}

    # Strategy 2: Full name exact match (high confidence)
    for full_name, pid in matchers["full_names"].items():
        if full_name in text_lower:
            idx = text_lower.index(full_name)
            start = max(0, idx - 50)
            end = min(len(text), idx + len(full_name) + 50)
            context = text[start:end].strip()
            confidence = 0.95

            if pid not in found or found[pid]["confidence"] < confidence:
                found[pid] = {"confidence": confidence, "context": context}

    # Strategy 3: Last name only (lower confidence, needs context validation)
    for last_name, person_list in matchers["last_names"].items():
        if f" {last_name} " in f" {text_lower} " or f" {last_name}," in f" {text_lower},":
            # Validate: is it likely a person reference? Check surrounding context
            idx = text_lower.find(last_name)
            if idx < 0:
                continue

            # Get surrounding words
            start = max(0, idx - 80)
            end = min(len(text), idx + len(last_name) + 80)
            context = text[start:end].strip()
            ctx_lower = context.lower()

            # Boost confidence if context suggests person reference
            confidence = 0.3
            person_indicators = ["mr.", "ms.", "mrs.", "dr.", "prof.", "sir", "lord",
                                 "attorney", "counsel", "witness", "defendant",
                                 "deposition", "testified", "said", "according to"]
            if any(ind in ctx_lower for ind in person_indicators):
                confidence = 0.5

            # Only use first match for each last name (avoid noise)
            if len(person_list) == 1:
                pid = person_list[0]["person_id"]
                if pid not in found or found[pid]["confidence"] < confidence:
                    found[pid] = {"confidence": confidence, "context": context}

    # Filter by minimum confidence
    results = []
    for pid, data in found.items():
        if data["confidence"] >= min_confidence:
            results.append({
                "person_id": pid,
                "confidence": data["confidence"],
                "context": data["context"][:300],  # truncate context
            })

    return sorted(results, key=lambda x: -x["confidence"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NER person linking for DS10")
    parser.add_argument("--batch-size", type=int, default=500, help="OCR docs to process per batch")
    parser.add_argument("--limit", type=int, default=None, help="Max docs to process")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum confidence for person match")
    parser.add_argument("--dry-run", action="store_true", help="Find matches but don't insert")
    parser.add_argument("--source-filter", default="doj-ds10", help="Only process docs from this source")
    args = parser.parse_args()

    import psycopg2
    from psycopg2.extras import execute_values

    print("DS10 NER Person Linking Pipeline", flush=True)

    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Load all persons
    print("Loading persons from Neon...", flush=True)
    cur.execute("SELECT id, name, slug FROM persons")
    persons = [{"id": r[0], "name": r[1], "slug": r[2]} for r in cur.fetchall()]
    print(f"  {len(persons):,} persons loaded", flush=True)

    # Build matchers
    matchers = build_person_matchers(persons)
    print(f"  {len(matchers['full_names']):,} full name patterns", flush=True)
    print(f"  {len(matchers['last_names']):,} last name patterns", flush=True)
    print(f"  {len(matchers['patterns']):,} regex patterns", flush=True)

    # Count DS10 docs with OCR
    cur.execute("""
        SELECT COUNT(*) FROM ocr_text o
        JOIN documents d ON o."docId" = d.id
        WHERE d.source = %s
    """, (args.source_filter,))
    total_ocr = cur.fetchone()[0]
    print(f"\n{total_ocr:,} DS10 documents with OCR text", flush=True)

    # Count existing links
    cur.execute("""
        SELECT COUNT(*) FROM document_persons dp
        JOIN documents d ON dp.doc_id = d.id
        WHERE d.source = %s
    """, (args.source_filter,))
    existing_links = cur.fetchone()[0]
    print(f"{existing_links:,} existing person links", flush=True)

    # Process in batches using cursor-based pagination
    offset = 0
    total_processed = 0
    total_matches = 0
    total_inserted = 0
    start = time.time()

    limit_clause = f"LIMIT {args.limit}" if args.limit else ""

    while True:
        cur.execute(f"""
            SELECT o."docId", o.text FROM ocr_text o
            JOIN documents d ON o."docId" = d.id
            WHERE d.source = %s
            ORDER BY o."docId"
            OFFSET %s LIMIT %s
        """, (args.source_filter, offset, args.batch_size))

        rows = cur.fetchall()
        if not rows:
            break

        # Find person matches for each document
        batch_links = []
        for doc_id, text in rows:
            matches = find_persons_in_text(text, matchers, args.min_confidence)
            total_matches += len(matches)

            for m in matches:
                batch_links.append((
                    doc_id,
                    m["person_id"],
                    m["context"],
                    m["confidence"],
                ))

        # Batch insert into document_persons
        if batch_links and not args.dry_run:
            execute_values(
                cur,
                """
                INSERT INTO document_persons (doc_id, person_id, mention_context, relevance_score)
                VALUES %s
                ON CONFLICT (doc_id, person_id) DO UPDATE
                SET relevance_score = GREATEST(document_persons.relevance_score, EXCLUDED.relevance_score),
                    mention_context = COALESCE(EXCLUDED.mention_context, document_persons.mention_context)
                """,
                batch_links,
                page_size=500,
            )
            conn.commit()
            total_inserted += len(batch_links)

        total_processed += len(rows)
        offset += args.batch_size

        elapsed = time.time() - start
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(
            f"  {total_processed:,}/{total_ocr:,} docs | "
            f"{total_matches:,} matches ({total_inserted:,} inserted) | "
            f"{rate:.0f} docs/s",
            flush=True,
        )

        if args.limit and total_processed >= args.limit:
            break

    conn.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Processed: {total_processed:,} documents", flush=True)
    print(f"  Matches found: {total_matches:,}", flush=True)
    print(f"  Links inserted: {total_inserted:,}", flush=True)
    avg_per_doc = total_matches / total_processed if total_processed > 0 else 0
    print(f"  Avg matches/doc: {avg_per_doc:.1f}", flush=True)


if __name__ == "__main__":
    main()
