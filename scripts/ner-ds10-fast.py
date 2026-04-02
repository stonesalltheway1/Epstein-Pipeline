"""Fast NER person linking for DS10 using Aho-Corasick.

~100x faster than regex approach. Builds a single automaton from all person
names and scans each document once.

Usage:
    python scripts/ner-ds10-fast.py
    python scripts/ner-ds10-fast.py --dry-run
    python scripts/ner-ds10-fast.py --batch-size 2000
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import ahocorasick

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ENV_FILE = SITE_DIR / ".env.local"

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

SKIP_NAMES = {"epstein", "maxwell", "doe", "unknown"}


def get_db_url() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"').strip("'")
            url = url.replace("&sslnegotiation=direct", "").replace("?sslnegotiation=direct", "?")
            return url
    raise RuntimeError(f"DATABASE_URL not found in {ENV_FILE}")


def build_automaton(persons: list[dict]) -> tuple[ahocorasick.Automaton, dict]:
    """Build Aho-Corasick automaton from all person name variants.

    Returns (automaton, pattern_meta) where pattern_meta maps
    pattern_string -> list of (person_id, confidence, variant_type).
    """
    A = ahocorasick.Automaton()
    pattern_meta: dict[str, list[tuple[str, float, str]]] = {}

    for p in persons:
        pid = p["id"]
        name = p["name"].strip()
        if not name or len(name) < 3:
            continue

        name_lower = name.lower()
        parts = name_lower.split()
        if len(parts) < 2:
            continue

        if name_lower in SKIP_NAMES:
            continue

        first = parts[0]
        last = parts[-1]

        # Full name: "jeffrey epstein" (confidence 0.95)
        variants = [
            (name_lower, 0.95, "full"),
            (f"{last}, {first}", 0.95, "full_reversed"),
            (f"{last} {first}", 0.85, "reversed"),
        ]

        # Last + first initial: "epstein j" (confidence 0.7)
        fi = first[0] if first else ""
        if fi:
            variants.append((f"{last} {fi}", 0.7, "last_init"))
            variants.append((f"{last}, {fi}", 0.7, "last_init_comma"))

        # Last name only for uncommon names (confidence 0.35)
        if last not in COMMON_LAST_NAMES and len(last) > 3:
            variants.append((last, 0.35, "last_only"))

        for pattern, confidence, vtype in variants:
            if len(pattern) < 3:
                continue
            key = pattern
            if key not in pattern_meta:
                pattern_meta[key] = []
                A.add_word(key, key)
            pattern_meta[key].append((pid, confidence, vtype))

    A.make_automaton()
    return A, pattern_meta


def is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if match at [start:end] has word boundaries."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def find_persons_aho(text: str, automaton: ahocorasick.Automaton,
                     meta: dict, min_confidence: float = 0.3) -> list[dict]:
    """Find person mentions using Aho-Corasick. Single pass through text."""
    if not text or len(text) < 10:
        return []

    text_lower = text.lower()
    found: dict[str, dict] = {}  # person_id -> best match

    for end_idx, pattern in automaton.iter(text_lower):
        start_idx = end_idx - len(pattern) + 1

        if not is_word_boundary(text_lower, start_idx, end_idx + 1):
            continue

        entries = meta.get(pattern, [])
        for pid, confidence, vtype in entries:
            # For last-name-only, boost if person indicators nearby
            if vtype == "last_only":
                ctx_start = max(0, start_idx - 80)
                ctx_end = min(len(text_lower), end_idx + 81)
                ctx = text_lower[ctx_start:ctx_end]
                indicators = ("mr.", "ms.", "mrs.", "dr.", "prof.", "sir",
                              "attorney", "counsel", "witness", "defendant",
                              "deposition", "testified", "said")
                if any(ind in ctx for ind in indicators):
                    confidence = 0.5
                # else keep at 0.35

            if confidence < min_confidence:
                continue

            # Extract context snippet
            ctx_start = max(0, start_idx - 50)
            ctx_end = min(len(text), end_idx + 51)
            context = text[ctx_start:ctx_end].strip()

            if pid not in found or found[pid]["confidence"] < confidence:
                found[pid] = {"confidence": confidence, "context": context[:300]}

    return [
        {"person_id": pid, "confidence": d["confidence"], "context": d["context"]}
        for pid, d in found.items()
    ]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast NER person linking (Aho-Corasick)")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-filter", default="doj-ds10")
    args = parser.parse_args()

    import psycopg2
    from psycopg2.extras import execute_values

    print("DS10 NER Person Linking (Aho-Corasick)", flush=True)

    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Load persons
    print("Loading persons...", flush=True)
    cur.execute("SELECT id, name, slug FROM persons")
    persons = [{"id": r[0], "name": r[1], "slug": r[2]} for r in cur.fetchall()]
    print(f"  {len(persons):,} persons", flush=True)

    # Build automaton
    t0 = time.time()
    automaton, meta = build_automaton(persons)
    print(f"  Automaton: {len(meta):,} patterns built in {time.time()-t0:.2f}s", flush=True)

    # Count DS10 docs
    cur.execute("""
        SELECT COUNT(*) FROM ocr_text o
        JOIN documents d ON o."docId" = d.id
        WHERE d.source = %s
    """, (args.source_filter,))
    total_ocr = cur.fetchone()[0]
    print(f"\n{total_ocr:,} DS10 documents with OCR text", flush=True)

    # Check existing
    cur.execute("""
        SELECT COUNT(*) FROM document_persons dp
        JOIN documents d ON dp.doc_id = d.id
        WHERE d.source = %s
    """, (args.source_filter,))
    existing = cur.fetchone()[0]
    print(f"{existing:,} existing person links", flush=True)

    # Process
    offset = 0
    total_processed = 0
    total_matches = 0
    total_inserted = 0
    start = time.time()

    while True:
        cur.execute("""
            SELECT o."docId", o.text FROM ocr_text o
            JOIN documents d ON o."docId" = d.id
            WHERE d.source = %s
            ORDER BY o."docId"
            OFFSET %s LIMIT %s
        """, (args.source_filter, offset, args.batch_size))

        rows = cur.fetchall()
        if not rows:
            break

        batch_links = []
        for doc_id, text in rows:
            matches = find_persons_aho(text, automaton, meta, args.min_confidence)
            total_matches += len(matches)

            for m in matches:
                batch_links.append((
                    doc_id,
                    m["person_id"],
                    m["context"],
                    m["confidence"],
                ))

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
                page_size=1000,
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
    print(f"  Matches: {total_matches:,}", flush=True)
    print(f"  Inserted: {total_inserted:,}", flush=True)
    if total_processed > 0:
        print(f"  Avg matches/doc: {total_matches/total_processed:.1f}", flush=True)


if __name__ == "__main__":
    main()
