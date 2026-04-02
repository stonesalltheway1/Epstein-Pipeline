"""GLiNER zero-shot NER pilot on zero-person documents.

Runs GLiNER on a batch of documents that have OCR text but zero person links,
extracts PERSON entities, matches against the person registry, and reports findings.

Usage:
    python scripts/gliner-pilot.py --batch-size 1000
    python scripts/gliner-pilot.py --batch-size 100 --dry-run
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ENV_FILE = SITE_DIR / ".env.local"


def get_db_url() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"').strip("'")
            # Remove sslnegotiation param that psycopg2 doesn't support
            url = url.replace("&sslnegotiation=direct", "").replace("?sslnegotiation=direct", "?")
            return url
    raise RuntimeError(f"DATABASE_URL not found in {ENV_FILE}")


def load_gliner_model():
    """Load GLiNER medium-v2.1 model."""
    print("Loading GLiNER model (gliner_medium-v2.1)...", flush=True)
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    print("GLiNER model loaded.", flush=True)
    return model


def extract_persons_gliner(model, text: str, threshold: float = 0.4) -> list[dict]:
    """Extract person entities from text using GLiNER."""
    # GLiNER has a token limit, chunk if needed
    MAX_CHARS = 4000
    all_entities = []

    chunks = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]

    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        try:
            entities = model.predict_entities(
                chunk,
                ["person"],
                threshold=threshold
            )
            for ent in entities:
                name = ent["text"].strip()
                if len(name) >= 3 and " " in name:  # At least first + last name
                    all_entities.append({
                        "name": name,
                        "score": round(ent["score"], 3),
                        "label": ent["label"]
                    })
        except Exception as e:
            continue

    return all_entities


def build_person_lookup(persons: list[dict]) -> dict:
    """Build name -> person_id lookup from DB persons."""
    lookup = {}
    for p in persons:
        name = p["name"].strip().lower()
        lookup[name] = p["id"]
        # Also add without middle names/initials
        parts = name.split()
        if len(parts) >= 3:
            lookup[f"{parts[0]} {parts[-1]}"] = p["id"]
        # Add aliases
        aliases = p.get("aliases") or []
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except:
                aliases = []
        for alias in aliases:
            if alias:
                lookup[alias.strip().lower()] = p["id"]
    return lookup


def match_entity_to_person(entity_name: str, lookup: dict) -> str | None:
    """Try to match an extracted entity name to a known person."""
    name_lower = entity_name.strip().lower()

    # Exact match
    if name_lower in lookup:
        return lookup[name_lower]

    # Without middle initial/name
    parts = name_lower.split()
    if len(parts) >= 3:
        short = f"{parts[0]} {parts[-1]}"
        if short in lookup:
            return lookup[short]

    # Try reversed (last, first)
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        if reversed_name in lookup:
            return lookup[reversed_name]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    import psycopg2
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Load persons for matching
    print("Loading person registry...", flush=True)
    cur.execute('SELECT id, name, aliases FROM persons')
    persons = [{"id": r[0], "name": r[1], "aliases": r[2]} for r in cur.fetchall()]
    print(f"  {len(persons)} persons in registry", flush=True)
    lookup = build_person_lookup(persons)
    print(f"  {len(lookup)} name variants in lookup", flush=True)

    # Load GLiNER
    model = load_gliner_model()

    # Fetch zero-person documents with OCR text
    print(f"\nFetching {args.batch_size} zero-person docs (offset {args.offset})...", flush=True)
    cur.execute("""
        SELECT d.id, LEFT(o.text, 8000) as text
        FROM documents d
        JOIN ocr_text o ON d.id = o."docId"
        WHERE (d."personIds" = '[]'::jsonb OR d."personIds" IS NULL)
        AND LENGTH(o.text) > 100
        ORDER BY d.id
        OFFSET %s LIMIT %s
    """, (args.offset, args.batch_size))
    docs = cur.fetchall()
    print(f"  Fetched {len(docs)} documents", flush=True)

    # Process
    total_entities = 0
    matched_entities = 0
    new_discoveries = Counter()
    matched_persons = Counter()
    doc_links_to_insert = []

    start = time.time()
    for i, (doc_id, text) in enumerate(docs):
        entities = extract_persons_gliner(model, text, threshold=args.threshold)
        total_entities += len(entities)

        for ent in entities:
            person_id = match_entity_to_person(ent["name"], lookup)
            if person_id:
                matched_entities += 1
                matched_persons[person_id] += 1
                doc_links_to_insert.append((doc_id, person_id))
            else:
                new_discoveries[ent["name"]] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1}/{len(docs)} docs ({rate:.1f} docs/sec) — {total_entities} entities, {matched_entities} matched", flush=True)

    elapsed = time.time() - start

    # Report
    print(f"\n{'='*60}")
    print(f"GLiNER Pilot Results ({len(docs)} documents)")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s ({len(docs)/elapsed:.1f} docs/sec)")
    print(f"  Total person entities extracted: {total_entities}")
    print(f"  Matched to known persons: {matched_entities}")
    print(f"  Unique known persons found: {len(matched_persons)}")
    print(f"  Unique new/unknown names: {len(new_discoveries)}")
    print(f"  Doc-person links to create: {len(doc_links_to_insert)}")

    print(f"\n  Top 20 matched persons:")
    for pid, count in matched_persons.most_common(20):
        name = next((p["name"] for p in persons if p["id"] == pid), pid)
        print(f"    {name}: {count} mentions")

    print(f"\n  Top 30 NEW person names (not in registry):")
    for name, count in new_discoveries.most_common(30):
        print(f"    {name}: {count} mentions")

    # Insert if not dry-run
    if not args.dry_run and doc_links_to_insert:
        print(f"\n  Inserting {len(doc_links_to_insert)} doc-person links...", flush=True)
        # Reconnect in case connection timed out during long extraction
        cur.close()
        conn.close()
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        inserted = 0
        for i, (doc_id, person_id) in enumerate(doc_links_to_insert):
            try:
                cur.execute(
                    "INSERT INTO document_persons (doc_id, person_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (doc_id, person_id)
                )
                cur.execute(
                    """UPDATE documents SET "personIds" = "personIds" || %s::jsonb
                       WHERE id = %s AND NOT "personIds" @> %s::jsonb""",
                    (json.dumps(person_id), doc_id, json.dumps(person_id))
                )
                inserted += 1
                if (i + 1) % 100 == 0:
                    conn.commit()
                    print(f"    Committed {i+1}/{len(doc_links_to_insert)}...", flush=True)
            except Exception as e:
                conn.rollback()
        conn.commit()
        print(f"  Inserted {inserted} links.", flush=True)
    elif args.dry_run:
        print(f"\n  DRY RUN — no links inserted.", flush=True)

    cur.close()
    conn.close()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
