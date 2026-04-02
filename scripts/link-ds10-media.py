"""Link DS10 media items to persons via document_persons table.

Cross-references EFTA IDs in ds10-media.json with person links in Neon,
updates personIds in the media JSON, and re-exports.

Also enriches titles with document titles from the DB where available.

Usage:
    python scripts/link-ds10-media.py
    python scripts/link-ds10-media.py --dry-run
    python scripts/link-ds10-media.py --output data/ds10-media-linked.json

Prerequisites:
    pip install psycopg2-binary python-dotenv
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ENV_FILE = SITE_DIR / ".env.local"
DS10_JSON = SITE_DIR / "data" / "ds10-media.json"


def get_db_url() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"DATABASE_URL not found in {ENV_FILE}")


def extract_efta_id(item: dict) -> str | None:
    """Extract EFTA ID from media item (checks id, thumbnailUrl, fullUrl).

    Handles both:
      id: "ds10-EFTA01262782-0" -> EFTA01262782
      thumbnailUrl: ".../ds10/EFTA0126/EFTA01262782_p0_i0.png" -> EFTA01262782
    """
    # Try from ID first
    match = re.match(r"ds10-(EFTA\d+)", item.get("id", ""))
    if match:
        return match.group(1)
    # Try from URL (thumbnailUrl or fullUrl)
    for key in ("thumbnailUrl", "fullUrl"):
        url = item.get(key, "")
        match = re.search(r"(EFTA\d{8,})", url)
        if match:
            return match.group(1)
    return None


def make_doc_id(efta_id: str) -> str:
    """Convert EFTA01262782 -> sd-10-EFTA01262782 (DS10 doc format)."""
    return f"sd-10-{efta_id}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Link DS10 media to persons")
    parser.add_argument("--input", type=str, default=str(DS10_JSON), help="Input ds10-media.json")
    parser.add_argument("--output", type=str, default=None, help="Output JSON (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    parser.add_argument("--min-relevance", type=float, default=0.0, help="Min relevance score for person links")
    args = parser.parse_args()

    import psycopg2

    print("DS10 Media Person Linker", flush=True)

    # Load media JSON
    input_path = Path(args.input)
    media = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(media):,} media items from {input_path.name}", flush=True)

    # Collect all EFTA IDs
    efta_ids = set()
    item_efta_map = {}  # item id -> efta_id
    for item in media:
        efta_id = extract_efta_id(item)
        if efta_id:
            efta_ids.add(efta_id)
            item_efta_map[item["id"]] = efta_id
    print(f"  {len(efta_ids):,} unique EFTA IDs", flush=True)

    # Connect to Neon
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Fetch all person links for these EFTA IDs
    print("Fetching person links from Neon...", flush=True)
    doc_ids = [make_doc_id(e) for e in efta_ids]

    # Batch query in chunks of 1000
    person_links = {}  # doc_id -> [person_id, ...]
    for i in range(0, len(doc_ids), 1000):
        chunk = doc_ids[i:i+1000]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(f"""
            SELECT doc_id, person_id, relevance_score
            FROM document_persons
            WHERE doc_id IN ({placeholders})
            AND (relevance_score IS NULL OR relevance_score >= %s)
            ORDER BY doc_id, relevance_score DESC
        """, chunk + [args.min_relevance])

        for doc_id, person_id, score in cur.fetchall():
            if doc_id not in person_links:
                person_links[doc_id] = []
            person_links[doc_id].append(person_id)

    linked_docs = len(person_links)
    total_links = sum(len(v) for v in person_links.values())
    print(f"  {linked_docs:,} documents with person links ({total_links:,} total links)", flush=True)

    # Fetch document titles for enrichment
    print("Fetching document titles...", flush=True)
    doc_titles = {}  # doc_id -> title
    for i in range(0, len(doc_ids), 1000):
        chunk = doc_ids[i:i+1000]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(f"""
            SELECT id, title FROM documents
            WHERE id IN ({placeholders})
        """, chunk)

        for doc_id, title in cur.fetchall():
            if title and len(title) > 5:
                doc_titles[doc_id] = title

    print(f"  {len(doc_titles):,} document titles fetched", flush=True)

    # Build person name lookup for title enrichment
    person_ids_needed = set()
    for pids in person_links.values():
        person_ids_needed.update(pids)

    person_names = {}
    if person_ids_needed:
        pid_list = list(person_ids_needed)
        for i in range(0, len(pid_list), 500):
            chunk = pid_list[i:i+500]
            placeholders = ",".join(["%s"] * len(chunk))
            cur.execute(f"SELECT id, name FROM persons WHERE id IN ({placeholders})", chunk)
            for pid, name in cur.fetchall():
                person_names[pid] = name

    conn.close()

    # Update media items
    updated_count = 0
    title_updated = 0

    for item in media:
        efta_id = item_efta_map.get(item["id"])
        if not efta_id:
            continue

        doc_id = make_doc_id(efta_id)

        # Update personIds
        if doc_id in person_links:
            item["personIds"] = person_links[doc_id]
            updated_count += 1

        # Enrich title with document title if available
        if doc_id in doc_titles:
            doc_title = doc_titles[doc_id]
            # Only replace generic titles
            if "Evidence Photo" in item.get("title", "") or "Page" in item.get("title", ""):
                # Extract page number from current title
                page_match = re.search(r"Page (\d+)", item["title"])
                page_str = f" (p.{page_match.group(1)})" if page_match else ""
                item["title"] = f"{doc_title}{page_str}"
                title_updated += 1

    print(f"\nResults:", flush=True)
    print(f"  Media items with person links: {updated_count:,}", flush=True)
    print(f"  Titles enriched from DB: {title_updated:,}", flush=True)

    # Person frequency
    all_pids = []
    for item in media:
        all_pids.extend(item.get("personIds", []))

    if all_pids:
        from collections import Counter
        top_persons = Counter(all_pids).most_common(20)
        print(f"\nTop referenced persons:", flush=True)
        for pid, count in top_persons:
            name = person_names.get(pid, pid)
            print(f"  {name}: {count:,} media items", flush=True)

    # Write output
    if not args.dry_run:
        output_path = Path(args.output) if args.output else input_path
        output_path.write_text(
            json.dumps(media, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWritten to {output_path}", flush=True)
    else:
        print("\n[DRY RUN] No changes written", flush=True)


if __name__ == "__main__":
    main()
