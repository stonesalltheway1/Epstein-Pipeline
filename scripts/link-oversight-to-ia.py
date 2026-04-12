"""Link house-oversight Neon records to Archive.org-hosted page images.

The Oversight Committee's Epstein estate documents are mirrored on Archive.org
as JPEG page images. Our 38K `house-oversight` records have bates numbers like
HOUSE_OVERSIGHT_018767 but no linked source image. This script:

1. Fetches IA metadata for the two known Oversight items
2. Builds a {bates_number: url} map from the IA file listing
3. Updates our Neon `documents` records with imageUrl/sourceUrl pointing at IA

No files are downloaded — the JPGs stay hosted on IA's servers. We just add
URL pointers so the viewer can embed them.

Covers 22,933 of our records (bates 10477-33600).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("oversight-link")


IA_ITEMS = [
    "oversight-committee-additional-epstein-files",
    "Epstein_Estate_Documents_-_Seventh_Production",
]


def get_neon_url() -> str:
    url = os.environ.get("EPSTEIN_NEON_DATABASE_URL")
    if url:
        return url
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("EPSTEIN_NEON_DATABASE_URL="):
                u = line.split("=", 1)[1].strip().strip('"').strip("'")
                if "sslnegotiation" in u:
                    u = re.sub(r"[&?]sslnegotiation=[^&]*", "", u)
                return u
    raise RuntimeError("EPSTEIN_NEON_DATABASE_URL not set")


def build_ia_url_map() -> dict[str, str]:
    """Returns {bates_number: public_ia_url} for every page image in our target IA items."""
    mapping: dict[str, str] = {}

    for item_id in IA_ITEMS:
        logger.info("Fetching IA metadata for %s...", item_id)
        r = requests.get(f"https://archive.org/metadata/{item_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        files = data.get("files", [])
        img_count = 0
        for f in files:
            name = f.get("name", "")
            if not name.startswith("IMAGES/"):
                continue
            # Match HOUSE_OVERSIGHT_NNNNNN
            m = re.search(r"HOUSE_OVERSIGHT_(\d+)\.(jpeg|jpg)$", name, re.IGNORECASE)
            if not m:
                continue
            bates = m.group(1)  # preserve leading zeros
            # Canonical URL — URL-encode the path
            url = f"https://archive.org/download/{item_id}/{quote(name, safe='/')}"
            # Prefer the first IA item for a given bates (both overlap in 10477-33600)
            if bates not in mapping:
                mapping[bates] = url
                img_count += 1
        logger.info("  %s: added %d new bates → URL mappings", item_id, img_count)

    logger.info("Total unique bates numbers with IA URLs: %d", len(mapping))
    return mapping


def update_neon(mapping: dict[str, str], dry_run: bool = False) -> dict:
    conn = psycopg2.connect(get_neon_url())
    stats = {"matched": 0, "updated": 0, "not_matched": 0}

    cur = conn.cursor()
    # Pull all records with HOUSE_OVERSIGHT bates
    cur.execute("""
        SELECT id, "batesRange", "sourceUrl", "pdfUrl"
        FROM documents
        WHERE source = 'house-oversight'
          AND "batesRange" IS NOT NULL
    """)
    rows = cur.fetchall()
    logger.info("Scanning %d house-oversight records...", len(rows))

    updates: list[tuple[str, str]] = []
    for doc_id, bates, source_url, pdf_url in rows:
        m = re.search(r"HOUSE_OVERSIGHT_(\d+)", bates or "")
        if not m:
            stats["not_matched"] += 1
            continue
        bates_num = m.group(1)
        # Zero-pad to match IA's 6-digit format
        bates_num_padded = bates_num.zfill(6)
        ia_url = mapping.get(bates_num_padded) or mapping.get(bates_num)
        if not ia_url:
            stats["not_matched"] += 1
            continue
        stats["matched"] += 1
        # Only update if the pdfUrl is currently empty (don't clobber existing)
        if not pdf_url:
            updates.append((doc_id, ia_url))

    logger.info("Matched: %d, not matched: %d", stats["matched"], stats["not_matched"])
    logger.info("Updates to apply: %d", len(updates))

    if dry_run:
        logger.info("DRY RUN - sample updates:")
        for doc_id, url in updates[:5]:
            logger.info("  %s → %s", doc_id, url)
        cur.close()
        conn.close()
        return stats

    # Batch-apply updates
    if updates:
        from psycopg2.extras import execute_batch
        execute_batch(
            cur,
            """UPDATE documents SET "pdfUrl" = %s WHERE id = %s""",
            [(url, doc_id) for doc_id, url in updates],
            page_size=500,
        )
        conn.commit()
        stats["updated"] = len(updates)
        logger.info("Committed %d pdfUrl updates", stats["updated"])

    cur.close()
    conn.close()
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = build_ia_url_map()
    stats = update_neon(mapping, dry_run=args.dry_run)
    logger.info("=== Final stats: %s", stats)


if __name__ == "__main__":
    main()
