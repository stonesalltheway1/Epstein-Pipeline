"""Backfill pdfUrl for documents where we can construct a DOJ EFTA URL.

Many efta-ds9 and doj-efta documents have NULL pdfUrl and a non-PDF sourceUrl,
but their DOJ URL is constructible from the document ID:
    id=efta-efta00128482 → https://www.justice.gov/epstein/files/DataSet%209/EFTA00128482.pdf
    id=efta-00000495     → https://www.justice.gov/epstein/files/DataSet%201/EFTA00000495.pdf

This script:
1. Finds docs with missing pdfUrl by source
2. Constructs candidate URLs based on source dataset number
3. Verifies via HEAD request (in parallel with rate limiting)
4. Updates the Neon documents.pdfUrl column

Usage:
    python scripts/backfill-pdf-urls.py --source efta-ds9 --dry-run
    python scripts/backfill-pdf-urls.py --source efta-ds9
    python scripts/backfill-pdf-urls.py --all
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import sys
import time
from pathlib import Path

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backfill")


def get_neon_url() -> str:
    url = os.environ.get("EPSTEIN_NEON_DATABASE_URL")
    if url:
        return url
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("EPSTEIN_NEON_DATABASE_URL="):
                u = line.split("=", 1)[1].strip().strip('"').strip("'")
                if "sslnegotiation" in u:
                    u = re.sub(r"[&?]sslnegotiation=[^&]*", "", u)
                return u
    raise RuntimeError("EPSTEIN_NEON_DATABASE_URL not set")


# Mapping of source → dataset numbers to try (home DS first, then others as fallback)
# DOJ has shuffled files across datasets over time, so always try broadly
ALL_DATASETS = list(range(1, 13))

SOURCE_TO_DATASETS = {
    "efta-ds1": [1] + [d for d in ALL_DATASETS if d != 1],
    "efta-ds2": [2] + [d for d in ALL_DATASETS if d != 2],
    "efta-ds3": [3] + [d for d in ALL_DATASETS if d != 3],
    "efta-ds4": [4] + [d for d in ALL_DATASETS if d != 4],
    "efta-ds5": [5] + [d for d in ALL_DATASETS if d != 5],
    "efta-ds6": [6] + [d for d in ALL_DATASETS if d != 6],
    "efta-ds7": [7] + [d for d in ALL_DATASETS if d != 7],
    "efta-ds8": [8] + [d for d in ALL_DATASETS if d != 8],
    "efta-ds9": [9, 10, 11] + [d for d in ALL_DATASETS if d not in (9, 10, 11)],
    "efta-ds10": [10, 11, 9] + [d for d in ALL_DATASETS if d not in (9, 10, 11)],
    "efta-ds11": [11, 10, 9] + [d for d in ALL_DATASETS if d not in (9, 10, 11)],
    "efta-ds12": [12] + [d for d in ALL_DATASETS if d != 12],
    # These sources have docs spanning multiple datasets — try all
    "doj-efta": ALL_DATASETS,
    "doj": ALL_DATASETS,
    "house-oversight": [11, 10, 9] + [d for d in ALL_DATASETS if d not in (9, 10, 11)],
    "court-unsealed": ALL_DATASETS,
}


def extract_efta_num(doc_id: str, title: str = "") -> str | None:
    """Extract the 8-digit EFTA number from doc ID or title."""
    # Prefer ID: efta-efta00128482 or efta-00000495
    m = re.search(r"(\d{8})", doc_id)
    if m:
        return m.group(1)
    # Fall back to title: "[EFTA00003256]" or "EFTA00003256"
    m = re.search(r"EFTA(\d{8})", title, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def candidate_urls(efta_num: str, datasets: list[int]) -> list[tuple[int, str]]:
    """Build candidate DOJ URLs for each dataset."""
    return [
        (ds, f"https://www.justice.gov/epstein/files/DataSet%20{ds}/EFTA{efta_num}.pdf")
        for ds in datasets
    ]


def verify_url(url: str, session: requests.Session) -> bool:
    """HEAD request to check if URL exists."""
    try:
        r = session.head(url, timeout=10, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def find_pdf_url(doc_id: str, title: str, datasets: list[int],
                 session: requests.Session) -> str | None:
    """Find a working PDF URL for a document."""
    efta_num = extract_efta_num(doc_id, title)
    if not efta_num:
        return None
    for _ds, url in candidate_urls(efta_num, datasets):
        if verify_url(url, session):
            return url
    return None


def process_source(source: str, dry_run: bool = False, limit: int | None = None,
                   workers: int = 10) -> dict:
    """Backfill all docs for a given source."""
    datasets = SOURCE_TO_DATASETS.get(source)
    if not datasets:
        logger.error("No dataset mapping for source: %s", source)
        return {"found": 0, "missing": 0, "updated": 0}

    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()

    query = '''SELECT id, title FROM documents
               WHERE source = %s
                 AND ("pdfUrl" IS NULL OR "pdfUrl" = '')
                 AND ("sourceUrl" IS NULL OR "sourceUrl" NOT ILIKE '%%.pdf')'''
    params = [source]
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    cur.execute(query, tuple(params))
    docs = cur.fetchall()
    logger.info("Found %d docs to process for source=%s", len(docs), source)

    session = requests.Session()
    session.headers.update({"User-Agent": "EpsteinExposedBackfill/1.0"})

    found = 0
    missing = 0
    updates: list[tuple[str, str]] = []

    def _check(doc_tuple):
        doc_id, title = doc_tuple
        url = find_pdf_url(doc_id, title or "", datasets, session)
        return (doc_id, url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (doc_id, pdf_url) in enumerate(pool.map(_check, docs), 1):
            if pdf_url:
                found += 1
                updates.append((doc_id, pdf_url))
            else:
                missing += 1
            if i % 100 == 0:
                logger.info("Progress: %d/%d (found=%d, missing=%d)",
                            i, len(docs), found, missing)

    logger.info("Source=%s: found=%d, missing=%d", source, found, missing)

    updated = 0
    if updates and not dry_run:
        logger.info("Updating %d rows in Neon...", len(updates))
        cur.executemany(
            'UPDATE documents SET "pdfUrl" = %s WHERE id = %s',
            [(url, doc_id) for doc_id, url in updates],
        )
        conn.commit()
        updated = len(updates)
        logger.info("Committed %d updates", updated)
    elif updates and dry_run:
        logger.info("DRY RUN - would update %d docs. Examples:", len(updates))
        for doc_id, url in updates[:5]:
            logger.info("  %s → %s", doc_id, url)

    cur.close()
    conn.close()
    return {"found": found, "missing": missing, "updated": updated}


def main():
    parser = argparse.ArgumentParser(description="Backfill pdfUrl for documents")
    parser.add_argument("--source", help="Single source to process (e.g. efta-ds9)")
    parser.add_argument("--all", action="store_true",
                        help="Process all sources with known dataset mappings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check URLs but don't update DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit docs per source (for testing)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent HEAD requests (default 10)")
    args = parser.parse_args()

    if not args.source and not args.all:
        parser.error("Specify --source <name> or --all")

    if args.all:
        # Process sources with most missing docs first
        sources = ["efta-ds9", "house-oversight", "doj", "court-unsealed", "doj-efta",
                   "efta-ds8", "efta-ds1", "efta-ds2", "efta-ds4", "efta-ds5",
                   "efta-ds6", "efta-ds7"]
    else:
        sources = [args.source]

    totals = {"found": 0, "missing": 0, "updated": 0}
    for src in sources:
        logger.info("=== Processing source: %s ===", src)
        stats = process_source(src, dry_run=args.dry_run, limit=args.limit,
                               workers=args.workers)
        for k in totals:
            totals[k] += stats[k]

    logger.info("=== TOTALS ===")
    logger.info("Found: %d, Missing: %d, Updated: %d",
                totals["found"], totals["missing"], totals["updated"])


if __name__ == "__main__":
    main()
