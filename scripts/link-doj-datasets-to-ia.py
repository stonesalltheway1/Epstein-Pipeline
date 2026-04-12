"""Link DOJ EFTA DataSet records to Archive.org-hosted PDFs.

The DOJ DS1-DS12 releases are mirrored on Archive.org as zip files. IA's file
server extracts individual files from those zips on demand via a URL pattern:

    https://archive.org/download/{item}/{zip-name}/{path/inside/zip.pdf}

Our Neon records for efta-ds1, efta-ds8, efta-ds12 have 0 pdfUrl populated.
This script backfills pdfUrl by constructing IA URLs from the EFTA number in
each doc's id (e.g. id=efta-00000495 → EFTA00000495).

For each dataset, we fetch IA's files.xml to learn the exact in-zip path
(IMAGES/0001/ vs 0002/ etc), then map doc_id → IA URL and UPDATE Neon.

No files are downloaded locally.
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
logger = logging.getLogger("doj-link")


# IA items that contain each DataSet. Value is (item_id, zip_filename).
# None for zip_filename if files are directly accessible.
DATASET_MAP: dict[str, tuple[str, str | None]] = {
    "efta-ds1": ("epstein_library_transparency_act_hr_4405_dataset1_20260204",
                 "DataSet 1.zip"),
    "efta-ds8": ("epstein_library_transparency_act_hr_4405_dataset8", "DataSet 8.zip"),
    # Add more as we identify them:
    # "efta-ds9": ("epstein_library_transparency_act_hr_4405_dataset9_202602", None),
    # "efta-ds10": ("epstein_library_transparency_act_hr_4405_dataset10_202605", None),
    # "efta-ds11": ("epstein_library_transparency_act_hr_4405_dataset11_202602", None),
}


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


def build_efta_url_map(item_id: str, zip_name: str | None) -> dict[str, str]:
    """Map EFTA number → IA URL for every PDF in the item.

    If zip_name is given, files are inside that zip; IA serves them via
    the /item/zip/path-in-zip URL pattern.
    """
    logger.info("Fetching IA metadata for %s...", item_id)
    r = requests.get(f"https://archive.org/metadata/{item_id}", timeout=30)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", [])

    mapping: dict[str, str] = {}

    if zip_name:
        # The IA file listing only shows the zip itself; we need to enumerate
        # the zip contents via the in-zip URL pattern. We know the structure
        # from our local extraction.
        # Pattern: DataSet 1/DataSet 1/VOL00001/IMAGES/0001/EFTA00000001.pdf
        # For now, we ask the user to provide a local zip to scan, or we can
        # hit IA's `/{item}/{zip}/` endpoint which returns an HTML file listing.

        # Best path: scan the local zip if we have it
        local_base = Path("archive-org-downloads") / item_id
        local_zip = local_base / zip_name
        if local_zip.exists():
            import zipfile
            with zipfile.ZipFile(local_zip) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".pdf"):
                        continue
                    m = re.search(r"EFTA(\d{8})", name, re.IGNORECASE)
                    if not m:
                        continue
                    efta_num = m.group(1)
                    # Build the IA extract URL — encode each path segment
                    ia_url = (
                        f"https://archive.org/download/{item_id}/"
                        f"{quote(zip_name, safe='')}/{quote(name, safe='/')}"
                    )
                    mapping[efta_num] = ia_url
            logger.info("  Scanned local zip: %d EFTA PDFs mapped", len(mapping))
        else:
            logger.error("Local zip not found at %s; can't enumerate", local_zip)
            return {}
    else:
        # Direct file listing on IA (no zip)
        for f in files:
            name = f.get("name", "")
            if not name.lower().endswith(".pdf"):
                continue
            m = re.search(r"EFTA(\d{8})", name, re.IGNORECASE)
            if not m:
                continue
            efta_num = m.group(1)
            url = f"https://archive.org/download/{item_id}/{quote(name, safe='/')}"
            mapping[efta_num] = url
        logger.info("  %d direct EFTA PDFs found", len(mapping))

    return mapping


def update_neon(source: str, mapping: dict[str, str], dry_run: bool = False) -> dict:
    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()

    cur.execute('''SELECT id, "pdfUrl" FROM documents WHERE source = %s''', (source,))
    rows = cur.fetchall()
    logger.info("Source=%s: %d rows in Neon", source, len(rows))

    stats = {"scanned": len(rows), "matched": 0, "already_had": 0, "updated": 0}
    updates: list[tuple[str, str]] = []
    for doc_id, pdf_url in rows:
        # Extract EFTA number from id (efta-00000495 or efta-efta00000495 etc)
        m = re.search(r"(\d{8})", doc_id)
        if not m:
            continue
        efta_num = m.group(1)
        ia_url = mapping.get(efta_num)
        if not ia_url:
            continue
        stats["matched"] += 1
        if pdf_url:
            stats["already_had"] += 1
            continue
        updates.append((doc_id, ia_url))

    logger.info("Matched: %d, already had pdfUrl: %d, to update: %d",
                stats["matched"], stats["already_had"], len(updates))

    if dry_run:
        logger.info("DRY RUN - sample:")
        for doc_id, url in updates[:5]:
            logger.info("  %s → %s", doc_id, url)
    elif updates:
        from psycopg2.extras import execute_batch
        execute_batch(
            cur,
            '''UPDATE documents SET "pdfUrl" = %s WHERE id = %s''',
            [(url, doc_id) for doc_id, url in updates],
            page_size=500,
        )
        conn.commit()
        stats["updated"] = len(updates)
        logger.info("Committed %d pdfUrl updates for %s", stats["updated"], source)

    cur.close()
    conn.close()
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Single source (e.g. efta-ds1)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = DATASET_MAP if args.all else {args.source: DATASET_MAP[args.source]} \
        if args.source in DATASET_MAP else {}
    if not targets:
        parser.error(f"Specify --source {list(DATASET_MAP.keys())} or --all")

    totals = {"scanned": 0, "matched": 0, "already_had": 0, "updated": 0}
    for source, (item_id, zip_name) in targets.items():
        logger.info("=== %s via %s ===", source, item_id)
        mapping = build_efta_url_map(item_id, zip_name)
        if not mapping:
            continue
        stats = update_neon(source, mapping, dry_run=args.dry_run)
        for k in totals:
            totals[k] += stats[k]

    logger.info("=== TOTALS: %s", totals)


if __name__ == "__main__":
    main()
