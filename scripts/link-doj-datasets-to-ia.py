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
    "efta-ds12": ("data-set-12_20260131", "DataSet 12.zip"),
    # DS9 gap-repair: only covers 11 specific corrupted-then-fixed files
    "efta-ds9-gap": ("ds-9-efta-gap-repair", "DS9_EFTA_Gap_Repair.zip"),
    # Dec 19 2025 HR 4405 first compliance batch (3,951 EFTA PDFs, direct access)
    "efta-dec2025": ("efta-19-dec-2025", None),
    # Big DS9/10/11 mirrors — read zip dirs remotely via HTTP range (no local DL)
    "efta-ds9-full": ("epstein_library_transparency_act_hr_4405_dataset9_202602", "DataSet 9.zip"),
    "efta-ds10-full": ("epstein_library_transparency_act_hr_4405_dataset10_202605", "DataSet 10.zip"),
    "efta-ds11-full": ("epstein_library_transparency_act_hr_4405_dataset11_202602", "DataSet 11.zip"),
    # doj-ds10 source (17,052 records) has broken efts.fbi.gov URLs — fix with DS10 IA mirror
    "doj-ds10-fix": ("epstein_library_transparency_act_hr_4405_dataset10_202605", "DataSet 10.zip"),
    # TODO: efta-ds9, efta-ds10, efta-ds11 — would need the full 107GB / 84GB / 27GB
    #  mirrors if we want to fill the remaining 25K DS9 + 486K DS10 gaps
}


class _HttpRangeFile:
    """File-like wrapper that reads arbitrary byte ranges over HTTP.

    Lets zipfile.ZipFile read the central directory of a remote zip
    without downloading the whole file.
    """
    def __init__(self, url: str):
        self.url = url
        r = requests.head(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers.get("Content-Length"))
        self.pos = 0

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size < 0 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        end = self.pos + size - 1
        r = requests.get(self.url, timeout=120,
                         headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        self.pos += size
        return r.content


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
        # Prefer a local zip if we have it (faster); else fetch zip directory
        # remotely via HTTP range requests (no download needed).
        import zipfile
        local_zip = Path("archive-org-downloads") / item_id / zip_name

        def enumerate_zip(zf: zipfile.ZipFile, source_label: str) -> None:
            for name in zf.namelist():
                if not name.lower().endswith(".pdf"):
                    continue
                m = re.search(r"EFTA(\d{8})", name, re.IGNORECASE)
                if not m:
                    continue
                efta_num = m.group(1)
                ia_url = (
                    f"https://archive.org/download/{item_id}/"
                    f"{quote(zip_name, safe='')}/{quote(name, safe='/')}"
                )
                mapping[efta_num] = ia_url
            logger.info("  %s: %d EFTA PDFs mapped", source_label, len(mapping))

        if local_zip.exists():
            with zipfile.ZipFile(local_zip) as zf:
                enumerate_zip(zf, "Scanned local zip")
        else:
            # Stream zip directory from IA via HTTP range requests
            zip_url = (f"https://archive.org/download/{item_id}/"
                       f"{quote(zip_name, safe='')}")
            logger.info("  Reading remote zip directory: %s", zip_url)
            f = _HttpRangeFile(zip_url)
            with zipfile.ZipFile(f) as zf:
                enumerate_zip(zf, f"Remote scan ({f.size/1e9:.2f} GB zip)")
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


def update_neon(source: str, mapping: dict[str, str], dry_run: bool = False,
                force_replace_justice_gov: bool = False) -> dict:
    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()

    if source == "__any__":
        # Cross-source: match any record whose id contains an EFTA number
        cur.execute('''SELECT id, "pdfUrl" FROM documents WHERE id ~ 'efta' ''')
    else:
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
            # Swap broken/fragile URLs for IA (resilience)
            # - efts.fbi.gov: DNS does not resolve, always broken
            # - justice.gov: gated by Akamai WAF and may vanish
            if "efts.fbi.gov" in pdf_url:
                updates.append((doc_id, ia_url))
            elif force_replace_justice_gov and "justice.gov" in pdf_url:
                updates.append((doc_id, ia_url))
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
    parser.add_argument("--force-replace-justice-gov", action="store_true",
                        help="Also replace justice.gov pdfUrls with IA (for resilience)")
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
        # The ds9-gap key maps to source=efta-ds9 in Neon
        neon_source = "efta-ds9" if source == "efta-ds9-gap" else source
        # -full keys map to the base DS source
        if source.endswith("-full"):
            neon_source = source[:-5]  # strip "-full"
        if source == "doj-ds10-fix":
            neon_source = "doj-ds10"
        # Dec 2025 release spans multiple sources — match any EFTA id
        if source == "efta-dec2025":
            neon_source = "__any__"
        stats = update_neon(neon_source, mapping, dry_run=args.dry_run,
                            force_replace_justice_gov=args.force_replace_justice_gov)
        for k in totals:
            totals[k] += stats[k]

    logger.info("=== TOTALS: %s", totals)


if __name__ == "__main__":
    main()
