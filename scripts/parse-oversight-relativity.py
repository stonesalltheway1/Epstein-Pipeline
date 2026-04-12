"""Parse Oversight Relativity .dat load files and enrich Neon records.

The House Oversight Committee's Epstein estate productions follow the
Concordance/Relativity standard: document-level metadata is in a .dat
file using thorn (þ, 0xFE in UTF-8 is c3 be) as the text qualifier and
DC4 (\\x14) as the field separator.

Each row represents ONE document (not one page). It lists:
  - Bates Begin/End (single page docs have Begin=End)
  - Bates Begin Attach/End Attach (for email+attachment groups)
  - Pages, Author, Custodian, Dates
  - Email From/To/CC/BCC/Subject (for email documents)
  - Document Extension, Original Filename
  - Text Link (path to extracted text file)
  - Native Link (path to native file, e.g. .xls, .msg, .mov)

We use this to:
  1. Update Neon records with true document metadata
  2. For records with native files: point at native file URL
  3. For records where a slip-sheet JPG is useless (Excel etc): clear pdfUrl
     so the viewer defaults to the text view which has the real content.
  4. Tag each record with the native document extension for UI display.
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
from psycopg2.extras import Json, execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("relativity")


# Maps IA item id → (.dat file path within item, native URL prefix)
ITEMS = [
    ("oversight-committee-additional-epstein-files", "DATA/HOUSE_OVERSIGHT_009.dat"),
    ("Epstein_Estate_Documents_-_Seventh_Production", "DATA/HOUSE_OVERSIGHT_009.dat"),
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


def parse_dat(url: str) -> list[dict]:
    """Parse a Concordance-format .dat file with thorn/DC4 separators."""
    logger.info("Fetching .dat: %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")  # strips BOM

    # Records are \r\n separated; fields are separated by þ[DC4]þ
    # Each field is wrapped by þ...þ
    # Format: þval1þ[DC4]þval2þ[DC4]...[CRLF]
    lines = text.split("\r\n")
    records = []
    # First line is headers (same format)
    header_fields = [f for f in lines[0].split("\x14")]
    headers = [f.strip("\ufeff").strip("þ") for f in header_fields]
    logger.info("  Headers (%d): %s", len(headers), headers[:5])

    for line in lines[1:]:
        if not line.strip():
            continue
        fields = [f.strip("þ") for f in line.split("\x14")]
        if len(fields) < len(headers):
            fields += [""] * (len(headers) - len(fields))
        rec = dict(zip(headers, fields))
        records.append(rec)

    logger.info("  Parsed %d records", len(records))
    return records


def extract_bates_range(bates_str: str) -> list[str]:
    """Given 'HOUSE_OVERSIGHT_016552' return the numeric string for matching."""
    m = re.search(r"HOUSE_OVERSIGHT_(\d+)", bates_str or "")
    return m.group(1).zfill(6) if m else ""


def build_enrichment_map(item_id: str, dat_path: str) -> dict[str, dict]:
    """Returns {bates_number: enrichment_dict} for each record.

    For attachments (BatesBegin != BatesEnd), also adds entries for the
    entire range so child bates resolve to parent's metadata.
    """
    dat_url = f"https://archive.org/download/{item_id}/{quote(dat_path)}"
    records = parse_dat(dat_url)

    mapping: dict[str, dict] = {}
    for rec in records:
        begin = extract_bates_range(rec.get("Bates Begin", ""))
        end = extract_bates_range(rec.get("Bates End", ""))
        native_link = rec.get("Native Link", "").strip()
        text_link = rec.get("Text Link", "").strip()
        extension = rec.get("Document Extension", "").strip().lower()
        original_filename = rec.get("Original Filename", "").strip()

        # If extension is empty, try to derive from native_link
        if not extension and native_link:
            m = re.search(r"\.([a-z0-9]+)$", native_link.lower())
            if m:
                extension = m.group(1)

        # Build native file URL (replace backslash path with IA path)
        native_url = None
        if native_link:
            native_path = native_link.lstrip("\\").replace("\\", "/")
            # Strip the HOUSE_OVERSIGHT_009 prefix — IA uses flat NATIVES/
            native_path = re.sub(r"^HOUSE_OVERSIGHT_\d+/", "", native_path)
            native_url = (f"https://archive.org/download/{item_id}/"
                          f"{quote(native_path, safe='/')}")

        text_url = None
        if text_link:
            text_path = text_link.lstrip("\\").replace("\\", "/")
            text_path = re.sub(r"^HOUSE_OVERSIGHT_\d+/", "", text_path)
            text_url = (f"https://archive.org/download/{item_id}/"
                        f"{quote(text_path, safe='/')}")

        enrichment = {
            "extension": extension or None,
            "original_filename": original_filename or None,
            "native_url": native_url,
            "text_url": text_url,
            "author": rec.get("Author", "").strip() or None,
            "custodian": rec.get("Custodian/Source", "").strip() or None,
            "date_created": rec.get("Date Created", "").strip() or None,
            "date_sent": rec.get("Date Sent", "").strip() or None,
            "email_from": rec.get("Email From", "").strip() or None,
            "email_to": rec.get("Email To", "").strip() or None,
            "email_subject": rec.get("Email Subject/Title", "").strip() or None,
            "document_title": rec.get("Document Title", "").strip() or None,
            "file_size": rec.get("File Size", "").strip() or None,
            "ia_item": item_id,
            "bates_begin": begin,
            "bates_end": end,
        }

        # If begin == end, single doc. Otherwise, apply to every bates in range
        if begin and end:
            try:
                b_int = int(begin)
                e_int = int(end)
                for b in range(b_int, e_int + 1):
                    mapping[str(b).zfill(6)] = enrichment
            except ValueError:
                if begin:
                    mapping[begin] = enrichment

    return mapping


def update_neon(mapping: dict[str, dict], dry_run: bool = False) -> dict:
    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()

    cur.execute(
        '''SELECT id, "batesRange", "pdfUrl", tags FROM documents
           WHERE source = 'house-oversight' AND "batesRange" IS NOT NULL'''
    )
    rows = cur.fetchall()
    logger.info("Scanning %d house-oversight records", len(rows))

    stats = {
        "matched": 0,
        "native_linked": 0,
        "pdf_cleared_for_native": 0,
        "metadata_updated": 0,
        "no_match": 0,
    }

    doc_updates: list[tuple] = []  # (new_pdf_url, new_tags, new_summary, doc_id)
    metadata_updates: list[tuple] = []  # (metadata_json, doc_id) — sidecar table

    for doc_id, bates, old_pdf_url, old_tags in rows:
        b = extract_bates_range(bates)
        if not b:
            stats["no_match"] += 1
            continue
        meta = mapping.get(b)
        if not meta:
            stats["no_match"] += 1
            continue
        stats["matched"] += 1

        new_tags = list(old_tags or [])
        ext = meta.get("extension")
        if ext and f"native:{ext}" not in new_tags:
            new_tags.append(f"native:{ext}")

        # Decide pdfUrl:
        # - If native is PDF: link to native PDF
        # - If native is non-PDF (xls, msg, mov): clear pdfUrl so text view wins
        # - If no native link: keep the existing JPG url (it's a real scan)
        new_pdf_url = old_pdf_url
        native_url = meta.get("native_url")
        if native_url:
            if ext == "pdf":
                new_pdf_url = native_url
                stats["native_linked"] += 1
            else:
                # Non-PDF native: clear pdfUrl so text view is default
                new_pdf_url = None
                stats["pdf_cleared_for_native"] += 1
                if "native-format" not in new_tags:
                    new_tags.append("native-format")

        # Always update tags
        doc_updates.append((new_pdf_url, Json(new_tags), doc_id))
        metadata_updates.append((Json({k: v for k, v in meta.items() if v is not None}), doc_id))
        stats["metadata_updated"] += 1

    logger.info("Stats: %s", stats)

    if dry_run:
        logger.info("DRY RUN - sample doc updates:")
        for u in doc_updates[:5]:
            logger.info("  pdfUrl=%s tags=%s id=%s",
                        str(u[0])[:80] if u[0] else "NULL", u[1], u[2])
    else:
        logger.info("Applying %d doc updates...", len(doc_updates))
        execute_batch(
            cur,
            '''UPDATE documents SET "pdfUrl" = %s, tags = %s WHERE id = %s''',
            doc_updates, page_size=500,
        )
        conn.commit()
        logger.info("Committed")

    cur.close()
    conn.close()
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Merge enrichment from both IA items
    combined: dict[str, dict] = {}
    for item_id, dat_path in ITEMS:
        try:
            m = build_enrichment_map(item_id, dat_path)
            for bates, enr in m.items():
                if bates not in combined:
                    combined[bates] = enr
        except Exception as e:
            logger.exception("Failed to parse %s/%s: %s", item_id, dat_path, e)

    logger.info("Combined enrichment covers %d bates numbers", len(combined))
    stats = update_neon(combined, dry_run=args.dry_run)
    logger.info("FINAL: %s", stats)


if __name__ == "__main__":
    main()
