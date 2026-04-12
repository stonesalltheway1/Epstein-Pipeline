"""Fix two classes of mis-titled DS9 records:

1. "No Images Produced" (2,380 docs): DOJ slip-sheet placeholders for
   documents that were withheld from the production. Their placeholder
   PDFs are 2-3KB and OCR to just "No Images Produced". The title is
   technically accurate but confusing. Users click them expecting
   content and get nothing.

2. "� Menu" (1,923 docs): Apple AXIOM forensic reports on Epstein's
   seized computer/phones. Each is an HTML-export of a category
   (Contacts, Messages, Calendar, Calls, Notes, etc.) from a specific
   evidence ID (NYC024328.aff4, NYC024329.aff4, etc.). The "Menu" in
   the title comes from OCR'ing the navigation sidebar on page 1.
   All 1,923 have real OCR; they just need proper titles derived from
   content.

This script:
- Updates titles + summaries + tags in Neon
- Preserves bates ranges and OCR text (doesn't touch content)
- Is idempotent (safe to re-run)
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("fix-ds9")


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


def extract_bates(doc_id: str) -> str | None:
    m = re.search(r"(\d{8})", doc_id)
    return m.group(1) if m else None


# ── "Menu" docs: parse OCR to derive proper title ──────────────────────────

EVIDENCE_ID_RE = re.compile(r"(NYC\d{6}\.aff4)", re.IGNORECASE)
CATEGORY_KEYWORDS = [
    ("Contacts", r"Case Data:\s*Contacts\s*\((\d+-\d+\s*of\s*\d+)\)"),
    ("Contacts", r"\bContacts\b"),
    ("Calendar Items", r"\bCalendar Items\b"),
    ("Calls", r"\bCalls\b"),
    ("Messages", r"Case Data:\s*Messages\s*\((\d+-\d+\s*of\s*\d+)\)"),
    ("Messages", r"\bMessages\b"),
    ("Notes", r"\bNotes\b"),
    ("Top Contacts", r"\bTop Contacts\b"),
    ("User Accounts", r"\bUser Accounts\b"),
    ("Device Details", r"\bDevice Details\b"),
    ("Overview", r"\bOverview\b"),
]


def derive_menu_title(ocr_text: str, bates: str) -> tuple[str, str, list[str]]:
    """Given OCR text of a 'Menu' forensic doc, return (title, summary, tags)."""
    # Extract evidence ID
    m = EVIDENCE_ID_RE.search(ocr_text)
    evidence_id = m.group(1).upper() if m else None

    # Determine category + page range
    category = "Unknown"
    page_range = None
    for cat, pattern in CATEGORY_KEYWORDS:
        rm = re.search(pattern, ocr_text, re.IGNORECASE)
        if rm:
            category = cat
            if rm.groups():
                page_range = rm.group(1)
            break

    # Build title (no em-dashes — they display as garbage in some terminals
    # and also trip common AI-generated-content heuristics)
    title_parts = [f"Forensic Report: {category}"]
    if page_range:
        title_parts.append(f"({page_range})")
    if evidence_id:
        title_parts.append(f"- Evidence {evidence_id}")
    if not evidence_id:
        title_parts.append(f"- EFTA{bates}")
    title = " ".join(title_parts)

    # Summary
    summary_parts = [
        f"Apple AXIOM forensic report export on seized device evidence."
    ]
    if evidence_id:
        summary_parts.append(f"Evidence ID: {evidence_id}.")
    summary_parts.append(f"Category: {category}.")
    if page_range:
        summary_parts.append(f"Entries {page_range}.")
    summary_parts.append(f"Bates: EFTA{bates}.")
    summary = " ".join(summary_parts)

    # Tags
    tags = ["forensic-report", "device-extraction", "axiom", f"dataset-9"]
    if evidence_id:
        tags.append(f"evidence-{evidence_id.lower().replace('.', '-')}")
    if category != "Unknown":
        tags.append(f"category-{category.lower().replace(' ', '-')}")

    return title, summary, tags


def fix_no_images_produced(cur, dry_run: bool) -> int:
    """Update 2,380 'No Images Produced' docs with clearer titles."""
    cur.execute(
        """SELECT id, "batesRange", tags FROM documents
           WHERE title = 'No Images Produced' AND source = 'efta-ds9'"""
    )
    rows = cur.fetchall()
    logger.info("Found %d 'No Images Produced' records", len(rows))

    updates = []
    for doc_id, bates_range, tags in rows:
        bates = extract_bates(doc_id) or (bates_range or "").replace("EFTA", "")
        if not bates:
            continue
        new_title = f"EFTA{bates} (Withheld by DOJ: No Images Produced)"
        new_summary = (
            f"DOJ marked EFTA{bates} as 'No Images Produced' in the Data Set 9 "
            "production. The bates number was issued but the document contents "
            "were withheld. The placeholder PDF is a standard slip-sheet; no "
            "page images or text were released for this record."
        )
        new_tags = list(tags or [])
        if "withheld" not in new_tags:
            new_tags.append("withheld")
        if "no-content" not in new_tags:
            new_tags.append("no-content")
        if "dataset-9-placeholder" not in new_tags:
            new_tags.append("dataset-9-placeholder")
        updates.append((new_title, new_summary, Json(new_tags), doc_id))

    logger.info("Will update %d rows", len(updates))
    if dry_run:
        logger.info("DRY RUN — samples:")
        for u in updates[:3]:
            logger.info("  id=%s title=%s", u[3], u[0])
        return 0

    execute_batch(
        cur,
        """UPDATE documents
           SET title = %s, summary = %s, tags = %s
           WHERE id = %s""",
        updates,
        page_size=500,
    )
    return len(updates)


def fix_menu_docs(cur, dry_run: bool) -> int:
    """Update ~1,923 forensic-report docs with real titles derived from OCR."""
    cur.execute(
        """SELECT d.id, d."batesRange", d.tags, o.text
           FROM documents d
           JOIN ocr_text o ON o."docId" = d.id
           WHERE d.title ILIKE '%menu%' AND d.source = 'efta-ds9'
             AND LENGTH(o.text) > 200"""
    )
    rows = cur.fetchall()
    logger.info("Found %d Menu-titled forensic docs with OCR", len(rows))

    updates = []
    skipped = 0
    for doc_id, bates_range, tags, ocr_text in rows:
        bates = extract_bates(doc_id) or (bates_range or "").replace("EFTA", "")
        if not bates:
            skipped += 1
            continue
        title, summary, new_tags = derive_menu_title(ocr_text, bates)
        merged_tags = list(tags or [])
        for t in new_tags:
            if t not in merged_tags:
                merged_tags.append(t)
        updates.append((title, summary, Json(merged_tags), doc_id))

    logger.info("Will update %d rows (skipped %d)", len(updates), skipped)
    if dry_run:
        logger.info("DRY RUN — samples:")
        for u in updates[:5]:
            logger.info("  id=%s title=%s", u[3], u[0])
        return 0

    execute_batch(
        cur,
        """UPDATE documents
           SET title = %s, summary = %s, tags = %s
           WHERE id = %s""",
        updates,
        page_size=500,
    )
    return len(updates)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-images-only", action="store_true")
    parser.add_argument("--menu-only", action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(get_neon_url())
    conn.autocommit = False
    cur = conn.cursor()

    try:
        total = 0
        if not args.menu_only:
            total += fix_no_images_produced(cur, args.dry_run)
        if not args.no_images_only:
            total += fix_menu_docs(cur, args.dry_run)

        if not args.dry_run:
            conn.commit()
            logger.info("Committed. Total rows updated: %d", total)
        else:
            logger.info("DRY RUN complete. Would update: %d", total)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
