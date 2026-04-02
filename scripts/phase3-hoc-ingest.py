"""Phase 3: Full HOC Document Ingestion Pipeline

Orchestrates the complete ingestion of House Oversight Committee documents
from the Kaggle Concordance/Relativity dataset into Neon Postgres.

Stages:
  3A. Parse Concordance .dat/.opt -> document inventory
  3B. Assemble page images into per-document PDFs
  3C. OCR quality check (compare existing HF text vs image page count)
  3D. Entity extraction + person linking (GLiNER + Aho-Corasick)
  3E. Metadata enrichment (dates, filenames, categories from .dat)
  3F. Upsert enriched metadata to Neon

Requires:
  - Extracted images in E:/Epstein-Pipeline/ingest/kaggle-hoc/IMAGES-*/
  - .dat and .opt files in DATA-*/DATA/
  - psycopg, pymupdf installed
  - DATABASE_URL in .env.local
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add pipeline src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path("E:/Epstein-Pipeline/ingest/kaggle-hoc")
DATA_DIR = BASE_DIR / "DATA-20251116T222054Z-1-001" / "DATA"
OUTPUT_DIR = Path("E:/Epstein-Pipeline/output/hoc-pdfs")

# Find all IMAGES directories
IMAGES_DIRS = sorted(BASE_DIR.glob("IMAGES-*"))

ENV_PATH = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")

BATCH_SIZE = 100  # DB batch size
CHECKPOINT_FILE = Path("E:/Epstein-Pipeline/output/hoc-checkpoint.json")


def get_db_url() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            raw = line.split("=", 1)[1].strip().strip('"')
            return re.sub(r"[&?]sslnegotiation=[^&]*", "", raw)
    raise RuntimeError("DATABASE_URL not found")


def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"assembled": [], "enriched": [], "stage": ""}


def save_checkpoint(data: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Stage 3A: Parse Concordance files
# ---------------------------------------------------------------------------

def stage_3a():
    """Parse .dat + .opt into document inventory."""
    print("=" * 60)
    print("STAGE 3A: Parse Concordance files")
    print("=" * 60)

    from epstein_pipeline.importers.hoc_loader import HocLoader
    loader = HocLoader(DATA_DIR, IMAGES_DIRS)
    documents = loader.load()
    return documents, loader


# ---------------------------------------------------------------------------
# Stage 3B: Assemble PDFs from page images
# ---------------------------------------------------------------------------

def stage_3b(documents, loader):
    """Assemble page images into per-document PDFs."""
    print("\n" + "=" * 60)
    print("STAGE 3B: Assemble PDFs from page images")
    print("=" * 60)

    from epstein_pipeline.importers.hoc_loader import assemble_pdf

    checkpoint = load_checkpoint()
    already_assembled = set(checkpoint.get("assembled", []))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    assembled = 0
    skipped = 0
    failed = 0
    total = len(documents)

    for i, doc in enumerate(documents):
        if doc.doc_id in already_assembled:
            skipped += 1
            continue

        pdf_path = assemble_pdf(doc, loader, OUTPUT_DIR)
        if pdf_path:
            assembled += 1
            already_assembled.add(doc.doc_id)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (assembled + skipped + failed) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            print(f"  {i+1:>5} / {total} ({pct:.1f}%)  assembled={assembled} skipped={skipped} failed={failed}  ~{fmt_time(remaining)} remaining")

            # Save checkpoint periodically
            checkpoint["assembled"] = list(already_assembled)
            checkpoint["stage"] = "3b"
            save_checkpoint(checkpoint)

    checkpoint["assembled"] = list(already_assembled)
    checkpoint["stage"] = "3b_complete"
    save_checkpoint(checkpoint)

    elapsed = time.time() - t0
    print(f"\n  Done in {fmt_time(elapsed)}: {assembled} assembled, {skipped} skipped, {failed} failed (no images)")
    return assembled


# ---------------------------------------------------------------------------
# Stage 3E: Metadata enrichment -> Neon upsert
# ---------------------------------------------------------------------------

def stage_3e(documents):
    """Enrich existing kaggle-ho-* documents with metadata from .dat file.

    Updates: title, date, category, batesRange, pageCount, sourceUrl
    """
    print("\n" + "=" * 60)
    print("STAGE 3E: Metadata enrichment -> Neon upsert")
    print("=" * 60)

    import psycopg

    db_url = get_db_url()
    t0 = time.time()
    updated = 0
    not_found = 0
    errors = 0

    conn = psycopg.connect(db_url)

    for i, doc in enumerate(documents):
        doc_id = doc.doc_id
        title = doc.best_title
        date = doc.best_date
        category = doc.category
        page_count = doc.page_count or len(doc.pages)
        bates_range = f"{doc.bates_begin} - {doc.bates_end}" if doc.bates_begin != doc.bates_end else doc.bates_begin

        try:
            # Only update fields that have useful data (don't overwrite good titles with generic ones)
            if doc.doc_title or doc.email_subject or doc.original_filename:
                # We have a real title from metadata
                result = conn.execute(
                    """UPDATE documents SET
                        title = CASE WHEN title LIKE 'HOUSE_OVERSIGHT_%' OR title LIKE 'kaggle-ho-%' THEN %s ELSE title END,
                        date = COALESCE(%s::date, date),
                        category = CASE WHEN category = 'other' THEN %s ELSE category END,
                        "pageCount" = CASE WHEN "pageCount" = 0 THEN %s ELSE "pageCount" END,
                        "batesRange" = COALESCE("batesRange", %s)
                    WHERE id = %s
                    RETURNING id""",
                    (title, date, category, page_count, bates_range, doc_id),
                )
            else:
                # Only update non-title fields
                result = conn.execute(
                    """UPDATE documents SET
                        date = COALESCE(%s::date, date),
                        category = CASE WHEN category = 'other' THEN %s ELSE category END,
                        "pageCount" = CASE WHEN "pageCount" = 0 THEN %s ELSE "pageCount" END,
                        "batesRange" = COALESCE("batesRange", %s)
                    WHERE id = %s
                    RETURNING id""",
                    (date, category, page_count, bates_range, doc_id),
                )

            if result.fetchone():
                updated += 1
            else:
                not_found += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {doc_id}: {e}")
            conn.rollback()
            conn = psycopg.connect(db_url)

        if (i + 1) % 500 == 0:
            conn.commit()
            elapsed = time.time() - t0
            pct = (i + 1) / len(documents) * 100
            print(f"  {i+1:>5} / {len(documents)} ({pct:.1f}%)  updated={updated} not_found={not_found} errors={errors}")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"\n  Done in {fmt_time(elapsed)}: {updated} updated, {not_found} not in DB, {errors} errors")
    return updated


# ---------------------------------------------------------------------------
# Stage 3F: Count and verify
# ---------------------------------------------------------------------------

def stage_3f():
    """Verify the enrichment results."""
    print("\n" + "=" * 60)
    print("STAGE 3F: Verification")
    print("=" * 60)

    import psycopg

    db_url = get_db_url()
    with psycopg.connect(db_url) as conn:
        # Count HOC docs
        total = conn.execute("SELECT COUNT(*) FROM documents WHERE id LIKE 'kaggle-ho-%'").fetchone()[0]
        with_date = conn.execute("SELECT COUNT(*) FROM documents WHERE id LIKE 'kaggle-ho-%' AND date IS NOT NULL").fetchone()[0]
        with_bates = conn.execute("""SELECT COUNT(*) FROM documents WHERE id LIKE 'kaggle-ho-%' AND "batesRange" IS NOT NULL AND "batesRange" != ''""").fetchone()[0]
        with_ocr = conn.execute("""SELECT COUNT(DISTINCT o."docId") FROM ocr_text o WHERE o."docId" LIKE 'kaggle-ho-%'""").fetchone()[0]
        with_title = conn.execute("SELECT COUNT(*) FROM documents WHERE id LIKE 'kaggle-ho-%' AND title NOT LIKE 'HOUSE_OVERSIGHT_%'").fetchone()[0]
        with_persons = conn.execute("""SELECT COUNT(DISTINCT dp.doc_id) FROM document_persons dp WHERE dp.doc_id LIKE 'kaggle-ho-%'""").fetchone()[0]

        print(f"  Total kaggle-ho docs: {total:,}")
        print(f"  With dates:           {with_date:,} ({with_date/total*100:.1f}%)")
        print(f"  With Bates range:     {with_bates:,} ({with_bates/total*100:.1f}%)")
        print(f"  With OCR text:        {with_ocr:,} ({with_ocr/total*100:.1f}%)")
        print(f"  With real titles:     {with_title:,} ({with_title/total*100:.1f}%)")
        print(f"  With person links:    {with_persons:,} ({with_persons/total*100:.1f}%)")

    # Count assembled PDFs
    pdf_count = len(list(OUTPUT_DIR.glob("*.pdf"))) if OUTPUT_DIR.exists() else 0
    print(f"  Assembled PDFs:       {pdf_count:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3: HOC Document Ingestion Pipeline")
    print("=" * 60)
    print(f"Data dir:   {DATA_DIR}")
    print(f"Images dirs: {len(IMAGES_DIRS)} found")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    # Check prerequisites
    if not DATA_DIR.exists():
        print("ERROR: DATA directory not found. Is the Kaggle extraction complete?")
        sys.exit(1)

    # Check image extraction progress
    total_images = 0
    for d in IMAGES_DIRS:
        count = sum(1 for _ in d.rglob("*.jpg"))
        total_images += count
    print(f"Images extracted: {total_images:,} / 23,124")

    if total_images < 20000:
        print(f"WARNING: Only {total_images/23124*100:.0f}% of images extracted. PDF assembly will be incomplete.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            print("Aborted. Wait for extraction to complete.")
            sys.exit(0)

    t0 = time.time()

    # Stage 3A: Parse
    documents, loader = stage_3a()

    # Stage 3B: Assemble PDFs
    stage_3b(documents, loader)

    # Stage 3E: Metadata enrichment
    stage_3e(documents)

    # Stage 3F: Verify
    stage_3f()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 3 complete in {fmt_time(total_time)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
