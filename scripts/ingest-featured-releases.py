"""Ingest a curated list of high-value single-document releases.

Targets specific Senate, court, and government releases that are too
small/irregular for a full scraped-source downloader but are high-value
public records. Each entry is (url, id, title, source, category, date).

Downloads the PDF, runs OCR with PaddleOCR, and upserts:
- documents (title, source, category, pdfUrl, sourceUrl, pageCount, summary)
- ocr_text (docId, text)
- ocr_quality_scores
- document_chunks

Usage:
    python scripts/ingest-featured-releases.py --dry-run
    python scripts/ingest-featured-releases.py
    python scripts/ingest-featured-releases.py --only wyden-sars-2025
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("featured")

# Some court sites (e.g. nysd.uscourts.gov) reject UA strings containing "Bot".
# Use a plain Mozilla identifier so we can access court-released public records.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"


@dataclass
class FeaturedDoc:
    """A single curated document release."""
    id: str              # Neon primary key (e.g. "wyden-sars-2025-11-21")
    url: str             # Direct PDF URL
    title: str           # Short title for display
    source: str          # Grouping tag (e.g. "senate-finance")
    category: str        # Legal category (e.g. "legal", "financial", "investigation")
    date: str            # ISO date (YYYY-MM-DD)
    summary: str = ""    # Optional pre-written summary (overrides OCR-derived)
    tags: tuple[str, ...] = ()


# ── Curated manifest ──────────────────────────────────────────────────────
# Only add entries where the URL has been verified to return a real PDF.
# Mark unreachable-but-known URLs with TODO comments so we don't lose them.

MANIFEST: list[FeaturedDoc] = [
    FeaturedDoc(
        id="senate-finance-wyden-sars-2025-11-21",
        url="https://www.whitehouse.senate.gov/wp-content/uploads/2025/11/Jeffrey-Epstein-SARS-.pdf",
        title="Sen. Wyden Letter Releasing Treasury SARs — Jeffrey Epstein (Nov 21, 2025)",
        source="senate-finance",
        category="financial",
        date="2025-11-21",
        summary=(
            "Senate Finance Committee ranking member Sen. Ron Wyden's Nov 21, 2025 "
            "letter releasing Treasury Department Suspicious Activity Reports (SARs) "
            "related to Jeffrey Epstein. Documents 4,725 wire transfers totaling "
            "approximately $1.08 billion across JPMorgan Chase and Deutsche Bank, "
            "spanning Epstein's active banking relationships."
        ),
        tags=("senate", "wyden", "sars", "jpmorgan", "deutsche-bank", "financial-records"),
    ),
    FeaturedDoc(
        id="sdny-maxwell-20cr330-opinion-2026-01-21",
        url="https://www.nysd.uscourts.gov/sites/default/files/2026-01/Maxwell%2020cr330%20-%20Opinion%20&%20Order%201.21.26.pdf",
        title="United States v. Maxwell (20-cr-330) — Opinion & Order (Jan 21, 2026)",
        source="sdny-court",
        category="legal",
        date="2026-01-21",
        summary=(
            "Southern District of New York Judge Paul Engelmayer's Jan 21, 2026 "
            "opinion and order addressing the motion to unseal grand jury materials "
            "in United States v. Ghislaine Maxwell (20-cr-330). Greenlights "
            "unsealing subject to certification by the SDNY US Attorney."
        ),
        tags=("sdny", "maxwell", "grand-jury", "unsealing", "engelmayer"),
    ),
    # ── Known-but-unreachable URLs (Akamai-blocked from this IP at time of capture) ──
    # TODO: retry after DOJ rate-limit expires
    # FeaturedDoc(
    #     id="doj-wpb-grand-jury-2026-01-05",
    #     url="https://www.justice.gov/multimedia/Court%20Records/In%20re%20Grand%20Jury%2005-02%20(WPB)%20&%2007-103%20(WPB),%20No.%20925-mc-80920%20(S.D.%20Fla.%202025)/001.pdf",
    #     title="West Palm Beach Grand Jury Transcripts (2005-2007) — Unsealed Dec 5, 2025",
    #     source="court-grand-jury",
    #     category="legal",
    #     date="2025-12-05",
    # ),
]


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


def download_pdf(url: str, dest: Path) -> int:
    """Download a PDF, return size in bytes."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    r = session.get(url, timeout=60, stream=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    return total


def run_paddle_ocr(pdf_path: Path) -> tuple[str, float, int]:
    """Run PaddleOCR, return (text, avg_confidence, page_count)."""
    from epstein_pipeline.processors.ocr import PaddleOcrBackend

    backend = PaddleOcrBackend()
    result = backend.extract(pdf_path)
    page_count = len(result.page_confidences)
    return result.text, result.confidence, page_count


def compute_quality_scores(text: str) -> dict:
    """Copy of quality-score logic from reocr-document.py."""
    import string
    if not text:
        return {"overall_score": 0, "gibberish_ratio": 1.0, "encoding_errors": 1.0,
                "char_noise": 1.0, "line_fragmentation": 1.0, "whitespace_ratio": 1.0,
                "repetition_score": 0.0, "digit_confusion": 0.0, "header_noise": 0.0,
                "content_density": 0.0, "text_length": 0, "word_count": 0, "line_count": 0}
    sample = text[:50000]
    words = sample.split()
    lines = sample.splitlines()
    gibberish = sum(1 for w in words if len(w) > 2 and sum(1 for c in w if c.isalpha()) / len(w) < 0.5)
    gibberish_ratio = gibberish / max(len(words), 1)
    encoding_errors = sample.count("\ufffd") / max(len(sample), 1)
    noise = sum(1 for c in sample if c in "\ufffd|[]{}\\<>^~`")
    char_noise = noise / max(len(sample), 1)
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 5)
    line_frag = short_lines / max(len(lines), 1)
    ws = sum(1 for c in sample if c in " \t\n\r")
    ws_ratio = ws / max(len(sample), 1)
    content_density = len(words) / max(len(lines), 1)
    overall = 100.0 - gibberish_ratio * 200 - encoding_errors * 500 - char_noise * 300 - line_frag * 100
    if ws_ratio > 0.3:
        overall -= (ws_ratio - 0.3) * 100
    overall = max(0, min(100, overall))
    return {"overall_score": round(overall), "gibberish_ratio": round(gibberish_ratio, 3),
            "encoding_errors": round(encoding_errors, 5), "char_noise": round(char_noise, 4),
            "line_fragmentation": round(line_frag, 3), "whitespace_ratio": round(ws_ratio, 3),
            "repetition_score": 0.0, "digit_confusion": 0.0, "header_noise": 0.0,
            "content_density": round(content_density, 2), "text_length": len(sample),
            "word_count": len(words), "line_count": len(lines)}


def chunk_text(text: str, doc_id: str, chunk_size: int = 2000, overlap: int = 200) -> list[dict]:
    chunks = []
    idx = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            para = text.rfind("\n\n", start + chunk_size // 2, end)
            if para > start:
                end = para
            else:
                sent = text.rfind(". ", start + chunk_size // 2, end)
                if sent > start:
                    end = sent + 1
        piece = text[start:end].strip()
        if piece and len(piece) > 50:
            chunks.append({"document_id": doc_id, "chunk_index": idx, "chunk_text": piece})
            idx += 1
        start = end - overlap if end < len(text) else len(text)
    return chunks


def upsert_to_neon(doc: FeaturedDoc, text: str, confidence: float, page_count: int) -> None:
    import json as _json

    import psycopg2
    from psycopg2.extras import Json, execute_values

    quality = compute_quality_scores(text)
    chunks = chunk_text(text, doc.id)

    # Use pre-written summary if provided, else take first 500 chars of cleaned text
    summary = doc.summary
    if not summary:
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) >= 30:
                summary = stripped[:500]
                break

    conn = psycopg2.connect(get_neon_url())
    conn.autocommit = False
    cur = conn.cursor()
    try:
        # documents table
        cur.execute("""
            INSERT INTO documents
              (id, title, date, source, category, summary, tags, "pdfUrl", "sourceUrl", "pageCount")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              title = EXCLUDED.title,
              date = EXCLUDED.date,
              source = EXCLUDED.source,
              category = EXCLUDED.category,
              summary = EXCLUDED.summary,
              tags = EXCLUDED.tags,
              "pdfUrl" = EXCLUDED."pdfUrl",
              "sourceUrl" = EXCLUDED."sourceUrl",
              "pageCount" = EXCLUDED."pageCount"
        """, (
            doc.id, doc.title, doc.date, doc.source, doc.category, summary,
            Json(list(doc.tags)), doc.url, doc.url, page_count,
        ))

        # ocr_text (separate table)
        cur.execute("""
            INSERT INTO ocr_text ("docId", text) VALUES (%s, %s)
            ON CONFLICT ("docId") DO UPDATE SET text = EXCLUDED.text
        """, (doc.id, text))

        # ocr_quality_scores
        cur.execute("""
            INSERT INTO ocr_quality_scores
              (doc_id, overall_score, gibberish_ratio, encoding_errors, char_noise,
               line_fragmentation, whitespace_ratio, repetition_score, digit_confusion,
               header_noise, content_density, text_length, word_count, line_count, scored_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
            ON CONFLICT (doc_id) DO UPDATE SET
              overall_score = EXCLUDED.overall_score,
              gibberish_ratio = EXCLUDED.gibberish_ratio,
              encoding_errors = EXCLUDED.encoding_errors,
              char_noise = EXCLUDED.char_noise,
              line_fragmentation = EXCLUDED.line_fragmentation,
              whitespace_ratio = EXCLUDED.whitespace_ratio,
              repetition_score = EXCLUDED.repetition_score,
              digit_confusion = EXCLUDED.digit_confusion,
              header_noise = EXCLUDED.header_noise,
              content_density = EXCLUDED.content_density,
              text_length = EXCLUDED.text_length,
              word_count = EXCLUDED.word_count,
              line_count = EXCLUDED.line_count,
              scored_at = NOW()
        """, (doc.id, quality["overall_score"], quality["gibberish_ratio"],
              quality["encoding_errors"], quality["char_noise"],
              quality["line_fragmentation"], quality["whitespace_ratio"],
              quality["repetition_score"], quality["digit_confusion"],
              quality["header_noise"], quality["content_density"],
              quality["text_length"], quality["word_count"], quality["line_count"]))

        # document_chunks — replace all chunks for this doc
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc.id,))
        if chunks:
            execute_values(cur,
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text) VALUES %s",
                [(c["document_id"], c["chunk_index"], c["chunk_text"]) for c in chunks])

        conn.commit()
        logger.info("Neon upsert: id=%s pages=%d ocr_chars=%d chunks=%d quality=%d",
                    doc.id, page_count, len(text), len(chunks), quality["overall_score"])
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def process(doc: FeaturedDoc, download_dir: Path, dry_run: bool = False) -> dict:
    dest = download_dir / f"{doc.id}.pdf"

    if not dest.exists():
        logger.info("Downloading %s ...", doc.url)
        size = download_pdf(doc.url, dest)
        logger.info("Saved %.1f KB to %s", size / 1024, dest)
    else:
        logger.info("Already downloaded: %s", dest)

    txt_path = dest.with_suffix(".txt")
    meta_path = dest.with_suffix(".meta.json")

    # Use cached OCR if both text and page count are available
    if txt_path.exists() and meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        text = txt_path.read_text(encoding="utf-8")
        confidence = meta.get("confidence", 0.0)
        page_count = meta.get("page_count", 1)
        logger.info("Cached OCR loaded: %d chars, %.3f conf, %d pages",
                    len(text), confidence, page_count)
    else:
        logger.info("Running PaddleOCR on %s ...", dest.name)
        text, confidence, page_count = run_paddle_ocr(dest)
        logger.info("OCR done: %d chars, %.3f confidence, %d pages",
                    len(text), confidence, page_count)
        if not text.strip():
            logger.error("OCR produced empty text for %s", doc.id)
            return {"id": doc.id, "status": "empty"}
        # Cache text + metadata
        txt_path.write_text(text, encoding="utf-8")
        import json
        meta_path.write_text(json.dumps({
            "confidence": confidence, "page_count": page_count,
        }), encoding="utf-8")

    if dry_run:
        logger.info("DRY RUN — skipping Neon upsert for %s", doc.id)
        return {"id": doc.id, "status": "dry-run", "pages": page_count, "chars": len(text)}

    upsert_to_neon(doc, text, confidence, page_count)
    return {"id": doc.id, "status": "ok", "pages": page_count, "chars": len(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="Process only this id")
    parser.add_argument("--download-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "temp_ocr" / "featured")
    args = parser.parse_args()

    targets = MANIFEST if not args.only else [d for d in MANIFEST if d.id == args.only]
    if not targets:
        logger.error("No docs matched filter %r", args.only)
        sys.exit(1)

    logger.info("Processing %d featured doc(s) into %s", len(targets), args.download_dir)

    results = []
    for d in targets:
        try:
            r = process(d, args.download_dir, dry_run=args.dry_run)
            results.append(r)
        except Exception as e:
            logger.exception("Failed: %s", d.id)
            results.append({"id": d.id, "status": "error", "error": str(e)})

    logger.info("=== SUMMARY ===")
    for r in results:
        logger.info("  %s: %s (%s)", r["id"], r["status"],
                    f"{r.get('pages', '?')}p {r.get('chars', 0)}c" if r["status"] == "ok" else "")


if __name__ == "__main__":
    main()
