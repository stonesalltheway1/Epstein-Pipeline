"""Re-OCR a single document with SmolDocling and update Neon DB.

Usage:
    python scripts/reocr-document.py E:/Epstein-Pipeline/temp_ocr/reocr-26223474/epstein-001.pdf \
        --doc-id dc-26223474 --backend smoldocling

This script:
1. Runs OCR on the PDF using the specified backend (default: smoldocling)
2. Updates the ocr_text table in Neon
3. Updates the documents.summary with a clean snippet
4. Updates ocr_quality_scores
5. Regenerates document_chunks for search indexing
6. Saves the full OCR text locally as a backup
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("reocr")


def get_neon_url() -> str:
    """Get Neon DB URL from env or .env file."""
    url = os.environ.get("EPSTEIN_NEON_DATABASE_URL")
    if url:
        return url
    # Try loading from .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("EPSTEIN_NEON_DATABASE_URL="):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                # Remove sslnegotiation param (psycopg2 doesn't support it)
                if "sslnegotiation" in url:
                    import re
                    url = re.sub(r"[&?]sslnegotiation=[^&]*", "", url)
                return url
    raise RuntimeError("EPSTEIN_NEON_DATABASE_URL not set and no .env found")


def run_ocr(pdf_path: Path, backend: str) -> tuple[str, float, list[float]]:
    """Run OCR on a PDF and return (text, avg_confidence, page_confidences)."""
    from epstein_pipeline.config import Settings
    from epstein_pipeline.processors.ocr import OcrProcessor

    settings = Settings()
    processor = OcrProcessor(settings, backend=backend)

    logger.info("Starting OCR with backend=%s on %s", backend, pdf_path.name)
    start = time.time()
    result = processor.process_file(pdf_path)
    elapsed = time.time() - start

    if result.errors:
        for err in result.errors:
            logger.error("OCR error: %s", err)
    if result.warnings:
        for warn in result.warnings:
            logger.warning("OCR warning: %s", warn)

    text = result.document.ocrText if result.document and result.document.ocrText else ""
    confidence = result.ocr_confidence or 0.0

    # Get page confidences from the internal result if available
    page_confs: list[float] = []
    # The ProcessingResult doesn't expose page_confidences directly,
    # so we'll compute heuristic ones from splitting by page breaks
    pages = text.split("\n\n") if text else []
    page_count = len(pages)

    logger.info(
        "OCR complete: %d chars, %.2f confidence, ~%d pages, %.1fs elapsed",
        len(text), confidence, page_count, elapsed,
    )
    return text, confidence, page_confs


def compute_quality_scores(text: str) -> dict:
    """Compute OCR quality metrics for the text."""
    import string

    if not text:
        return {
            "overall_score": 0, "gibberish_ratio": 1.0, "encoding_errors": 1.0,
            "char_noise": 1.0, "line_fragmentation": 1.0, "whitespace_ratio": 1.0,
            "repetition_score": 0.0, "digit_confusion": 0.0, "header_noise": 0.0,
            "content_density": 0.0, "text_length": 0, "word_count": 0, "line_count": 0,
        }

    # Sample first 50K chars for scoring (same as existing scores)
    sample = text[:50000]
    words = sample.split()
    lines = sample.splitlines()

    # Gibberish ratio: words with >50% non-alpha chars
    gibberish_count = 0
    for w in words:
        alpha = sum(1 for c in w if c.isalpha())
        if len(w) > 2 and alpha / len(w) < 0.5:
            gibberish_count += 1
    gibberish_ratio = gibberish_count / max(len(words), 1)

    # Encoding errors: replacement chars
    encoding_errors = sample.count("\ufffd") / max(len(sample), 1)

    # Char noise: unusual density of punctuation/symbols
    noise_chars = sum(1 for c in sample if c in "�|[]{}\\<>^~`")
    char_noise = noise_chars / max(len(sample), 1)

    # Line fragmentation: ratio of very short lines (<5 chars)
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 5)
    line_fragmentation = short_lines / max(len(lines), 1)

    # Whitespace ratio
    ws = sum(1 for c in sample if c in " \t\n\r")
    whitespace_ratio = ws / max(len(sample), 1)

    # Content density: words per line
    content_density = len(words) / max(len(lines), 1)

    # Overall score (0-100, higher = better)
    overall = 100.0
    overall -= gibberish_ratio * 200  # heavy penalty
    overall -= encoding_errors * 500
    overall -= char_noise * 300
    overall -= line_fragmentation * 100
    if whitespace_ratio > 0.3:
        overall -= (whitespace_ratio - 0.3) * 100
    overall = max(0, min(100, overall))

    return {
        "overall_score": round(overall),
        "gibberish_ratio": round(gibberish_ratio, 3),
        "encoding_errors": round(encoding_errors, 5),
        "char_noise": round(char_noise, 4),
        "line_fragmentation": round(line_fragmentation, 3),
        "whitespace_ratio": round(whitespace_ratio, 3),
        "repetition_score": 0.0,
        "digit_confusion": 0.0,
        "header_noise": 0.0,
        "content_density": round(content_density, 2),
        "text_length": len(sample),
        "word_count": len(words),
        "line_count": len(lines),
    }


def make_summary(text: str, max_len: int = 500) -> str:
    """Extract a clean summary from the first ~500 chars of readable text."""
    # Skip past any initial garbage/headers to find real content
    lines = text.splitlines()
    summary_lines = []
    char_count = 0
    for line in lines:
        stripped = line.strip()
        # Skip very short lines, page markers, bates numbers
        if len(stripped) < 10:
            continue
        if stripped.startswith("HOUSE_OVERSIGHT_") or stripped.startswith("EFTA"):
            continue
        summary_lines.append(stripped)
        char_count += len(stripped)
        if char_count >= max_len:
            break
    return " ".join(summary_lines)[:max_len]


def chunk_text(text: str, doc_id: str, chunk_size: int = 2000, overlap: int = 200) -> list[dict]:
    """Split text into overlapping chunks for search indexing."""
    chunks = []
    idx = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                sent_break = text.rfind(". ", start + chunk_size // 2, end)
                if sent_break > start:
                    end = sent_break + 1

        chunk_text_str = text[start:end].strip()
        if chunk_text_str and len(chunk_text_str) > 50:
            chunks.append({
                "document_id": doc_id,
                "chunk_index": idx,
                "chunk_text": chunk_text_str,
            })
            idx += 1
        start = end - overlap if end < len(text) else len(text)

    return chunks


def update_neon(doc_id: str, text: str, confidence: float, quality: dict) -> None:
    """Update all OCR-related tables in Neon DB."""
    import psycopg2

    url = get_neon_url()
    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1. Update ocr_text table
        logger.info("Updating ocr_text table...")
        cur.execute(
            """INSERT INTO ocr_text ("docId", text)
               VALUES (%s, %s)
               ON CONFLICT ("docId") DO UPDATE SET text = EXCLUDED.text""",
            (doc_id, text),
        )

        # 2. Update documents.summary with cleaner snippet
        summary = make_summary(text)
        if summary:
            logger.info("Updating documents.summary...")
            cur.execute(
                """UPDATE documents SET summary = %s WHERE id = %s""",
                (summary, doc_id),
            )

        # 3. Update ocr_quality_scores
        logger.info("Updating ocr_quality_scores...")
        cur.execute(
            """INSERT INTO ocr_quality_scores
               (doc_id, overall_score, gibberish_ratio, encoding_errors,
                char_noise, line_fragmentation, whitespace_ratio,
                repetition_score, digit_confusion, header_noise,
                content_density, text_length, word_count, line_count, scored_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
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
                scored_at = NOW()""",
            (
                doc_id, quality["overall_score"], quality["gibberish_ratio"],
                quality["encoding_errors"], quality["char_noise"],
                quality["line_fragmentation"], quality["whitespace_ratio"],
                quality["repetition_score"], quality["digit_confusion"],
                quality["header_noise"], quality["content_density"],
                quality["text_length"], quality["word_count"],
                quality["line_count"],
            ),
        )

        # 4. Replace document_chunks
        logger.info("Regenerating document_chunks...")
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        chunks = chunk_text(text, doc_id)
        if chunks:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """INSERT INTO document_chunks (document_id, chunk_index, chunk_text)
                   VALUES %s""",
                [(c["document_id"], c["chunk_index"], c["chunk_text"]) for c in chunks],
            )
            logger.info("Inserted %d chunks", len(chunks))

        conn.commit()
        logger.info("All Neon updates committed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Re-OCR a document and update Neon DB")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("--doc-id", required=True, help="Document ID in Neon (e.g. dc-26223474)")
    parser.add_argument(
        "--backend", default="paddleocr",
        choices=["paddleocr", "granite-docling", "smoldocling", "surya", "olmocr", "docling", "pymupdf"],
        help="OCR backend (default: paddleocr)",
    )
    parser.add_argument("--save-text", type=Path, default=None,
                        help="Save OCR text to file (default: next to PDF)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip Neon DB update (OCR only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run OCR but only print stats, don't update DB")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        logger.error("PDF not found: %s", args.pdf_path)
        sys.exit(1)

    # Run OCR
    text, confidence, page_confs = run_ocr(args.pdf_path, args.backend)

    if not text:
        logger.error("OCR produced no text!")
        sys.exit(1)

    # Save text locally
    save_path = args.save_text or args.pdf_path.with_suffix(".txt")
    save_path.write_text(text, encoding="utf-8")
    logger.info("Saved OCR text to %s (%.1f MB)", save_path, save_path.stat().st_size / 1024 / 1024)

    # Compute quality
    quality = compute_quality_scores(text)
    logger.info("Quality scores: overall=%d, gibberish=%.3f, encoding_errors=%.5f",
                quality["overall_score"], quality["gibberish_ratio"], quality["encoding_errors"])

    if args.dry_run:
        logger.info("DRY RUN - skipping DB update")
        logger.info("Text length: %d chars, %d words", len(text), len(text.split()))
        logger.info("Summary: %s", make_summary(text, 200))
        chunks = chunk_text(text, args.doc_id)
        logger.info("Would create %d chunks", len(chunks))
        return

    if not args.skip_db:
        update_neon(args.doc_id, text, confidence, quality)
    else:
        logger.info("Skipping DB update (--skip-db)")

    logger.info("Done! Document %s re-OCR complete.", args.doc_id)


if __name__ == "__main__":
    main()
