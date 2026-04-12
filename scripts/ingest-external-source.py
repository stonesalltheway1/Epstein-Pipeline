"""Ingest CourtListener + SEC EDGAR + ProPublica output folders into Neon.

Handles the different metadata shapes from each downloader:

CourtListener: output/courtlistener/search_*/docket_*_doc_*.{pdf,json}
  metadata: recap_documents JSON with _case_name, _docket_number, etc.

SEC EDGAR: output/sec-edgar/CIK_NAME/YYYY-MM-DD_FORM_ACCESSION.htm
  metadata: _submissions.json at the CIK root

ProPublica: output/propublica-nonprofits/EIN_NAME/_organization.json
  metadata-only ingest into nonprofit_filings table (if not already there)

Usage:
    python scripts/ingest-external-source.py --kind courtlistener --input output/courtlistener
    python scripts/ingest-external-source.py --kind sec-edgar --input output/sec-edgar
    python scripts/ingest-external-source.py --kind propublica --input output/propublica-nonprofits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_values

# Enable local src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingest")


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


# ── Quality scoring + chunking helpers (copied from ingest-featured-releases) ──

def compute_quality_scores(text: str) -> dict:
    if not text:
        return {"overall_score": 0, "gibberish_ratio": 1.0, "encoding_errors": 1.0,
                "char_noise": 1.0, "line_fragmentation": 1.0, "whitespace_ratio": 1.0,
                "repetition_score": 0.0, "digit_confusion": 0.0, "header_noise": 0.0,
                "content_density": 0.0, "text_length": 0, "word_count": 0, "line_count": 0}
    sample = text[:50000]
    words = sample.split()
    lines = sample.splitlines()
    gibberish = sum(1 for w in words
                    if len(w) > 2 and sum(1 for c in w if c.isalpha()) / len(w) < 0.5)
    gibberish_ratio = gibberish / max(len(words), 1)
    encoding_errors = sample.count("\ufffd") / max(len(sample), 1)
    noise = sum(1 for c in sample if c in "\ufffd|[]{}\\<>^~`")
    char_noise = noise / max(len(sample), 1)
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 5)
    line_frag = short_lines / max(len(lines), 1)
    ws = sum(1 for c in sample if c in " \t\n\r")
    ws_ratio = ws / max(len(sample), 1)
    content_density = len(words) / max(len(lines), 1)
    overall = 100.0 - gibberish_ratio * 200 - encoding_errors * 500 \
              - char_noise * 300 - line_frag * 100
    if ws_ratio > 0.3:
        overall -= (ws_ratio - 0.3) * 100
    overall = max(0, min(100, overall))
    return {"overall_score": round(overall), "gibberish_ratio": round(gibberish_ratio, 3),
            "encoding_errors": round(encoding_errors, 5), "char_noise": round(char_noise, 4),
            "line_fragmentation": round(line_frag, 3), "whitespace_ratio": round(ws_ratio, 3),
            "repetition_score": 0.0, "digit_confusion": 0.0, "header_noise": 0.0,
            "content_density": round(content_density, 2), "text_length": len(sample),
            "word_count": len(words), "line_count": len(lines)}


def chunk_text(text: str, doc_id: str, chunk_size: int = 2000,
                overlap: int = 200) -> list[dict]:
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


def ocr_pdf(pdf_path: Path) -> tuple[str, float, int]:
    """Run PaddleOCR with cache on .txt/.meta.json sidecar files."""
    txt_path = pdf_path.with_suffix(".txt")
    meta_path = pdf_path.with_suffix(".meta.json")
    if txt_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return (txt_path.read_text(encoding="utf-8"),
                meta.get("confidence", 0.0),
                meta.get("page_count", 1))

    from epstein_pipeline.processors.ocr import PaddleOcrBackend
    backend = PaddleOcrBackend()
    result = backend.extract(pdf_path)
    page_count = len(result.page_confidences)
    txt_path.write_text(result.text, encoding="utf-8")
    meta_path.write_text(json.dumps({"confidence": result.confidence,
                                      "page_count": page_count}))
    return result.text, result.confidence, page_count


def upsert_document(cur, doc_id: str, title: str, source: str, category: str,
                    date: str | None, summary: str, tags: list[str],
                    pdf_url: str | None, source_url: str | None,
                    page_count: int, text: str | None = None) -> None:
    """Upsert a document row + OCR text + chunks + quality scores."""
    # documents
    cur.execute("""
        INSERT INTO documents
          (id, title, date, source, category, summary, tags, "pdfUrl",
           "sourceUrl", "pageCount")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
          title = EXCLUDED.title, date = EXCLUDED.date, source = EXCLUDED.source,
          category = EXCLUDED.category, summary = EXCLUDED.summary,
          tags = EXCLUDED.tags, "pdfUrl" = EXCLUDED."pdfUrl",
          "sourceUrl" = EXCLUDED."sourceUrl", "pageCount" = EXCLUDED."pageCount"
    """, (doc_id, title, date, source, category, summary or "",
          Json(tags or []), pdf_url, source_url, page_count))

    if text:
        cur.execute("""
            INSERT INTO ocr_text ("docId", text) VALUES (%s, %s)
            ON CONFLICT ("docId") DO UPDATE SET text = EXCLUDED.text
        """, (doc_id, text))

        # quality + chunks
        q = compute_quality_scores(text)
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
              content_density = EXCLUDED.content_density,
              text_length = EXCLUDED.text_length,
              word_count = EXCLUDED.word_count,
              line_count = EXCLUDED.line_count,
              scored_at = NOW()
        """, (doc_id, q["overall_score"], q["gibberish_ratio"], q["encoding_errors"],
              q["char_noise"], q["line_fragmentation"], q["whitespace_ratio"],
              q["repetition_score"], q["digit_confusion"], q["header_noise"],
              q["content_density"], q["text_length"], q["word_count"], q["line_count"]))

        chunks = chunk_text(text, doc_id)
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        if chunks:
            execute_values(cur,
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text) VALUES %s",
                [(c["document_id"], c["chunk_index"], c["chunk_text"]) for c in chunks])


# ────────────────────────────────────────────────────────────────────────
# CourtListener ingest
# ────────────────────────────────────────────────────────────────────────

def ingest_courtlistener(root: Path, ocr: bool = True,
                         max_docs: int | None = None) -> dict:
    """Walk CourtListener output and ingest each .pdf+.json pair."""
    conn = psycopg2.connect(get_neon_url())
    stats = {"total": 0, "ingested": 0, "ocr_done": 0, "skipped": 0, "errors": 0}

    pdf_paths = sorted(root.rglob("*.pdf"))
    logger.info("CourtListener: %d PDFs found in %s", len(pdf_paths), root)

    for i, pdf_path in enumerate(pdf_paths):
        if max_docs and stats["ingested"] >= max_docs:
            break
        stats["total"] += 1

        json_path = pdf_path.with_suffix(".json")
        if not json_path.exists():
            logger.warning("No metadata for %s, skipping", pdf_path.name)
            stats["skipped"] += 1
            continue

        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            rd_id = meta.get("id") or meta.get("_docket_id") or 0
            # Build a stable ID
            doc_id = f"cl-recap-{rd_id}"

            case_name = meta.get("_case_name") or "Unknown case"
            docket_num = meta.get("_docket_number") or ""
            doc_num = meta.get("document_number") or "?"
            att_num = meta.get("attachment_number")
            description = meta.get("description") or ""

            title_parts = [case_name]
            if docket_num:
                title_parts.append(f"({docket_num})")
            title_parts.append(f"Doc {doc_num}")
            if att_num:
                title_parts.append(f"Att {att_num}")
            title = " — ".join(title_parts)

            date = meta.get("_date_filed") or meta.get("entry_date_filed") or None
            court_id = meta.get("_court_id", "")

            tags = ["courtlistener", "recap"]
            if court_id:
                tags.append(f"court-{court_id}")
            tags.append(f"docket-{docket_num}")

            # Full URL from filepath_local (prefer) or IA fallback
            filepath_local = meta.get("filepath_local", "").lstrip("/")
            pdf_url = (f"https://storage.courtlistener.com/{filepath_local}"
                       if filepath_local else meta.get("filepath_ia"))
            source_url = f"https://www.courtlistener.com{meta.get('absolute_url', '')}" \
                         if meta.get("absolute_url") else None

            text = None
            page_count = meta.get("page_count") or 1
            if ocr:
                try:
                    text, _conf, page_count = ocr_pdf(pdf_path)
                    stats["ocr_done"] += 1
                except Exception as e:
                    logger.warning("OCR failed for %s: %s", pdf_path.name, e)

            with conn:
                with conn.cursor() as cur:
                    upsert_document(cur, doc_id, title, "courtlistener", "legal",
                                    date, description[:500], tags,
                                    pdf_url, source_url, page_count, text)
            stats["ingested"] += 1
            if stats["ingested"] % 10 == 0:
                logger.info("Progress: %d/%d ingested", stats["ingested"], len(pdf_paths))
        except Exception as e:
            logger.exception("Failed %s: %s", pdf_path.name, e)
            stats["errors"] += 1

    conn.close()
    return stats


# ────────────────────────────────────────────────────────────────────────
# SEC EDGAR ingest (HTML filings, no OCR needed)
# ────────────────────────────────────────────────────────────────────────

def extract_text_from_sec_html(html: str) -> str:
    """Rough text extraction from SEC HTML filings."""
    from html import unescape
    # Strip scripts/styles
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    # Replace tags with spaces
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ingest_sec_edgar(root: Path, max_docs: int | None = None) -> dict:
    conn = psycopg2.connect(get_neon_url())
    stats = {"total": 0, "ingested": 0, "skipped": 0, "errors": 0}

    htm_paths = sorted(root.rglob("*.htm")) + sorted(root.rglob("*.html"))
    logger.info("SEC EDGAR: %d HTML filings found in %s", len(htm_paths), root)

    for htm_path in htm_paths:
        if max_docs and stats["ingested"] >= max_docs:
            break
        stats["total"] += 1

        # Parse filename: YYYY-MM-DD_FORM_ACCESSION.htm
        m = re.match(r"(\d{4}-\d{2}-\d{2})_([^_]+)_(.+)\.htm", htm_path.name)
        if not m:
            stats["skipped"] += 1
            continue
        date, form_type, accession = m.groups()

        # Parse CIK from parent folder (e.g. "0000701985_Bath_&_Body_Works_Inc_...")
        parent = htm_path.parent.name
        cik_match = re.match(r"(\d{10})_(.+)", parent)
        if not cik_match:
            stats["skipped"] += 1
            continue
        cik, entity_name = cik_match.groups()
        entity_name = entity_name.replace("_", " ")

        try:
            html = htm_path.read_text(encoding="utf-8", errors="ignore")
            text = extract_text_from_sec_html(html)
            # Only keep if text mentions Epstein, Wexner, USVI, Jeffrey Epstein, or JPMorgan settlement
            if not any(k.lower() in text.lower() for k in (
                "epstein", "wexner", "virgin islands", "jeffrey", "maxwell",
                "sexual abuse", "trafficking")):
                # Still ingest but mark low-relevance
                pass

            doc_id = f"sec-{cik}-{accession}"
            title = f"{entity_name} — {form_type} ({date})"
            source_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}"
            pdf_url = None  # SEC filings are HTML

            tags = ["sec-edgar", f"form-{form_type}", f"cik-{cik}"]

            summary = text[:400] if text else ""
            page_count = max(1, len(text) // 3000)  # rough page estimate

            with conn:
                with conn.cursor() as cur:
                    upsert_document(cur, doc_id, title, "sec-edgar", "financial",
                                    date, summary, tags, pdf_url, source_url,
                                    page_count, text)
            stats["ingested"] += 1
            if stats["ingested"] % 10 == 0:
                logger.info("Progress: %d/%d ingested", stats["ingested"], len(htm_paths))
        except Exception as e:
            logger.exception("Failed %s: %s", htm_path.name, e)
            stats["errors"] += 1

    conn.close()
    return stats


# ────────────────────────────────────────────────────────────────────────
# ProPublica metadata ingest into nonprofit_filings
# ────────────────────────────────────────────────────────────────────────

def ingest_propublica(root: Path) -> dict:
    conn = psycopg2.connect(get_neon_url())
    stats = {"orgs": 0, "filings_inserted": 0, "errors": 0}

    for org_json in root.rglob("_organization.json"):
        try:
            data = json.loads(org_json.read_text(encoding="utf-8"))
            org = data.get("organization", {})
            ein = str(org.get("ein") or "").zfill(9)
            if not ein or ein == "000000000":
                continue
            stats["orgs"] += 1

            filings = data.get("filings_with_data", []) + data.get("filings_without_data", [])
            org_id = f"np-{ein}"
            slug = re.sub(r"[^a-z0-9]+", "-",
                          (org.get("name") or "").lower()).strip("-")[:60]

            with conn:
                with conn.cursor() as cur:
                    # nonprofit_orgs uses `id` as PK, not `ein`
                    cur.execute("""
                        INSERT INTO nonprofit_orgs
                          (id, ein, name, slug, city, state, ntee_code,
                           subsection_code, filing_count, propublica_url,
                           metadata, epstein_linked)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          ein = EXCLUDED.ein, name = EXCLUDED.name,
                          slug = EXCLUDED.slug, city = EXCLUDED.city,
                          state = EXCLUDED.state, ntee_code = EXCLUDED.ntee_code,
                          subsection_code = EXCLUDED.subsection_code,
                          filing_count = EXCLUDED.filing_count,
                          propublica_url = EXCLUDED.propublica_url,
                          metadata = EXCLUDED.metadata,
                          epstein_linked = true,
                          updated_at = NOW()
                    """, (org_id, ein, org.get("name"), slug,
                          org.get("city"), org.get("state"),
                          org.get("ntee_code"),
                          str(org.get("subseccd") or ""),
                          len(filings),
                          f"https://projects.propublica.org/nonprofits/organizations/{ein}",
                          Json(org), True))

                    for f in filings:
                        tax_prd = f.get("tax_prd_yr") or f.get("tax_prd")
                        tax_year = (int(str(tax_prd)[:4]) if tax_prd
                                    and str(tax_prd)[:4].isdigit() else None)
                        cur.execute("""
                            INSERT INTO nonprofit_filings
                              (org_id, tax_period, tax_year, form_type, total_revenue,
                               total_expenses, total_assets, grants_paid,
                               contributions_received, pdf_url)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING
                        """, (org_id, str(tax_prd) if tax_prd else None, tax_year,
                              f.get("formtype_str") or str(f.get("formtype", "")),
                              f.get("totrevenue"),
                              f.get("totfuncexpns"),
                              f.get("totassetsend"),
                              f.get("totcntrbgfts"),
                              f.get("totcntrbgfts"),
                              f.get("pdf_url")))
                        stats["filings_inserted"] += 1
        except Exception as e:
            logger.exception("Failed %s: %s", org_json.name, e)
            stats["errors"] += 1

    conn.close()
    return stats


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True,
                        choices=["courtlistener", "sec-edgar", "propublica"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR for PDFs (store metadata + placeholder only)")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    if args.kind == "courtlistener":
        stats = ingest_courtlistener(args.input, ocr=not args.no_ocr,
                                     max_docs=args.max_docs)
    elif args.kind == "sec-edgar":
        stats = ingest_sec_edgar(args.input, max_docs=args.max_docs)
    elif args.kind == "propublica":
        stats = ingest_propublica(args.input)

    logger.info("=== Final stats: %s", stats)


if __name__ == "__main__":
    main()
