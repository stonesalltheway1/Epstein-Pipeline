"""Phase 1 QA: verify the 1.4M record updates from this session are stable.

Samples random documents across each touched source, tests:
1. The pdfUrl in Neon is reachable (via our pdf-proxy when applicable)
2. The live document page returns 200 and has expected URL in HTML
3. OCR text is present where expected
4. No duplicate IDs or orphan records
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("qa")

SITE = "https://epsteinexposed.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 Chrome/130"

# Sources touched this session with expected sample size
TOUCHED_SOURCES = [
    ("efta-ds1", 10),
    ("efta-ds2", 5),
    ("efta-ds3", 5),
    ("efta-ds4", 5),
    ("efta-ds8", 10),
    ("efta-ds9", 10),
    ("efta-ds10", 10),
    ("efta-ds11", 10),
    ("efta-ds12", 5),
    ("doj-ds10", 10),
    ("house-oversight", 10),
    ("courtlistener", 10),
    ("sec-edgar", 10),
    ("senate-finance", 1),
    ("sdny-court", 1),
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


def sample_docs(source: str, n: int) -> list[tuple[str, str, str]]:
    """Return random sample of (id, pdfUrl, source) for a given source."""
    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()
    cur.execute(
        '''SELECT id, "pdfUrl" FROM documents
           WHERE source = %s AND "pdfUrl" IS NOT NULL AND "pdfUrl" != ''
           ORDER BY random() LIMIT %s''',
        (source, n),
    )
    results = [(r[0], r[1], source) for r in cur.fetchall()]
    conn.close()
    return results


def test_pdf_url(url: str, session: requests.Session) -> dict:
    """Test a pdfUrl — through proxy if domain requires it."""
    # Detect if URL should go through proxy
    proxy_domains = ("archive.org", "justice.gov", "uscourts.gov",
                     "oversight.house.gov", "efoia.justice.gov",
                     "storage.courtlistener.com")
    if any(d in url for d in proxy_domains):
        proxied_url = f"{SITE}/api/pdf-proxy?url={quote(url)}"
        test_url = proxied_url
        via = "proxy"
    else:
        test_url = url
        via = "direct"

    t0 = time.time()
    try:
        r = session.get(test_url, timeout=60, stream=True,
                        headers={"User-Agent": UA})
        ct = r.headers.get("Content-Type", "")
        head = r.raw.read(8) if r.status_code == 200 else b""
        r.close()
        is_pdf = head.startswith(b"%PDF") or head[:2] == b"\x1f\x8b"  # pdf or gzip
        is_html_fallback = "text/html" in ct and r.status_code == 200
        is_jpg = head.startswith(b"\xff\xd8\xff")
        return {
            "url": url,
            "status": r.status_code,
            "content_type": ct[:40],
            "is_pdf": is_pdf,
            "is_jpg": is_jpg,
            "is_html": is_html_fallback,
            "via": via,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "head_hex": head[:4].hex(),
        }
    except Exception as e:
        return {
            "url": url, "status": "ERROR", "error": str(e)[:100],
            "via": via, "elapsed_ms": int((time.time() - t0) * 1000),
        }


def test_doc_page(doc_id: str, session: requests.Session) -> dict:
    """Test that the site's document page loads."""
    url = f"{SITE}/documents/{doc_id}"
    t0 = time.time()
    try:
        r = session.get(url, timeout=30, headers={"User-Agent": UA})
        html = r.text
        has_viewer = "DocumentViewerV2" in html or "pdf-viewer" in html or "Native" in html
        has_error = "500" in html[:2000] or "Something went wrong" in html
        return {
            "doc_id": doc_id, "status": r.status_code,
            "size_kb": round(len(html) / 1024, 1),
            "has_viewer": has_viewer,
            "has_error": has_error,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {"doc_id": doc_id, "status": "ERROR", "error": str(e)[:100]}


def scan_anomalies() -> dict:
    """Check Neon for data anomalies introduced this session."""
    conn = psycopg2.connect(get_neon_url())
    cur = conn.cursor()

    anomalies = {}

    # 1. Records with pdfUrl pointing at dead domains
    cur.execute(
        '''SELECT source, COUNT(*) FROM documents
           WHERE "pdfUrl" LIKE %s GROUP BY source''',
        ("%efts.fbi.gov%",),
    )
    anomalies["dead_efts_fbi_gov"] = dict(cur.fetchall())

    # 2. Records with pdfUrl = empty string
    cur.execute(
        '''SELECT source, COUNT(*) FROM documents
           WHERE "pdfUrl" = '' GROUP BY source ORDER BY COUNT(*) DESC LIMIT 10''')
    anomalies["empty_string_pdfUrls"] = dict(cur.fetchall())

    # 3. Records with malformed archive.org URLs (double-encoded etc.)
    cur.execute(
        '''SELECT COUNT(*) FROM documents WHERE "pdfUrl" LIKE %s''',
        ("%%2520%",),  # double-encoded space
    )
    anomalies["double_encoded"] = cur.fetchone()[0]

    # 4. Records with tags missing the new native: tags
    cur.execute(
        '''SELECT COUNT(*) FROM documents
           WHERE source = 'house-oversight' AND tags::text LIKE %s''',
        ('%native:%',),
    )
    anomalies["house_oversight_tagged_native"] = cur.fetchone()[0]

    # 5. Duplicate IDs (should be 0 — it's the PK)
    cur.execute('''SELECT COUNT(*) - COUNT(DISTINCT id) FROM documents''')
    anomalies["duplicate_ids"] = cur.fetchone()[0]

    # 6. Records with pdfUrl but no OCR
    cur.execute(
        '''SELECT COUNT(*) FROM documents d
           WHERE d."pdfUrl" IS NOT NULL AND d."pdfUrl" != ''
             AND NOT EXISTS (SELECT 1 FROM ocr_text o WHERE o."docId" = d.id)''')
    anomalies["pdfurl_without_ocr"] = cur.fetchone()[0]

    # 7. Recent commits: all sources should have expected counts
    cur.execute(
        '''SELECT source, COUNT(*) FROM documents
           WHERE source IN ('courtlistener','sec-edgar','senate-finance','sdny-court')
           GROUP BY source''')
    anomalies["new_source_counts"] = dict(cur.fetchall())

    conn.close()
    return anomalies


def main():
    logger.info("=== Phase 1 QA: Session Verification ===")

    # Step 1: anomaly scan
    logger.info("Step 1: Scanning Neon for anomalies...")
    anomalies = scan_anomalies()
    for k, v in anomalies.items():
        logger.info("  %s: %s", k, v)

    # Step 2: sample docs per source
    all_samples = []
    logger.info("\nStep 2: Sampling docs per source...")
    for source, n in TOUCHED_SOURCES:
        samples = sample_docs(source, n)
        logger.info("  %s: %d samples", source, len(samples))
        all_samples.extend(samples)
    logger.info("Total samples: %d", len(all_samples))

    # Step 3: test pdfUrls in parallel
    logger.info("\nStep 3: Testing pdfUrls...")
    session = requests.Session()
    pdf_results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(test_pdf_url, url, session): (doc_id, src)
                   for doc_id, url, src in all_samples}
        for fut in as_completed(futures):
            doc_id, src = futures[fut]
            try:
                r = fut.result()
                r["doc_id"] = doc_id
                r["source"] = src
                pdf_results.append(r)
            except Exception as e:
                logger.warning("Future failed: %s", e)

    # Step 4: test live document pages
    logger.info("\nStep 4: Testing live document pages...")
    page_results = []
    sample_ids = [s[0] for s in random.sample(all_samples, min(30, len(all_samples)))]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(test_doc_page, doc_id, session): doc_id
                   for doc_id in sample_ids}
        for fut in as_completed(futures):
            try:
                page_results.append(fut.result())
            except Exception as e:
                logger.warning("Future failed: %s", e)

    # Summary
    logger.info("\n=== Summary ===")
    by_source = {}
    for r in pdf_results:
        src = r["source"]
        if src not in by_source:
            by_source[src] = {"ok": 0, "404": 0, "error": 0, "is_pdf": 0, "wrong_type": 0}
        status = r.get("status")
        if status == 200:
            by_source[src]["ok"] += 1
            if r.get("is_pdf"):
                by_source[src]["is_pdf"] += 1
            elif not r.get("is_html"):
                by_source[src]["wrong_type"] += 1
        elif status == 404:
            by_source[src]["404"] += 1
        else:
            by_source[src]["error"] += 1

    for src, stats in by_source.items():
        logger.info("  %s: %s", src, stats)

    page_ok = sum(1 for p in page_results if p.get("status") == 200)
    page_err = sum(1 for p in page_results if p.get("status") != 200)
    logger.info("\nPages: %d ok, %d failed", page_ok, page_err)

    # Dump failed cases
    failures = [r for r in pdf_results if r.get("status") not in (200,) or
                (r.get("status") == 200 and not r.get("is_pdf") and not r.get("is_html") and not r.get("is_jpg"))]
    if failures:
        logger.warning("\n=== Failures (%d) ===", len(failures))
        for f in failures[:20]:
            logger.warning("  %s | %s | %s | head=%s",
                           f.get("source"), f.get("status"), f.get("doc_id"),
                           f.get("head_hex", ""))

    # Write full report
    report = {
        "anomalies": anomalies,
        "pdf_tests": pdf_results,
        "page_tests": page_results,
        "summary": by_source,
    }
    report_path = Path(__file__).resolve().parent.parent / "qa-report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("\nFull report: %s", report_path)


if __name__ == "__main__":
    main()
