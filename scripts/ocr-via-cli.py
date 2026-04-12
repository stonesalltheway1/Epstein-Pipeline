"""Run PaddleOCR via its CLI (not Python API) to avoid silent-exit bug.

PaddleOCR 3.4.0 + PaddlePaddle 3.2.0 on Windows has a silent-exit issue
when called via the Python API from a long-lived process (native oneDNN
layer calls std::abort() after first OCR). The CLI runs in an isolated
subprocess and doesn't hit this bug.

This script takes a list of PDFs, invokes the CLI on each, aggregates
the per-page JSON outputs into a single text file, and writes the
sidecar .txt + .meta.json files that ingest-featured-releases.py
expects when resuming with OCR cache.

Usage:
    python scripts/ocr-via-cli.py path/to/a.pdf path/to/b.pdf ...
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ocr-cli")


def ocr_pdf_via_cli(pdf_path: Path) -> tuple[str, float, int]:
    """Run paddleocr CLI on a PDF; return (full_text, avg_confidence, page_count)."""
    out_dir = pdf_path.parent / f"_{pdf_path.stem}_ocr_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running paddleocr CLI on %s ...", pdf_path.name)
    result = subprocess.run(
        [
            "paddleocr", "ocr",
            "-i", str(pdf_path),
            "--save_path", str(out_dir),
            "--use_doc_orientation_classify", "false",
            "--use_doc_unwarping", "false",
            "--lang", "en",
        ],
        env={
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONIOENCODING": "utf-8",
            "PATH": __import__("os").environ.get("PATH", ""),
            "SystemRoot": __import__("os").environ.get("SystemRoot", ""),
            "TEMP": __import__("os").environ.get("TEMP", ""),
            "USERPROFILE": __import__("os").environ.get("USERPROFILE", ""),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=3600,
    )
    if result.returncode != 0:
        logger.error("CLI failed (code %d): %s", result.returncode, result.stderr[-500:])
        raise RuntimeError(f"paddleocr CLI failed: {result.stderr[-200:]}")

    # Read per-page JSON results
    json_files = sorted(out_dir.glob(f"{pdf_path.stem}_*_res.json"))
    if not json_files:
        raise RuntimeError(f"No JSON results found in {out_dir}")

    all_text = []
    all_scores = []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        # CLI wraps in 'res' key sometimes; handle both
        res = data.get("res", data)
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        page_text = "\n".join(texts)
        all_text.append(page_text)
        all_scores.extend([float(s) for s in scores])

    full_text = "\n\n".join(all_text)
    avg_conf = sum(all_scores) / max(len(all_scores), 1)
    page_count = len(json_files)

    logger.info("  OCR'd %d pages, %d chars, %.3f avg confidence",
                page_count, len(full_text), avg_conf)
    return full_text, avg_conf, page_count


def process_pdf(pdf_path: Path) -> dict:
    """OCR a PDF and write sidecar cache files (.txt, .meta.json)."""
    txt_path = pdf_path.with_suffix(".txt")
    meta_path = pdf_path.with_suffix(".meta.json")

    if txt_path.exists() and meta_path.exists():
        logger.info("Already cached: %s", pdf_path.name)
        return {"status": "cached", "pdf": str(pdf_path)}

    try:
        text, conf, pages = ocr_pdf_via_cli(pdf_path)
    except Exception as e:
        logger.exception("Failed %s: %s", pdf_path.name, e)
        return {"status": "error", "pdf": str(pdf_path), "error": str(e)}

    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps({"confidence": conf, "page_count": pages}))
    logger.info("Wrote %s (%d chars)", txt_path.name, len(text))
    return {"status": "ok", "pdf": str(pdf_path), "chars": len(text), "pages": pages}


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr-via-cli.py <pdf> [<pdf> ...]")
        sys.exit(1)

    results = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            logger.error("Not found: %s", p)
            continue
        results.append(process_pdf(p))

    logger.info("=== Summary ===")
    for r in results:
        logger.info("  %s", r)


if __name__ == "__main__":
    main()
