"""Turbo DS10 image extraction with NTFS-optimized output.

Key optimizations:
1. Direct xref scan (no page.get_images() overhead)
2. Output to subdirectories (avoids NTFS large-dir penalty)
3. High chunksize to minimize IPC overhead
4. Completed log for fast resumability

Usage:
    python scripts/extract-ds10-turbo.py --workers 16
    python scripts/extract-ds10-turbo.py --workers 16 --limit 5000
    python scripts/extract-ds10-turbo.py --workers 16 --bench  # compare to fresh dir
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INPUT_DIR = Path(r"E:\epstein-ds10\extracted\VOL00010\IMAGES")
OUTPUT_DIR = Path(r"E:\epstein-ds10\images2")  # New subdir-based output
COMPLETED_LOG = Path(r"E:\epstein-ds10\completed2.txt")

MIN_SIZE = 5000  # 5KB


def get_subdir(efta_id: str) -> str:
    """Group into subdirs of ~1000 files: EFTA0126/EFTA0127/etc."""
    # EFTA01262782 -> first 8 chars = EFTA0126
    return efta_id[:8] if len(efta_id) >= 8 else efta_id[:4]


def extract_one(pdf_path_str: str) -> tuple[str, int, str]:
    """Extract images via direct xref scan."""
    import fitz

    pdf_path = Path(pdf_path_str)
    efta_id = pdf_path.stem
    subdir = OUTPUT_DIR / get_subdir(efta_id)

    try:
        doc = fitz.open(pdf_path_str)
        count = 0
        img_idx = 0

        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_get_key(xref, "Subtype")[1] != "/Image":
                    continue

                img = doc.extract_image(xref)
                if not img:
                    continue

                data = img["image"]
                w = img.get("width", 0)
                h = img.get("height", 0)

                if w < 100 or h < 100 or len(data) < MIN_SIZE:
                    continue

                ext = img.get("ext", "png")
                filename = f"{efta_id}_p{img_idx}_i0.{ext}"

                # Ensure subdir exists (cheap after first call)
                subdir.mkdir(parents=True, exist_ok=True)
                (subdir / filename).write_bytes(data)
                count += 1
                img_idx += 1

            except Exception:
                continue

        doc.close()
        return (efta_id, count, "")

    except Exception as e:
        return (efta_id, 0, str(e)[:100])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bench", action="store_true", help="Benchmark mode (fresh output dir)")
    args = parser.parse_args()

    if args.bench:
        import shutil
        global OUTPUT_DIR, COMPLETED_LOG
        OUTPUT_DIR = Path(r"E:\epstein-ds10\_bench_output")
        COMPLETED_LOG = Path(r"E:\epstein-ds10\_bench_completed.txt")
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        if COMPLETED_LOG.exists():
            COMPLETED_LOG.unlink()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning for PDFs...", flush=True)
    pdfs = []
    for subdir_entry in sorted(os.scandir(INPUT_DIR), key=lambda e: e.name):
        if not subdir_entry.is_dir():
            continue
        for entry in os.scandir(subdir_entry.path):
            if entry.name.endswith(".pdf"):
                pdfs.append(entry.path)
    print(f"Found {len(pdfs):,} PDFs", flush=True)

    # Fast skip — check both old and new completed logs
    completed = set()
    old_log = Path(r"E:\epstein-ds10\completed.txt")
    if old_log.exists():
        completed.update(old_log.read_text().splitlines())
        print(f"Loaded {len(completed):,} from old completed log", flush=True)
    if COMPLETED_LOG.exists():
        completed.update(COMPLETED_LOG.read_text().splitlines())
        print(f"Total completed (old+new): {len(completed):,}", flush=True)

    before = len(pdfs)
    pdfs = [p for p in pdfs if Path(p).stem not in completed]
    if before - len(pdfs) > 0:
        print(f"Skipping {before - len(pdfs):,} already done", flush=True)

    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print("Nothing to extract!", flush=True)
        return

    print(f"Extracting from {len(pdfs):,} PDFs with {args.workers} workers...", flush=True)

    total_images = 0
    total_errors = 0
    start = time.time()
    newly_completed = []
    chunksize = max(1, min(200, len(pdfs) // (args.workers * 2)))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (efta_id, count, error) in enumerate(pool.map(extract_one, pdfs, chunksize=chunksize)):
            total_images += count
            if error:
                total_errors += 1
            else:
                newly_completed.append(efta_id)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(pdfs) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:,}/{len(pdfs):,} | {total_images:,} imgs | {rate:.0f} PDFs/s | ~{remaining/60:.0f}m left", flush=True)

                if newly_completed:
                    with open(COMPLETED_LOG, "a") as f:
                        f.write("\n".join(newly_completed) + "\n")
                    newly_completed = []

    if newly_completed:
        with open(COMPLETED_LOG, "a") as f:
            f.write("\n".join(newly_completed) + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Processed: {len(pdfs):,}", flush=True)
    print(f"  Extracted: {total_images:,} images", flush=True)
    print(f"  Errors: {total_errors:,}", flush=True)
    print(f"  Rate: {len(pdfs)/elapsed:.1f} PDFs/s", flush=True)

    if args.bench:
        import shutil
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        COMPLETED_LOG.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
