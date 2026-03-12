"""Fast DS10 image extraction using Poppler pdfimages + parallel subprocesses.

Uses pdfimages -png for native extraction, writes to per-PDF temp dirs
to avoid NTFS large-directory glob penalty, then moves to flat output.

Usage:
    python scripts/extract-ds10-fast.py --workers 16
    python scripts/extract-ds10-fast.py --workers 24 --limit 10000
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PDFIMAGES = r"C:\tools\poppler\poppler-24.08.0\Library\bin\pdfimages.exe"
INPUT_DIR = Path(r"E:\epstein-ds10\extracted\VOL00010\IMAGES")
OUTPUT_DIR = Path(r"E:\epstein-ds10\images")
TEMP_DIR = Path(r"E:\epstein-ds10\_tmp_extract")
COMPLETED_LOG = Path(r"E:\epstein-ds10\completed.txt")

MIN_SIZE = 5000  # 5KB


def extract_one(pdf_path_str: str) -> tuple[str, int, str]:
    """Extract images from one PDF using pdfimages into a temp subdir."""
    pdf_path = Path(pdf_path_str)
    efta_id = pdf_path.stem

    # Write to isolated temp subdir (avoids NTFS large-dir penalty)
    tmp_sub = TEMP_DIR / efta_id
    tmp_sub.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [PDFIMAGES, "-png", pdf_path_str, str(tmp_sub / efta_id)],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            shutil.rmtree(tmp_sub, ignore_errors=True)
            return (efta_id, 0, result.stderr.decode("utf-8", errors="replace")[:100])

        # Process files in the tiny temp dir (fast - only this PDF's files)
        count = 0
        for entry in os.scandir(tmp_sub):
            if entry.stat().st_size < MIN_SIZE:
                os.unlink(entry.path)
                continue

            # Rename: EFTA...-000.png -> EFTA..._p0_i0.png
            name = entry.name
            stem = Path(name).stem
            ext = Path(name).suffix
            parts = stem.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                new_name = f"{efta_id}_p{int(parts[1])}_i0{ext}"
            else:
                new_name = name

            dest = OUTPUT_DIR / new_name
            os.rename(entry.path, str(dest))
            count += 1

        # Clean up temp subdir
        shutil.rmtree(tmp_sub, ignore_errors=True)
        return (efta_id, count, "")

    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_sub, ignore_errors=True)
        return (efta_id, 0, "timeout")
    except Exception as e:
        shutil.rmtree(tmp_sub, ignore_errors=True)
        return (efta_id, 0, str(e)[:100])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(24, cpu_count()), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(PDFIMAGES).exists():
        print(f"ERROR: pdfimages not found at {PDFIMAGES}", flush=True)
        sys.exit(1)

    # Scan for PDFs
    print("Scanning for PDFs...", flush=True)
    pdfs = []
    for subdir_entry in sorted(os.scandir(INPUT_DIR), key=lambda e: e.name):
        if not subdir_entry.is_dir():
            continue
        for entry in os.scandir(subdir_entry.path):
            if entry.name.endswith(".pdf"):
                pdfs.append(entry.path)
    print(f"Found {len(pdfs):,} PDFs", flush=True)

    # Fast skip using completed log (avoids scanning 300K+ output dir)
    completed = set()
    if COMPLETED_LOG.exists():
        completed = set(COMPLETED_LOG.read_text().splitlines())
        print(f"Loaded {len(completed):,} completed IDs from log", flush=True)
    else:
        # First run or log missing: rebuild from output dir
        print("Building skip set from output files (one-time)...", flush=True)
        seen = set()
        for entry in os.scandir(OUTPUT_DIR):
            parts = entry.name.split("_p")
            if parts:
                seen.add(parts[0])
        completed = seen
        if completed:
            COMPLETED_LOG.write_text("\n".join(sorted(completed)))
            print(f"Found {len(completed):,} already-extracted PDFs", flush=True)

    before = len(pdfs)
    pdfs = [p for p in pdfs if Path(p).stem not in completed]
    skipped = before - len(pdfs)
    if skipped:
        print(f"Skipping {skipped:,} already-extracted", flush=True)

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
    chunksize = max(1, min(50, len(pdfs) // (args.workers * 4)))

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

                # Periodic flush to completed log
                if newly_completed:
                    with open(COMPLETED_LOG, "a") as f:
                        f.write("\n".join(newly_completed) + "\n")
                    newly_completed = []

    # Final flush
    if newly_completed:
        with open(COMPLETED_LOG, "a") as f:
            f.write("\n".join(newly_completed) + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Processed: {len(pdfs):,} PDFs", flush=True)
    print(f"  Extracted: {total_images:,} images", flush=True)
    print(f"  Errors: {total_errors:,}", flush=True)
    print(f"  Rate: {len(pdfs)/elapsed:.1f} PDFs/s", flush=True)

    # Cleanup temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
