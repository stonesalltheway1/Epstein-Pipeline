"""Fast batch extraction of images from DS10 EFTA PDFs.

Extracts embedded images from all PDFs in IMAGES/ subdirectories.
Uses multiprocessing for speed. Skips already-extracted PDFs.

Usage:
    python scripts/extract-ds10-images.py --input E:/epstein-ds10/extracted/VOL00010/IMAGES --output E:/epstein-ds10/images --workers 8
"""

import os
import sys
import time
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Minimum dimensions to keep (skip tiny artifacts)
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_SIZE = 5000  # 5KB


def extract_one_pdf(args):
    """Extract images from a single PDF. Returns (efta_id, count, error)."""
    pdf_path, output_dir = args
    efta_id = Path(pdf_path).stem

    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)

            for img_idx, img in enumerate(images):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    if not base:
                        continue

                    w = base.get("width", 0)
                    h = base.get("height", 0)
                    data = base["image"]

                    if w < MIN_WIDTH or h < MIN_HEIGHT or len(data) < MIN_SIZE:
                        continue

                    ext = base.get("ext", "png")
                    filename = f"{efta_id}_p{page_num}_i{img_idx}.{ext}"
                    out_path = Path(output_dir) / filename
                    out_path.write_bytes(data)
                    count += 1
                except Exception:
                    continue

            # If no embedded images found on first page, render as pixmap
            if count == 0 and page_num == 0 and len(images) == 0:
                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    if pix.width >= MIN_WIDTH and pix.height >= MIN_HEIGHT:
                        filename = f"{efta_id}_p{page_num}_render.png"
                        pix.save(str(Path(output_dir) / filename))
                        count += 1
                except Exception:
                    pass

        doc.close()
        return (efta_id, count, None)

    except Exception as e:
        return (efta_id, 0, str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from DS10 PDFs")
    parser.add_argument("--input", type=Path, required=True, help="IMAGES directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for extracted images")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit PDFs (for testing)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    print(f"Scanning {args.input} for PDFs...")
    pdfs = []
    for subdir in sorted(args.input.iterdir()):
        if not subdir.is_dir():
            continue
        for pdf in sorted(subdir.glob("*.pdf")):
            pdfs.append(pdf)

    print(f"Found {len(pdfs):,} PDFs across {len(list(args.input.iterdir()))} subdirectories")

    # Skip already-extracted
    if args.skip_existing:
        existing = set()
        for f in args.output.iterdir():
            # Extract EFTA ID from filename like EFTA01262782_p0_i0.png
            name = f.stem
            parts = name.split("_p")
            if parts:
                existing.add(parts[0])

        before = len(pdfs)
        pdfs = [p for p in pdfs if p.stem not in existing]
        skipped = before - len(pdfs)
        if skipped:
            print(f"Skipping {skipped:,} already-extracted PDFs")

    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print("Nothing to extract!")
        return

    print(f"Extracting images from {len(pdfs):,} PDFs using {args.workers} workers...")

    # Prepare work items
    work = [(str(pdf), str(args.output)) for pdf in pdfs]

    total_images = 0
    total_errors = 0
    start = time.time()

    with Pool(args.workers) as pool:
        for i, (efta_id, count, error) in enumerate(pool.imap_unordered(extract_one_pdf, work, chunksize=50)):
            total_images += count
            if error:
                total_errors += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(pdfs) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:,}/{len(pdfs):,} PDFs | {total_images:,} images | {rate:.0f} PDFs/s | ~{remaining/60:.0f}m remaining")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  PDFs processed: {len(pdfs):,}")
    print(f"  Images extracted: {total_images:,}")
    print(f"  Errors: {total_errors:,}")
    print(f"  Output: {args.output}")

    # Count output files
    output_files = list(args.output.glob("*.*"))
    print(f"  Files on disk: {len(output_files):,}")


if __name__ == "__main__":
    main()
