"""Fast DS10 media export — reads PNG headers directly, no PIL required.

Scans extracted images, reads PNG IHDR chunk for dimensions (24 bytes),
and generates media JSON compatible with the site's media gallery.

~10x faster than PIL-based export for large batches.

Usage:
    python scripts/export-ds10-media-fast.py
"""

import json
import os
import struct
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

IMAGES_DIR = Path("E:/epstein-ds10/images")
SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
OUTPUT_PATH = SITE_DIR / "data" / "ds10-media.json"
R2_PROXY = "https://media.epsteinexposed.com"

MIN_SIZE = 10_000  # 10KB minimum
DOC_RATIO = 1.6    # height/width > 1.6 = document page


def read_png_dimensions(filepath: Path) -> tuple[int, int]:
    """Read width/height from PNG IHDR chunk (first 24 bytes)."""
    try:
        with open(filepath, "rb") as f:
            sig = f.read(8)
            if sig[:4] != b'\x89PNG':
                return 0, 0
            # IHDR chunk: 4 bytes length + 4 bytes "IHDR" + 4 bytes width + 4 bytes height
            f.read(4)  # chunk length
            chunk_type = f.read(4)
            if chunk_type != b'IHDR':
                return 0, 0
            w, h = struct.unpack(">II", f.read(8))
            return w, h
    except Exception:
        return 0, 0


def process_file(img_path: Path, idx: int) -> dict | None:
    """Process a single image file into a media item dict."""
    efta_id = img_path.stem.split("_p")[0]

    # Get dimensions from PNG header (fast, no PIL)
    w, h = read_png_dimensions(img_path)
    if w == 0 and h == 0:
        return None

    ratio = h / w if w > 0 else 1
    img_type = "document" if ratio > DOC_RATIO else "photo"

    thumb_url = f"{R2_PROXY}/ds10-images/{img_path.name}"

    return {
        "id": f"m-{10000 + idx}",
        "title": f"{efta_id} - Evidence Photo" if img_type == "photo" else f"{efta_id} - Document Page",
        "type": img_type,
        "description": "",
        "date": "",
        "source": "DOJ EFTA Data Set 10",
        "sourceUrl": "https://www.justice.gov/epstein/doj-disclosures",
        "thumbnailUrl": thumb_url,
        "fullUrl": thumb_url,
        "personIds": [],
        "locationIds": [],
        "documentIds": [efta_id] if efta_id.startswith("EFTA") else [],
        "tags": ["doj", "data-set-10", "fbi-evidence"],
        "verified": True,
        "width": w,
        "height": h,
    }


def main():
    print(f"Scanning {IMAGES_DIR}...")
    start = time.time()

    files = sorted(IMAGES_DIR.glob("*.*"))
    print(f"Found {len(files):,} files ({time.time() - start:.1f}s)")

    # Filter by size
    print("Filtering by size...")
    files = [f for f in files if f.stat().st_size >= MIN_SIZE]
    print(f"After filter (>={MIN_SIZE // 1000}KB): {len(files):,} ({time.time() - start:.1f}s)")

    # Process files with thread pool (I/O bound, threads help)
    media_items = [None] * len(files)
    photos = 0
    documents = 0
    skipped = 0

    print(f"Processing {len(files):,} images...")

    # Use threads for I/O parallelism (reading PNG headers)
    BATCH = 10000
    for batch_start in range(0, len(files), BATCH):
        batch_end = min(batch_start + BATCH, len(files))
        batch_files = files[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(process_file, f, batch_start + i): i
                for i, f in enumerate(batch_files)
            }
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()
                if result is None:
                    skipped += 1
                else:
                    media_items[batch_start + i] = result
                    if result["type"] == "photo":
                        photos += 1
                    else:
                        documents += 1

        elapsed = time.time() - start
        processed = batch_end
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  {processed:,}/{len(files):,} | {photos:,} photos, {documents:,} docs, {skipped:,} skipped | {rate:.0f}/s | {elapsed:.0f}s", flush=True)

    # Remove None entries (skipped files)
    media_items = [m for m in media_items if m is not None]

    # Reassign sequential IDs
    for i, item in enumerate(media_items):
        item["id"] = f"m-{10000 + i}"

    elapsed = time.time() - start

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(media_items):,} items to {OUTPUT_PATH}...")
    OUTPUT_PATH.write_text(json.dumps(media_items, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total items: {len(media_items):,}")
    print(f"  Photos: {photos:,}")
    print(f"  Documents: {documents:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
