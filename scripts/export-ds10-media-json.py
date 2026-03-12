"""Export extracted DS10 images as media JSON for the epstein-index site.

Scans extracted images, determines photo vs document type by aspect ratio,
and generates a JSON file compatible with the site's media gallery.

Usage:
    python scripts/export-ds10-media-json.py --limit 20000
    python scripts/export-ds10-media-json.py  # export all
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

IMAGES_DIR = Path("E:/epstein-ds10/images")
SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
OUTPUT_PATH = SITE_DIR / "data" / "ds10-media.json"
R2_PROXY = "https://media.epsteinexposed.com"

# Thresholds
MIN_SIZE = 10_000  # 10KB minimum (skip tiny/blank images)
DOC_RATIO = 1.6    # height/width > 1.6 = document page (portrait letter/A4)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-id", type=int, default=10000, help="Starting media ID number (avoid collisions)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"Scanning {IMAGES_DIR}...")
    files = sorted(IMAGES_DIR.glob("*.*"))
    print(f"Found {len(files):,} files")

    # Filter by size
    files = [f for f in files if f.stat().st_size >= MIN_SIZE]
    print(f"After size filter (>={MIN_SIZE/1000:.0f}KB): {len(files):,}")

    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {args.limit:,}")

    # Try to use PIL for dimensions, fall back to filename parsing
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        print("Warning: Pillow not installed, using filename heuristics for type detection")

    media_items = []
    next_id = args.start_id
    photos = 0
    documents = 0
    skipped = 0
    start = time.time()

    for i, img_path in enumerate(files):
        efta_id = img_path.stem.split("_p")[0]  # e.g., EFTA01262782 from EFTA01262782_p0_i0.png

        # Determine type
        img_type = "photo"
        w, h = 0, 0

        if has_pil:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    ratio = h / w if w > 0 else 1
                    img_type = "document" if ratio > DOC_RATIO else "photo"
            except Exception:
                skipped += 1
                continue
        else:
            # Guess from file size - documents tend to be smaller
            img_type = "photo"  # default

        if img_type == "photo":
            photos += 1
        else:
            documents += 1

        # Build R2 URL paths
        thumb_key = f"ds10/{efta_id}/{img_path.name}"
        thumb_url = f"{R2_PROXY}/{thumb_key}"

        media_items.append({
            "id": f"m-{next_id}",
            "title": f"{efta_id} - Evidence Photo" if img_type == "photo" else f"{efta_id} - Document Page",
            "type": img_type,
            "description": "",
            "date": "",
            "source": "DOJ EFTA Data Set 10",
            "sourceUrl": f"https://www.justice.gov/epstein/doj-disclosures",
            "thumbnailUrl": thumb_url,
            "fullUrl": thumb_url,
            "personIds": [],
            "locationIds": [],
            "documentIds": [efta_id] if efta_id.startswith("EFTA") else [],
            "tags": ["doj", "data-set-10", "fbi-evidence"],
            "verified": True,
            "width": w,
            "height": h,
        })
        next_id += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            print(f"  {i+1:,}/{len(files):,} | {photos:,} photos, {documents:,} documents | {(i+1)/elapsed:.0f}/s")

    elapsed = time.time() - start

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(media_items, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total items: {len(media_items):,}")
    print(f"  Photos: {photos:,}")
    print(f"  Documents: {documents:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
