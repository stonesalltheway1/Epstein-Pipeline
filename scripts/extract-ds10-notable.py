"""Extract only the notable/cataloged images from DS10 using rhowardstone's catalog.

Instead of extracting all 503K PDFs (18+ hours), this extracts only the ~15K PDFs
that rhowardstone identified as containing notable images, along with their AI descriptions.

Usage:
    python scripts/extract-ds10-notable.py
"""

import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DS10_IMAGES = Path("E:/epstein-ds10/extracted/VOL00010/IMAGES")
CATALOG_PATH = Path("E:/Epstein-research-data/image_catalog.csv.gz")
OUTPUT_IMAGES = Path("E:/epstein-ds10/notable-images")
OUTPUT_JSON = Path("E:/epstein-ds10/notable-catalog.json")
SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")

MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_SIZE = 5000


def load_catalog():
    """Load rhowardstone's image catalog."""
    catalog = {}  # efta_id -> list of image records
    with gzip.open(str(CATALOG_PATH), "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            efta = r.get("efta_number", "")
            if not efta:
                continue
            if efta not in catalog:
                catalog[efta] = []
            catalog[efta].append({
                "image_name": r.get("image_name", ""),
                "page_number": int(r.get("page_number", 0)),
                "people": r.get("people", ""),
                "text_content": r.get("text_content", ""),
                "objects": r.get("objects", ""),
                "setting": r.get("setting", ""),
                "activity": r.get("activity", ""),
                "notable": r.get("notable", ""),
            })
    return catalog


def find_pdf(efta_id):
    """Find the PDF file for an EFTA ID across subdirectories."""
    for subdir in DS10_IMAGES.iterdir():
        if not subdir.is_dir():
            continue
        pdf_path = subdir / f"{efta_id}.pdf"
        if pdf_path.exists():
            return pdf_path
    return None


def extract_one(args):
    """Extract images from one PDF, return metadata."""
    pdf_path, efta_id, output_dir, catalog_entries = args
    results = []

    try:
        import fitz
        doc = fitz.open(str(pdf_path))

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

                    # Find matching catalog entry
                    description = ""
                    people = ""
                    setting = ""
                    for entry in catalog_entries:
                        if entry["page_number"] == page_num:
                            description = entry.get("notable", "") or entry.get("objects", "")
                            people = entry.get("people", "")
                            setting = entry.get("setting", "")
                            break

                    results.append({
                        "efta_id": efta_id,
                        "filename": filename,
                        "page": page_num,
                        "width": w,
                        "height": h,
                        "size_bytes": len(data),
                        "description": description.strip(),
                        "people": people.strip(),
                        "setting": setting.strip(),
                    })

                except Exception:
                    continue

        doc.close()
    except Exception as e:
        pass

    return results


def main():
    print("Loading rhowardstone image catalog...")
    catalog = load_catalog()
    total_images = sum(len(v) for v in catalog.values())
    print(f"  {total_images:,} images across {len(catalog):,} EFTA documents")

    # Filter to DS10 range (EFTA numbers in the VOL00010 range)
    # Check which EFTA IDs actually exist in our extracted DS10
    print(f"\nScanning {DS10_IMAGES} for matching PDFs...")
    matches = {}
    scanned = 0
    for subdir in sorted(DS10_IMAGES.iterdir()):
        if not subdir.is_dir():
            continue
        for pdf in subdir.glob("*.pdf"):
            efta_id = pdf.stem
            if efta_id in catalog:
                matches[efta_id] = (str(pdf), catalog[efta_id])
        scanned += 1
        if scanned % 50 == 0:
            print(f"  Scanned {scanned} subdirs, found {len(matches):,} matches...")

    print(f"\nFound {len(matches):,} EFTA PDFs with catalog entries")
    total_catalog_images = sum(len(v[1]) for v in matches.values())
    print(f"  Expected ~{total_catalog_images:,} notable images")

    if not matches:
        print("No matches found!")
        return

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Prepare work items
    work = [
        (pdf_path, efta_id, str(OUTPUT_IMAGES), entries)
        for efta_id, (pdf_path, entries) in matches.items()
    ]

    print(f"\nExtracting images from {len(work):,} PDFs using {max(1, cpu_count()-2)} workers...")
    workers = max(1, cpu_count() - 2)
    all_results = []
    start = time.time()

    with Pool(workers) as pool:
        for i, results in enumerate(pool.imap_unordered(extract_one, work, chunksize=20)):
            all_results.extend(results)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(work) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1:,}/{len(work):,} PDFs | {len(all_results):,} images | {rate:.0f}/s | ~{remaining/60:.0f}m remaining")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Extracted: {len(all_results):,} images")

    # Save catalog
    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Catalog: {OUTPUT_JSON}")

    # Stats
    with_desc = sum(1 for r in all_results if r["description"])
    with_people = sum(1 for r in all_results if r["people"] and "no people" not in r["people"].lower() and "no visible" not in r["people"].lower())
    print(f"  With descriptions: {with_desc:,}")
    print(f"  With people detected: {with_people:,}")


if __name__ == "__main__":
    main()
