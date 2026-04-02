"""Fast PDF assembly for HOC documents using pre-built image index."""

import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

INDEX_PATH = Path("E:/Epstein-Pipeline/output/hoc-image-index.json")
OPT_PATH = Path("E:/Epstein-Pipeline/ingest/kaggle-hoc/DATA-20251116T222054Z-1-001/DATA/HOUSE_OVERSIGHT_009.opt")
OUTPUT_DIR = Path("E:/Epstein-Pipeline/output/hoc-pdfs")


def main():
    import fitz  # PyMuPDF

    # Load pre-built image index
    index = json.loads(INDEX_PATH.read_text())
    print(f"Image index: {len(index):,} files")

    # Parse .opt to get page groupings
    docs = {}  # doc_id -> [image_filenames]
    current_doc = None

    with open(OPT_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            bates_id = row[0].strip()
            img_relative = row[2].strip()
            # Extract just the filename
            img_filename = img_relative.replace("\\", "/").split("/")[-1]
            is_start = row[3].strip().upper() == "Y"

            if is_start:
                m = re.search(r"(\d+)$", bates_id)
                bates_num = int(m.group(1)) if m else 0
                current_doc = f"kaggle-ho-{bates_num:06d}"
                docs[current_doc] = []

            if current_doc:
                docs[current_doc].append(img_filename)

    print(f"Documents: {len(docs):,}")
    total_pages = sum(len(p) for p in docs.values())
    print(f"Total pages: {total_pages:,}")

    # Assemble PDFs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    assembled = 0
    failed = 0
    skipped = 0

    for i, (doc_id, page_files) in enumerate(docs.items()):
        pdf_path = OUTPUT_DIR / f"{doc_id}.pdf"
        if pdf_path.exists():
            skipped += 1
            continue

        resolved = []
        for fn in page_files:
            if fn in index:
                resolved.append(index[fn])

        if not resolved:
            failed += 1
            continue

        try:
            pdf = fitz.open()
            for img_path in resolved:
                img = fitz.open(img_path)
                pdf_bytes = img.convert_to_pdf()
                img_pdf = fitz.open("pdf", pdf_bytes)
                pdf.insert_pdf(img_pdf)
                img.close()
                img_pdf.close()

            if pdf.page_count > 0:
                pdf.save(str(pdf_path))
                assembled += 1
            pdf.close()
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Error on {doc_id}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            done = assembled + failed
            rate = done / elapsed if elapsed > 0 else 1
            remaining = (len(docs) - i - 1) / rate if rate > 0 else 0
            m, s = divmod(int(remaining), 60)
            pct = (i + 1) / len(docs) * 100
            print(f"  {i+1}/{len(docs)} ({pct:.0f}%) assembled={assembled} skipped={skipped} failed={failed} ~{m}m{s}s")

    elapsed = time.time() - t0
    print(f"\nDone in {int(elapsed)}s: {assembled} assembled, {skipped} skipped, {failed} failed")

    import subprocess
    r = subprocess.run(["du", "-sh", str(OUTPUT_DIR)], capture_output=True, text=True)
    print(f"Total size: {r.stdout.strip()}")


if __name__ == "__main__":
    main()
