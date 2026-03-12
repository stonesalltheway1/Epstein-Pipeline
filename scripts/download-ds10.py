"""Download DOJ Data Set 10 (78.6 GB ZIP) from Archive.org.

Downloads the ZIP file with resume support, then extracts it.

Usage:
    python scripts/download-ds10.py --output-dir E:/epstein-ds10
    python scripts/download-ds10.py --output-dir E:/epstein-ds10 --extract-only  # skip download
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import zipfile
from pathlib import Path

import httpx

# Archive.org DS10 download URL
DS10_URL = "https://archive.org/download/Epstein_Files_Transparency_Act_Data_Set_10_Zip/DataSet-10.zip"
DS10_ALT_URL = "https://archive.org/download/data-set-10/DataSet%2010.zip"
EXPECTED_SIZE = 84_439_381_640  # ~78.6 GB


def download_with_resume(url: str, dest: Path) -> bool:
    """Download a large file with resume support and progress."""
    headers = {}
    mode = "wb"
    existing = 0

    if dest.exists():
        existing = dest.stat().st_size
        if existing >= EXPECTED_SIZE:
            print(f"  Already downloaded: {dest} ({existing / 1e9:.1f} GB)")
            return True
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"  Resuming from {existing / 1e9:.2f} GB...")

    try:
        with httpx.Client(timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as stream:
                if stream.status_code == 416:
                    print("  File already complete (416)")
                    return True

                if stream.status_code not in (200, 206):
                    print(f"  HTTP {stream.status_code} - trying alternate URL...")
                    return False

                total = int(stream.headers.get("content-length", 0)) + existing
                print(f"  Total size: {total / 1e9:.1f} GB")

                with open(dest, mode) as f:
                    downloaded = existing
                    last_report = time.time()
                    last_bytes = downloaded

                    for chunk in stream.iter_bytes(chunk_size=1_048_576):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress every 10 seconds
                        now = time.time()
                        if now - last_report >= 10:
                            speed = (downloaded - last_bytes) / (now - last_report) / 1e6
                            pct = (downloaded / total * 100) if total else 0
                            remaining = ((total - downloaded) / (speed * 1e6)) / 3600 if speed > 0 else 0
                            print(f"  {downloaded / 1e9:.2f} / {total / 1e9:.1f} GB ({pct:.1f}%) - {speed:.1f} MB/s - ~{remaining:.1f}h remaining")
                            last_report = now
                            last_bytes = downloaded

        print(f"  Download complete: {downloaded / 1e9:.1f} GB")
        return True

    except KeyboardInterrupt:
        print(f"\n  Interrupted at {dest.stat().st_size / 1e9:.2f} GB. Run again to resume.")
        return False
    except Exception as e:
        print(f"  Download error: {e}")
        print(f"  Downloaded so far: {dest.stat().st_size / 1e9:.2f} GB. Run again to resume.")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> int:
    """Extract PDF/image files from the DS10 ZIP."""
    print(f"\nExtracting {zip_path.name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        print(f"  Archive contains {len(members):,} files")

        # Filter to PDFs and images
        targets = [
            m for m in members
            if m.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".mp4", ".mov"))
            and not m.startswith("__MACOSX")
        ]
        print(f"  Target files (PDF/image/video): {len(targets):,}")

        extracted = 0
        for i, member in enumerate(targets):
            dest = output_dir / member
            if dest.exists() and dest.stat().st_size > 0:
                extracted += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(member) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(1_048_576)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted += 1
            except Exception as e:
                print(f"  Error extracting {member}: {e}")

            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i + 1:,} / {len(targets):,} files...")

    print(f"  Extracted {extracted:,} files to {output_dir}")
    return extracted


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download DOJ Data Set 10 from Archive.org")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (needs ~85 GB free)")
    parser.add_argument("--extract-only", action="store_true", help="Skip download, extract existing ZIP")
    parser.add_argument("--no-extract", action="store_true", help="Download only, don't extract")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "DataSet-10.zip"

    print("=" * 60)
    print("DOJ Data Set 10 Download Pipeline")
    print(f"Output: {out.resolve()}")
    print(f"Expected: ~78.6 GB ZIP -> ~180,000 images + ~2,000 videos")
    print("=" * 60)

    # Download
    if not args.extract_only:
        print(f"\nDownloading {DS10_URL}")
        print(f"  Destination: {zip_path}")

        ok = download_with_resume(DS10_URL, zip_path)
        if not ok:
            print(f"\nTrying alternate URL: {DS10_ALT_URL}")
            ok = download_with_resume(DS10_ALT_URL, zip_path)

        if not ok:
            print("\nDownload failed. Run again to resume from where it left off.")
            sys.exit(1)

    # Extract
    if not args.no_extract:
        if not zip_path.exists():
            print(f"ZIP not found at {zip_path}")
            sys.exit(1)
        extract_zip(zip_path, out / "extracted")

    print("\nDone! Next steps:")
    print(f"  1. Extract images: python scripts/ingest-ds10-photos.py extract --input-dir {out}/extracted --output-dir {out}/images")
    print(f"  2. Generate thumbs: python scripts/ingest-ds10-photos.py thumbnails --input-dir {out}/images --output-dir {out}/thumbs")
    print(f"  3. Export JSON:     python scripts/ingest-ds10-photos.py export --input-dir {out}/images --output ds10-media.json")


if __name__ == "__main__":
    main()
