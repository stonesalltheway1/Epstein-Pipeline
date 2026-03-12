"""DOJ Data Set 10 photo ingestion pipeline.

Downloads DS10 from Archive.org, extracts images from EFTA PDFs,
generates thumbnails, uploads to R2, and produces media JSON for the site.

This is a LONG-RUNNING pipeline (78.6 GB download + extraction + upload).
Run in stages:

    # Stage 1: Download DS10 from Archive.org
    python scripts/ingest-ds10-photos.py download --output-dir E:/epstein-ds10

    # Stage 2: Extract images from PDFs
    python scripts/ingest-ds10-photos.py extract --input-dir E:/epstein-ds10 --output-dir E:/epstein-ds10/images

    # Stage 3: Generate thumbnails
    python scripts/ingest-ds10-photos.py thumbnails --input-dir E:/epstein-ds10/images --output-dir E:/epstein-ds10/thumbs

    # Stage 4: Upload to R2
    python scripts/ingest-ds10-photos.py upload --input-dir E:/epstein-ds10/thumbs --bucket epstein-media

    # Stage 5: Generate media JSON for site
    python scripts/ingest-ds10-photos.py export --input-dir E:/epstein-ds10/images --output data/ds10-media.json

    # Or run everything:
    python scripts/ingest-ds10-photos.py all --output-dir E:/epstein-ds10
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)
console = Console()

# ── Constants ─────────────────────────────────────────────────────────────
ARCHIVE_DS10_ID = "Epstein_Files_Transparency_Act_Data_Set_10_Zip"
ARCHIVE_DS10_ALT = "data-set-10"
ARCHIVE_DOWNLOAD = "https://archive.org/download"
ARCHIVE_METADATA = "https://archive.org/metadata"

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
R2_PROXY = "https://media.epsteinexposed.com"

# Min image dimensions to keep (skip tiny icons, artifacts)
MIN_WIDTH = 100
MIN_HEIGHT = 100
# Min file size (skip empty/corrupt images)
MIN_SIZE_BYTES = 5_000  # 5KB


@click.group()
def cli():
    """DOJ Data Set 10 photo ingestion pipeline."""
    pass


# ── Stage 1: Download ────────────────────────────────────────────────────

@cli.command()
@click.option("--output-dir", type=click.Path(), required=True, help="Directory to download DS10 files")
@click.option("--max-files", type=int, default=None, help="Limit number of files (for testing)")
@click.option("--extensions", default=".pdf,.jpg,.jpeg,.png,.mp4,.mov", help="File extensions to download")
def download(output_dir: str, max_files: int | None, extensions: str):
    """Stage 1: Download DS10 files from Archive.org."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext_set = {e.strip().lower() for e in extensions.split(",")}

    console.print(f"[bold cyan]DS10 Photo Pipeline — Stage 1: Download[/bold cyan]")
    console.print(f"Output: {out.resolve()}")
    console.print(f"Extensions: {ext_set}")

    # Fetch file list
    console.print("[dim]Fetching DS10 file manifest...[/dim]")
    files = _fetch_ds10_files()
    if not files:
        console.print("[red]No files found! Check Archive.org availability.[/red]")
        return

    # Filter by extension
    target_files = [
        f for f in files
        if any(f["name"].lower().endswith(ext) for ext in ext_set)
        and f.get("source", "original") == "original"
    ]

    if max_files:
        target_files = target_files[:max_files]

    total_size = sum(int(f.get("size", 0)) for f in target_files)
    console.print(f"[green]Found {len(target_files):,} files ({total_size / 1e9:.1f} GB)[/green]")

    # Download with progress
    downloaded = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading DS10", total=len(target_files))

        for file_meta in target_files:
            filename = file_meta["name"]
            dest = out / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            expected_size = int(file_meta.get("size", 0))

            # Skip if already downloaded
            if dest.exists() and expected_size > 0 and dest.stat().st_size == expected_size:
                skipped += 1
                progress.advance(task)
                continue

            url = f"{ARCHIVE_DOWNLOAD}/{ARCHIVE_DS10_ID}/{filename}"
            success = _download_file(url, dest)
            if success:
                downloaded += 1
            progress.advance(task)
            time.sleep(0.1)

    console.print(f"\n[bold green]Downloaded {downloaded:,} files, skipped {skipped:,} existing[/bold green]")


# ── Stage 2: Extract images ──────────────────────────────────────────────

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="Directory with DS10 PDFs")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory for extracted images")
@click.option("--workers", type=int, default=4, help="Parallel extraction workers")
@click.option("--skip-existing", is_flag=True, default=True, help="Skip already extracted PDFs")
def extract(input_dir: str, output_dir: str, workers: int, skip_existing: bool):
    """Stage 2: Extract images from DS10 PDFs."""
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]DS10 Photo Pipeline — Stage 2: Extract Images[/bold cyan]")

    # Find all PDFs
    pdfs = sorted(inp.rglob("*.pdf"))
    console.print(f"Found {len(pdfs):,} PDF files")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        console.print("[red]PyMuPDF required: pip install pymupdf[/red]")
        return

    extracted_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting images", total=len(pdfs))

        for pdf_path in pdfs:
            efta_id = pdf_path.stem  # e.g., EFTA01262782
            img_dir = out / efta_id

            if skip_existing and img_dir.exists() and any(img_dir.iterdir()):
                progress.advance(task)
                continue

            try:
                count = _extract_pdf_images(pdf_path, img_dir)
                extracted_count += count
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path.name}: {e}")
                error_count += 1

            progress.advance(task)

    console.print(f"\n[bold green]Extracted {extracted_count:,} images from {len(pdfs):,} PDFs ({error_count} errors)[/bold green]")


# ── Stage 3: Generate thumbnails ─────────────────────────────────────────

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="Directory with extracted images")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory for thumbnails")
@click.option("--size", type=int, default=640, help="Thumbnail max dimension")
def thumbnails(input_dir: str, output_dir: str, size: int):
    """Stage 3: Generate thumbnails for extracted images."""
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]DS10 Photo Pipeline — Stage 3: Thumbnails[/bold cyan]")

    try:
        from PIL import Image
    except ImportError:
        console.print("[red]Pillow required: pip install Pillow[/red]")
        return

    images = list(inp.rglob("*.png")) + list(inp.rglob("*.jpg")) + list(inp.rglob("*.jpeg"))
    # Filter by minimum size
    images = [i for i in images if i.stat().st_size >= MIN_SIZE_BYTES]
    console.print(f"Found {len(images):,} images to thumbnail")

    created = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating thumbnails", total=len(images))

        for img_path in images:
            thumb_path = out / img_path.parent.name / f"{img_path.stem}_thumb.jpg"
            thumb_path.parent.mkdir(parents=True, exist_ok=True)

            if thumb_path.exists():
                progress.advance(task)
                continue

            try:
                with Image.open(img_path) as img:
                    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                        progress.advance(task)
                        continue
                    img.thumbnail((size, size), Image.LANCZOS)
                    img.convert("RGB").save(thumb_path, "JPEG", quality=85, optimize=True)
                    created += 1
            except Exception as e:
                logger.warning(f"Thumbnail failed for {img_path.name}: {e}")

            progress.advance(task)

    console.print(f"\n[bold green]Created {created:,} thumbnails[/bold green]")


# ── Stage 4: Upload to R2 ────────────────────────────────────────────────

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="Directory with thumbnails")
@click.option("--full-dir", type=click.Path(exists=True), default=None, help="Directory with full-size images")
@click.option("--workers", type=int, default=8, help="Upload concurrency")
@click.option("--dry-run", is_flag=True, help="Don't actually upload")
def upload(input_dir: str, full_dir: str | None, workers: int, dry_run: bool):
    """Stage 4: Upload images to Cloudflare R2 via the media proxy."""
    console.print(f"[bold cyan]DS10 Photo Pipeline — Stage 4: Upload to R2[/bold cyan]")

    # R2 upload requires S3 API credentials
    # Check for environment variables
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_endpoint = os.environ.get("R2_ENDPOINT")

    if not all([r2_access_key, r2_secret_key, r2_endpoint]):
        console.print("[yellow]R2 credentials not set. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT[/yellow]")
        console.print("[yellow]Skipping upload — images will be served directly from the media proxy on first request[/yellow]")
        console.print("[dim]The media proxy at media.epsteinexposed.com will cache images from source on first access[/dim]")
        return

    try:
        import boto3
    except ImportError:
        console.print("[red]boto3 required for R2 upload: pip install boto3[/red]")
        return

    inp = Path(input_dir)
    thumbs = list(inp.rglob("*.jpg"))
    console.print(f"Found {len(thumbs):,} thumbnails to upload")

    if dry_run:
        console.print("[yellow]DRY RUN — would upload to R2[/yellow]")
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
    )

    uploaded = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading to R2", total=len(thumbs))

        for thumb in thumbs:
            key = f"ds10/{thumb.parent.name}/{thumb.name}"
            try:
                s3.upload_file(
                    str(thumb), "epstein-media", key,
                    ExtraArgs={"ContentType": "image/jpeg", "CacheControl": "public, max-age=31536000"},
                )
                uploaded += 1
            except Exception as e:
                logger.warning(f"Upload failed for {thumb.name}: {e}")
            progress.advance(task)

    console.print(f"\n[bold green]Uploaded {uploaded:,} thumbnails to R2[/bold green]")


# ── Stage 5: Export media JSON ────────────────────────────────────────────

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="Directory with extracted images")
@click.option("--thumb-dir", type=click.Path(exists=True), default=None, help="Directory with thumbnails")
@click.option("--jmail-metadata", type=click.Path(exists=True), default=None, help="jmail photo metadata JSON for AI descriptions")
@click.option("--output", type=click.Path(), required=True, help="Output JSON path")
@click.option("--start-id", type=int, default=5000, help="Starting media ID number")
def export(input_dir: str, thumb_dir: str | None, jmail_metadata: str | None, output: str, start_id: int):
    """Stage 5: Generate media JSON for the site."""
    inp = Path(input_dir)
    out_path = Path(output)

    console.print(f"[bold cyan]DS10 Photo Pipeline — Stage 5: Export Media JSON[/bold cyan]")

    # Load jmail descriptions if available
    jmail_descs = {}
    if jmail_metadata:
        jmail_path = Path(jmail_metadata)
        if jmail_path.exists():
            jmail_data = json.loads(jmail_path.read_text("utf-8"))
            for item in jmail_data:
                # Map EFTA ID to description
                photo_id = item.get("id", "")
                desc = item.get("image_description", "")
                if desc:
                    efta_id = photo_id.split("-")[0] if "-" in photo_id else photo_id.replace(".png", "")
                    jmail_descs[efta_id] = desc
            console.print(f"Loaded {len(jmail_descs):,} AI descriptions from jmail metadata")

    # Scan extracted images
    image_dirs = sorted(d for d in inp.iterdir() if d.is_dir())
    console.print(f"Found {len(image_dirs):,} EFTA document directories")

    media_items = []
    next_id = start_id

    for img_dir in image_dirs:
        efta_id = img_dir.name
        images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
        images = [i for i in images if i.stat().st_size >= MIN_SIZE_BYTES]

        if not images:
            continue

        for img in images:
            # Build thumbnail URL
            thumb_key = f"ds10/{efta_id}/{img.stem}_thumb.jpg"
            thumb_url = f"{R2_PROXY}/{thumb_key}"

            # Full image URL (direct from R2 or DOJ source)
            full_key = f"ds10/{efta_id}/{img.name}"
            full_url = f"{R2_PROXY}/{full_key}"

            # Get description from jmail if available
            desc = jmail_descs.get(efta_id, "")

            # Determine if this is a photo vs document based on image characteristics
            try:
                from PIL import Image
                with Image.open(img) as pil_img:
                    w, h = pil_img.size
                    # Very tall/narrow = likely document page, not photo
                    ratio = h / w if w > 0 else 1
                    img_type = "document" if ratio > 1.8 else "photo"
            except Exception:
                img_type = "photo"
                w, h = 0, 0

            media_items.append({
                "id": f"m-{next_id}",
                "title": f"{efta_id} — {img.stem}",
                "type": img_type,
                "description": desc,
                "date": "",
                "source": "DOJ EFTA Data Set 10",
                "sourceUrl": f"https://www.justice.gov/epstein/doj-disclosures",
                "thumbnailUrl": thumb_url,
                "fullUrl": full_url,
                "personIds": [],
                "locationIds": [],
                "documentIds": [efta_id] if efta_id.startswith("EFTA") else [],
                "tags": ["doj", "data-set-10", "fbi-evidence"],
                "verified": True,
                "width": w,
                "height": h,
            })
            next_id += 1

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(media_items, indent=2, ensure_ascii=False), encoding="utf-8")

    photos = sum(1 for m in media_items if m["type"] == "photo")
    docs = sum(1 for m in media_items if m["type"] == "document")
    console.print(f"\n[bold green]Exported {len(media_items):,} items ({photos:,} photos, {docs:,} documents)[/bold green]")
    console.print(f"Output: {out_path.resolve()}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _fetch_ds10_files() -> list[dict]:
    """Fetch the file list for DS10 from Archive.org."""
    for ident in [ARCHIVE_DS10_ID, ARCHIVE_DS10_ALT]:
        url = f"{ARCHIVE_METADATA}/{ident}/files"
        try:
            resp = httpx.get(url, timeout=60.0, follow_redirects=True)
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("result", data) if isinstance(data, dict) else data
                if files:
                    console.print(f"[green]Using Archive.org item: {ident}[/green]")
                    return files
        except Exception:
            continue
    return []


def _download_file(url: str, dest: Path) -> bool:
    """Download a single file with resume support."""
    headers = {}
    mode = "wb"
    existing = 0

    if dest.exists():
        existing = dest.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    try:
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as stream:
                if stream.status_code == 416:
                    return True
                stream.raise_for_status()
                with open(dest, mode) as f:
                    for chunk in stream.iter_bytes(65_536):
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Download failed: {dest.name}: {e}")
        return False


def _extract_pdf_images(pdf_path: Path, output_dir: Path) -> int:
    """Extract images from a single PDF using PyMuPDF."""
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Method 1: Extract embedded images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                w = base_image.get("width", 0)
                h = base_image.get("height", 0)

                if w < MIN_WIDTH or h < MIN_HEIGHT or len(image_bytes) < MIN_SIZE_BYTES:
                    continue

                filename = f"p{page_num:04d}_i{img_idx:03d}.{ext}"
                (output_dir / filename).write_bytes(image_bytes)
                count += 1
            except Exception:
                continue

        # Method 2: If no images found, render page as image (for scanned PDFs)
        if count == 0 and page_num == 0:
            try:
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for quality
                pix = page.get_pixmap(matrix=mat)
                if pix.width >= MIN_WIDTH and pix.height >= MIN_HEIGHT:
                    filename = f"p{page_num:04d}_render.png"
                    pix.save(str(output_dir / filename))
                    count += 1
            except Exception:
                pass

    doc.close()
    return count


if __name__ == "__main__":
    cli()
