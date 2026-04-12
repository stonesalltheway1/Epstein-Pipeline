"""House Oversight Committee document release downloader.

Oversight posts release announcements on oversight.house.gov but hosts the
actual documents on Google Drive and Dropbox. This module:
1. Scrapes release pages for Drive/Dropbox folder URLs
2. Uses gdown for public Google Drive folders (no API key needed)
3. Falls back to rsync-from-dropbox helper for Dropbox folders

Known release batches:
    2025-09 (first Epstein estate batch)
    2025-11-13 (20K+ additional estate docs)
    2025-12-03 (JPMorgan + Deutsche Bank records, ~5K docs)
    2026-02-09 (Maxwell virtual deposition transcript)
    2026-02-26 (Hillary Clinton closed-door testimony)
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; EpsteinExposedBot/1.0)"


@dataclass
class OversightRelease:
    """A single Oversight release announcement."""
    id: str            # short slug (e.g. "2025-11-13-estate")
    url: str           # release-page URL
    title: str
    date: str          # ISO date
    source: str = "house-oversight"
    tags: tuple[str, ...] = field(default_factory=tuple)


# Known public releases — add new ones here as they're announced
KNOWN_RELEASES: list[OversightRelease] = [
    OversightRelease(
        id="2025-09-epstein-estate",
        url="https://oversight.house.gov/release/oversight-committee-releases-records-provided-by-the-epstein-estate-chairman-comer-provides-statement/",
        title="Oversight Committee Releases Records Provided by the Epstein Estate (Sept 2, 2025)",
        date="2025-09-02",
        tags=("epstein-estate", "initial-batch"),
    ),
    OversightRelease(
        id="2025-11-13-epstein-estate",
        url="https://oversight.house.gov/release/oversight-committee-releases-additional-epstein-estate-documents/",
        title="Oversight Committee Releases Additional Epstein Estate Documents (Nov 13, 2025)",
        date="2025-11-13",
        tags=("epstein-estate", "additional"),
    ),
    OversightRelease(
        id="2025-11-subpoenas-banks",
        url="https://oversight.house.gov/release/chairman-comer-subpoenas-banks-for-epstein-records-seeks-information-from-usvi-attorney-general/",
        title="Comer Subpoenas Banks for Epstein Records, Seeks Information from USVI AG (Nov 2025)",
        date="2025-11-06",
        tags=("jpmorgan", "deutsche-bank", "subpoena"),
    ),
    # TODO: Dec 3, 2025 JPM+Deutsche records transmittal — no public release page found yet
    # CNBC reported records "transmitted to Oversight" — may be in a later public release
]


def scrape_release_links(url: str) -> dict[str, list[str]]:
    """Find Google Drive folder IDs and Dropbox URLs on a release page."""
    r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    html = r.text

    # Google Drive folder URLs: drive.google.com/drive/folders/ID
    gdrive_ids = set(re.findall(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_\-]+)", html))
    # Google Drive file URLs: drive.google.com/file/d/ID
    gdrive_file_ids = set(re.findall(r"drive\.google\.com/file/d/([a-zA-Z0-9_\-]+)", html))
    # Dropbox folder URLs
    dropbox_urls = set(re.findall(r'https?://(?:www\.)?dropbox\.com/[^\s"<>]+', html))
    # Direct PDFs
    pdf_urls = set(re.findall(r'href="(https?://[^\"]+\.pdf[^\"]*)"', html))

    return {
        "gdrive_folders": sorted(gdrive_ids),
        "gdrive_files": sorted(gdrive_file_ids),
        "dropbox": sorted(dropbox_urls),
        "pdf_direct": sorted(pdf_urls),
    }


def download_gdrive_folder(folder_id: str, output_dir: Path) -> int:
    """Download a public Google Drive folder using gdown. Returns file count."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown not installed. Install with: pip install gdown")

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info("Downloading Google Drive folder %s → %s", folder_id, output_dir)
    # gdown.download_folder returns list of downloaded file paths
    files = gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    return len(files) if files else 0


def download_dropbox_folder(dropbox_url: str, output_dir: Path) -> int:
    """Download a public Dropbox folder as a zip, then extract.

    Dropbox supports ?dl=1 to force a zip download of the whole folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Force zip download
    if "dl=0" in dropbox_url:
        zip_url = dropbox_url.replace("dl=0", "dl=1")
    elif "dl=1" not in dropbox_url:
        sep = "&" if "?" in dropbox_url else "?"
        zip_url = dropbox_url + sep + "dl=1"
    else:
        zip_url = dropbox_url

    zip_path = output_dir / "_dropbox_bundle.zip"
    logger.info("Downloading Dropbox zip: %s", zip_url)
    r = requests.get(zip_url, stream=True, timeout=120,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Extract
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(output_dir)
    logger.info("Extracted %d files from Dropbox zip", len(names))
    zip_path.unlink(missing_ok=True)
    return len(names)


def pull_release(release: OversightRelease, output_root: Path) -> dict:
    """Pull all documents linked on a release page."""
    release_dir = output_root / release.id
    links = scrape_release_links(release.url)
    logger.info("Release %s: %d gdrive folders, %d gdrive files, %d dropbox, %d direct PDFs",
                release.id, len(links["gdrive_folders"]),
                len(links["gdrive_files"]), len(links["dropbox"]),
                len(links["pdf_direct"]))

    counts = {"gdrive": 0, "dropbox": 0, "direct_pdf": 0}

    # Google Drive folders
    for folder_id in links["gdrive_folders"]:
        sub = release_dir / f"gdrive_{folder_id}"
        try:
            n = download_gdrive_folder(folder_id, sub)
            counts["gdrive"] += n
        except Exception as e:
            logger.exception("Drive folder %s failed: %s", folder_id, e)

    # Dropbox folders
    for url in links["dropbox"]:
        sub = release_dir / "dropbox"
        try:
            n = download_dropbox_folder(url, sub)
            counts["dropbox"] += n
        except Exception as e:
            logger.exception("Dropbox %s failed: %s", url, e)

    # Direct PDFs (usually just committee rules etc, but include for completeness)
    for pdf_url in links["pdf_direct"]:
        if "119th-Committee-Rules" in pdf_url:
            continue  # skip non-Epstein boilerplate
        sub = release_dir / "direct"
        sub.mkdir(parents=True, exist_ok=True)
        name = urlparse(pdf_url).path.rsplit("/", 1)[-1] or "file.pdf"
        dest = sub / name
        if dest.exists():
            continue
        try:
            r = requests.get(pdf_url, stream=True, timeout=60,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            counts["direct_pdf"] += 1
        except Exception as e:
            logger.warning("Direct PDF %s failed: %s", pdf_url, e)

    return {
        "id": release.id, "title": release.title, "date": release.date,
        "output_dir": str(release_dir), "counts": counts, "links": links,
    }


def pull_all(output_root: Path) -> list[dict]:
    results = []
    for release in KNOWN_RELEASES:
        try:
            r = pull_release(release, output_root)
            results.append(r)
        except Exception as e:
            logger.exception("Release %s failed: %s", release.id, e)
            results.append({"id": release.id, "error": str(e)})
    return results


if __name__ == "__main__":
    import argparse
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="Specific release id (e.g. 2025-11-13-epstein-estate)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("./output/house-oversight"))
    parser.add_argument("--scan-only", action="store_true",
                        help="Scrape and list links without downloading")
    args = parser.parse_args()

    if args.scan_only:
        targets = KNOWN_RELEASES if args.all else [r for r in KNOWN_RELEASES if r.id == args.release]
        for r in targets:
            links = scrape_release_links(r.url)
            print(f"\n=== {r.id} ({r.date}) ===")
            print(f"  URL: {r.url}")
            for key, vals in links.items():
                if vals:
                    print(f"  {key}: {len(vals)}")
                    for v in vals[:5]:
                        print(f"    {v}")
    elif args.all:
        results = pull_all(args.output)
        (args.output / "_summary.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
    elif args.release:
        rel = next((r for r in KNOWN_RELEASES if r.id == args.release), None)
        if not rel:
            parser.error(f"Unknown release id: {args.release}")
        r = pull_release(rel, args.output)
        print(json.dumps(r, indent=2, default=str))
    else:
        parser.error("Specify --release <id>, --all, or --scan-only")
