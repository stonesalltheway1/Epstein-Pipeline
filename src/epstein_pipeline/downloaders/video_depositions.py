"""Video deposition downloader for Epstein case materials.

Downloads and catalogs video depositions from known public sources:
  - DOJ Dataset 10 (seized device media, already on disk)
  - Archive.org (House Oversight depositions, C-SPAN recordings)
  - justice.gov (Maxwell prison interview audio)
  - Court exhibit videos

Usage:
    epstein-pipeline download depositions --output E:/epstein-video-depositions/raw
    epstein-pipeline download depositions --source archive
    epstein-pipeline download depositions --source doj-interview
    epstein-pipeline download depositions --catalog-ds10 E:/epstein-ds10/extracted
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()

MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv", ".webm", ".wav", ".mp3", ".m4a", ".flac"}


@dataclass(frozen=True)
class DepositionSource:
    """Metadata for a known video deposition source."""

    id: str
    title: str
    deponent: str  # Person name
    case: str
    date: str  # ISO format
    source_type: str  # "archive", "doj", "cspan", "justice-gov", "ds10"
    url: str | None = None
    archive_id: str | None = None  # Archive.org identifier
    files: list[str] = field(default_factory=list)  # Specific filenames to download
    person_id: str | None = None  # Pipeline person registry ID if known
    description: str = ""


# ──────────────────────────────────────────────────────────────
# Known deposition sources — curated registry
# ──────────────────────────────────────────────────────────────
KNOWN_DEPOSITIONS: list[DepositionSource] = [
    # ── House Oversight Committee depositions ──
    DepositionSource(
        id="vd-maxwell-oversight-2026",
        title="Ghislaine Maxwell — House Oversight Committee Virtual Deposition",
        deponent="Ghislaine Maxwell",
        case="House Oversight Committee Investigation",
        date="2026-02-14",
        source_type="archive",
        archive_id="CSPAN_20260214_203900_Ghislaine_Maxwell_Virtual_Deposition_Before_House_Oversight_Committee",
        person_id="p-0002",
        description="Maxwell invoked the Fifth Amendment throughout the virtual deposition.",
    ),
    DepositionSource(
        id="vd-clinton-bill-oversight-2026",
        title="Bill Clinton — House Oversight Committee Deposition",
        deponent="Bill Clinton",
        case="House Oversight Committee Investigation",
        date="2026-03-02",
        source_type="archive",
        archive_id="CSPAN_20260302_Bill_Clinton_Deposition",
        person_id="p-0029",
        description="Former President Clinton deposed by the House Oversight Committee.",
    ),
    DepositionSource(
        id="vd-clinton-hillary-oversight-2026",
        title="Hillary Clinton — House Oversight Committee Deposition",
        deponent="Hillary Clinton",
        case="House Oversight Committee Investigation",
        date="2026-03-02",
        source_type="archive",
        archive_id="CSPAN_20260302_Hillary_Clinton_Deposition",
        description="Former Secretary of State Clinton deposed by the House Oversight Committee.",
    ),
    DepositionSource(
        id="vd-indyke-oversight-2026",
        title="Darren Indyke — House Oversight Committee Deposition",
        deponent="Darren Indyke",
        case="House Oversight Committee Investigation",
        date="2026-03-24",
        source_type="archive",
        description="Epstein estate co-executor deposed by House Oversight Committee.",
    ),
    DepositionSource(
        id="vd-kahn-oversight-2026",
        title="Richard Kahn — House Oversight Committee Deposition",
        deponent="Richard Kahn",
        case="House Oversight Committee Investigation",
        date="2026-03-24",
        source_type="archive",
        description="Epstein estate co-executor deposed by House Oversight Committee.",
    ),

    # ── DOJ / justice.gov releases ──
    DepositionSource(
        id="vd-maxwell-interview-2025",
        title="Ghislaine Maxwell — DOJ Prison Interview (Todd Blanche)",
        deponent="Ghislaine Maxwell",
        case="DOJ Investigation",
        date="2025-07-01",
        source_type="justice-gov",
        url="https://www.justice.gov/maxwell-interview",
        person_id="p-0002",
        files=[
            "maxwell-interview-day1-part1.wav",
            "maxwell-interview-day1-part2.wav",
            "maxwell-interview-day1-part3.wav",
            "maxwell-interview-day1-part4.wav",
            "maxwell-interview-day1-part5.wav",
            "maxwell-interview-day1-part6.wav",
            "maxwell-interview-day2-part1.wav",
            "maxwell-interview-day2-part2.wav",
            "maxwell-interview-day2-part3.wav",
            "maxwell-interview-day2-part4.wav",
            "maxwell-interview-day2-part5.wav",
        ],
        description="11 WAV audio files. 2-day interview by AG Todd Blanche at federal prison.",
    ),

    # ── Civil case depositions (Giuffre v. Maxwell, 2016) ──
    DepositionSource(
        id="vd-maxwell-civil-2016",
        title="Ghislaine Maxwell — Giuffre v. Maxwell Civil Deposition",
        deponent="Ghislaine Maxwell",
        case="Giuffre v. Maxwell (15-cv-07433)",
        date="2016-04-22",
        source_type="archive",
        person_id="p-0002",
        description="Video deposition discovered in DOJ Dataset 10 by ITV News (Feb 2026). Maxwell fought to suppress for years.",
    ),

    # ── SEC deposition ──
    DepositionSource(
        id="vd-epstein-sec-2010",
        title="Jeffrey Epstein — SEC Deposition",
        deponent="Jeffrey Epstein",
        case="SEC Investigation",
        date="2010-01-01",
        source_type="archive",
        person_id="p-0001",
        description="Epstein pleaded the Fifth throughout. Transcript and exhibits at sec.gov.",
    ),
]


def list_sources() -> None:
    """Print a table of all known deposition sources."""
    table = Table(title="Known Video Deposition Sources", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Deponent", style="bold")
    table.add_column("Case")
    table.add_column("Date")
    table.add_column("Type")
    table.add_column("Status")

    for dep in KNOWN_DEPOSITIONS:
        status = "[green]URL available[/green]" if dep.url or dep.archive_id else "[yellow]Manual[/yellow]"
        table.add_row(dep.id, dep.deponent, dep.case, dep.date, dep.source_type, status)

    console.print(table)
    console.print(f"\n  [bold]{len(KNOWN_DEPOSITIONS)}[/bold] known deposition sources")


def catalog_ds10_media(ds10_path: Path, output_path: Path) -> list[dict]:
    """Scan DS10 extracted directory for video/audio files and create a catalog.

    Args:
        ds10_path: Path to E:/epstein-ds10/extracted/
        output_path: Path to write catalog JSON

    Returns:
        List of media file metadata dicts
    """
    console.print(f"  Scanning DS10 at [bold]{ds10_path}[/bold] for media files...")

    catalog: list[dict] = []
    for path in sorted(ds10_path.rglob("*")):
        if path.suffix.lower() in MEDIA_EXTENSIONS and path.is_file():
            stat = path.stat()
            efta_id = path.stem  # e.g., "EFTA01600796"
            catalog.append({
                "id": f"ds10-{efta_id}",
                "efta": efta_id,
                "filename": path.name,
                "path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "source": "ds10",
                "title": f"DS10 Media: {efta_id}",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    console.print(f"  [green]Found {len(catalog)} media files[/green]")

    # Summary by extension
    by_ext: dict[str, int] = {}
    for item in catalog:
        ext = item["extension"]
        by_ext[ext] = by_ext.get(ext, 0) + 1
    for ext, count in sorted(by_ext.items()):
        console.print(f"    {ext}: {count}")

    total_size = sum(item["size_bytes"] for item in catalog)
    console.print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")

    return catalog


def download_archive_org(
    identifier: str,
    output_dir: Path,
    media_only: bool = True,
) -> list[Path]:
    """Download files from an Archive.org item.

    Args:
        identifier: Archive.org item identifier
        output_dir: Directory to save files
        media_only: If True, only download video/audio files

    Returns:
        List of downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://archive.org/metadata/{identifier}"

    console.print(f"  Fetching metadata for [bold]{identifier}[/bold]...")

    with httpx.Client(timeout=30) as client:
        resp = client.get(base_url)
        if resp.status_code != 200:
            console.print(f"  [red]Failed to fetch metadata: {resp.status_code}[/red]")
            return []

        metadata = resp.json()
        files = metadata.get("files", [])

        # Filter to media files if requested
        if media_only:
            files = [
                f for f in files
                if any(f["name"].lower().endswith(ext) for ext in MEDIA_EXTENSIONS)
            ]

        if not files:
            console.print(f"  [yellow]No media files found in {identifier}[/yellow]")
            return []

        console.print(f"  Found {len(files)} files to download")

        downloaded: list[Path] = []
        download_base = f"https://archive.org/download/{identifier}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            for file_info in files:
                name = file_info["name"]
                size = int(file_info.get("size", 0))
                out_path = output_dir / name

                if out_path.exists() and out_path.stat().st_size == size:
                    console.print(f"    [dim]Skipping {name} (already downloaded)[/dim]")
                    downloaded.append(out_path)
                    continue

                task = progress.add_task(f"  {name[:50]}", total=size or None)
                url = f"{download_base}/{name}"

                try:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in response.iter_bytes(chunk_size=65536):
                                f.write(chunk)
                                progress.advance(task, len(chunk))
                    downloaded.append(out_path)
                except Exception as exc:
                    console.print(f"    [red]Failed: {name} — {exc}[/red]")
                    if out_path.exists():
                        out_path.unlink()

                time.sleep(0.5)  # Rate limiting

    console.print(f"  [green]Downloaded {len(downloaded)} files[/green]")
    return downloaded


def download_justice_gov(
    source: DepositionSource,
    output_dir: Path,
) -> list[Path]:
    """Download files from justice.gov (Maxwell interview WAV files)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    if not source.url or not source.files:
        console.print("  [yellow]No URL or file list for this source[/yellow]")
        return downloaded

    base_url = source.url.rstrip("/")

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("Downloading", total=len(source.files))
            for filename in source.files:
                out_path = output_dir / filename
                if out_path.exists():
                    console.print(f"    [dim]Skipping {filename} (exists)[/dim]")
                    downloaded.append(out_path)
                    progress.advance(task)
                    continue

                url = f"{base_url}/{filename}"
                progress.update(task, description=f"  {filename[:50]}")

                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    out_path.write_bytes(resp.content)
                    downloaded.append(out_path)
                except Exception as exc:
                    console.print(f"    [red]Failed: {filename} — {exc}[/red]")

                progress.advance(task)
                time.sleep(1)

    console.print(f"  [green]Downloaded {len(downloaded)} files[/green]")
    return downloaded


def download_depositions(
    output_dir: Path,
    source_filter: str | None = None,
    source_id: str | None = None,
    ds10_path: Path | None = None,
) -> dict:
    """Main entry point: download video depositions from known sources.

    Args:
        output_dir: Base output directory (e.g., E:/epstein-video-depositions/raw)
        source_filter: Filter by source_type ("archive", "justice-gov", "ds10")
        source_id: Download a specific deposition by ID
        ds10_path: Path to DS10 extracted directory (for catalog-only mode)

    Returns:
        Summary dict with counts
    """
    console.print("\n[bold]Video Deposition Downloader[/bold]\n")

    results = {
        "downloaded": 0,
        "cataloged": 0,
        "skipped": 0,
        "errors": 0,
    }

    # DS10 catalog mode
    if ds10_path:
        catalog_path = output_dir / "ds10-media-catalog.json"
        catalog = catalog_ds10_media(ds10_path, catalog_path)
        results["cataloged"] = len(catalog)
        return results

    # Filter sources
    sources = KNOWN_DEPOSITIONS
    if source_id:
        sources = [s for s in sources if s.id == source_id]
    elif source_filter:
        sources = [s for s in sources if s.source_type == source_filter]

    if not sources:
        console.print("[yellow]No matching sources found.[/yellow]")
        list_sources()
        return results

    console.print(f"  Processing {len(sources)} deposition sources...\n")

    for source in sources:
        console.print(f"\n[bold cyan]{source.title}[/bold cyan]")
        console.print(f"  Deponent: {source.deponent}")
        console.print(f"  Case: {source.case}")
        console.print(f"  Date: {source.date}")

        source_dir = output_dir / source.id

        if source.source_type == "archive" and source.archive_id:
            files = download_archive_org(source.archive_id, source_dir)
            results["downloaded"] += len(files)

        elif source.source_type == "justice-gov":
            files = download_justice_gov(source, source_dir)
            results["downloaded"] += len(files)

        else:
            console.print(f"  [yellow]Source type '{source.source_type}' requires manual download[/yellow]")
            results["skipped"] += 1

        # Save source metadata
        meta_path = source_dir / "metadata.json"
        source_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps({
                "id": source.id,
                "title": source.title,
                "deponent": source.deponent,
                "case": source.case,
                "date": source.date,
                "source_type": source.source_type,
                "person_id": source.person_id,
                "description": source.description,
            }, indent=2),
            encoding="utf-8",
        )

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Downloaded: {results['downloaded']} files")
    console.print(f"  Cataloged: {results['cataloged']} files")
    console.print(f"  Skipped: {results['skipped']} sources")

    return results
