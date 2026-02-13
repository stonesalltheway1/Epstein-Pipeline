"""Archive.org collection downloader for Epstein-related materials."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Archive.org API base URLs
# ---------------------------------------------------------------------------
_SEARCH_URL = "https://archive.org/advancedsearch.php"
_METADATA_URL = "https://archive.org/metadata"
_DOWNLOAD_URL = "https://archive.org/download"


@dataclass(frozen=True)
class _Collection:
    """Metadata for a known Archive.org collection."""

    identifier: str
    description: str
    media_types: list[str] = field(default_factory=lambda: ["texts"])


_KNOWN_COLLECTIONS: list[_Collection] = [
    _Collection(
        identifier="epstein-files",
        description="Epstein Files - DOJ EFTA document releases",
        media_types=["texts", "data"],
    ),
    _Collection(
        identifier="jeffrey-epstein-court-documents",
        description="Jeffrey Epstein court documents and filings",
        media_types=["texts"],
    ),
    _Collection(
        identifier="epstein-maxwell-documents",
        description="Ghislaine Maxwell trial documents and exhibits",
        media_types=["texts"],
    ),
    _Collection(
        identifier="jeffrey-epstein-photos",
        description="FBI raid photos and property images",
        media_types=["image"],
    ),
    _Collection(
        identifier="epstein-flight-logs",
        description="Lolita Express flight log scans and transcriptions",
        media_types=["texts", "image"],
    ),
]


@dataclass
class ArchiveItem:
    """A single item from an Archive.org collection."""

    identifier: str
    title: str
    media_type: str
    files: list[dict]
    size_bytes: int = 0


class ArchiveDownloader:
    """Download items and files from Archive.org collections.

    Uses the Archive.org Advanced Search API to enumerate items in a
    collection, then downloads individual files with progress tracking.
    """

    COLLECTIONS = _KNOWN_COLLECTIONS

    def __init__(self, rate_limit_delay: float = 1.0) -> None:
        """
        Parameters
        ----------
        rate_limit_delay:
            Seconds to wait between API requests to avoid rate limiting.
        """
        self._console = Console()
        self._rate_limit_delay = rate_limit_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        collection: str,
        output_dir: Path,
        media_types: list[str] | None = None,
        max_items: int | None = None,
        file_extensions: list[str] | None = None,
    ) -> Path:
        """Download items from an Archive.org collection.

        Parameters
        ----------
        collection:
            Archive.org collection identifier (e.g. ``"epstein-files"``).
        output_dir:
            Directory to save downloaded files.
        media_types:
            Filter items by media type (e.g. ``["texts", "image"]``).
            If ``None``, downloads all types.
        max_items:
            Maximum number of items to download.  ``None`` means all.
        file_extensions:
            Only download files with these extensions (e.g. ``[".pdf", ".jpg"]``).
            If ``None``, downloads all files.

        Returns
        -------
        Path
            The directory containing downloaded files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._console.print(f"[cyan]Collection:[/cyan] [bold]{collection}[/bold]")
        self._console.print(f"[cyan]Output directory:[/cyan] {output_dir.resolve()}")
        if media_types:
            self._console.print(f"[cyan]Media types:[/cyan] {', '.join(media_types)}")
        if file_extensions:
            self._console.print(f"[cyan]File extensions:[/cyan] {', '.join(file_extensions)}")
        self._console.print()

        # Enumerate items in the collection
        items = self._search_collection(collection, media_types, max_items)

        if not items:
            self._console.print(f"[yellow]No items found in collection '{collection}'.[/yellow]")
            return output_dir

        self._console.print(f"[green]Found {len(items):,} item(s) to process.[/green]")
        self._console.print()

        # Download files from each item
        total_files = 0
        total_bytes = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self._console,
        ) as progress:
            item_task = progress.add_task("Downloading items...", total=len(items))

            for item in items:
                item_dir = output_dir / item.identifier
                item_dir.mkdir(parents=True, exist_ok=True)

                files = self._get_item_files(item.identifier)

                # Filter by extension if requested
                if file_extensions:
                    ext_set = {e.lower().lstrip(".") for e in file_extensions}
                    files = [
                        f
                        for f in files
                        if any(f.get("name", "").lower().endswith(f".{ext}") for ext in ext_set)
                    ]

                for file_meta in files:
                    filename = file_meta.get("name", "")
                    if not filename:
                        continue

                    # Skip metadata/derivative files unless explicitly wanted
                    source = file_meta.get("source", "original")
                    if source == "metadata" and file_extensions is None:
                        continue

                    dest = item_dir / filename
                    if dest.exists():
                        file_size = dest.stat().st_size
                        expected = int(file_meta.get("size", 0))
                        if expected > 0 and file_size == expected:
                            total_files += 1
                            total_bytes += file_size
                            continue

                    file_url = f"{_DOWNLOAD_URL}/{item.identifier}/{filename}"
                    downloaded = self._download_file(file_url, dest)
                    if downloaded > 0:
                        total_files += 1
                        total_bytes += downloaded

                    time.sleep(self._rate_limit_delay * 0.2)

                progress.advance(item_task)
                time.sleep(self._rate_limit_delay)

        self._console.print()
        size_mb = total_bytes / (1024 * 1024)
        self._console.print(
            f"[bold green]Downloaded {total_files:,} files "
            f"({size_mb:.1f} MB) to {output_dir.resolve()}[/bold green]"
        )
        return output_dir

    def list_collections(self) -> None:
        """Print a formatted table of known Archive.org collections."""
        table = Table(
            title="Known Archive.org Epstein Collections",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Identifier", style="bold", min_width=30)
        table.add_column("Description", min_width=40)
        table.add_column("Media Types", style="green")
        table.add_column("URL", style="dim", max_width=50, overflow="fold")

        for coll in self.COLLECTIONS:
            table.add_row(
                coll.identifier,
                coll.description,
                ", ".join(coll.media_types),
                f"https://archive.org/details/{coll.identifier}",
            )

        self._console.print()
        self._console.print(table)
        self._console.print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_collection(
        self,
        collection: str,
        media_types: list[str] | None = None,
        max_items: int | None = None,
    ) -> list[ArchiveItem]:
        """Search Archive.org for items in a collection."""
        query_parts = [f"collection:{collection}"]
        if media_types:
            type_clause = " OR ".join(f"mediatype:{mt}" for mt in media_types)
            query_parts.append(f"({type_clause})")

        query = " AND ".join(query_parts)
        rows = min(max_items, 500) if max_items else 500
        items: list[ArchiveItem] = []
        page = 1

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while True:
                params = {
                    "q": query,
                    "fl[]": ["identifier", "title", "mediatype"],
                    "sort[]": "addeddate desc",
                    "rows": str(rows),
                    "page": str(page),
                    "output": "json",
                }

                self._console.print(f"[dim]Searching page {page}...[/dim]")

                resp = client.get(_SEARCH_URL, params=params)
                resp.raise_for_status()

                data = resp.json()
                docs = data.get("response", {}).get("docs", [])

                if not docs:
                    break

                for doc in docs:
                    items.append(
                        ArchiveItem(
                            identifier=doc.get("identifier", ""),
                            title=doc.get("title", ""),
                            media_type=doc.get("mediatype", ""),
                            files=[],
                        )
                    )

                    if max_items and len(items) >= max_items:
                        return items

                num_found = data.get("response", {}).get("numFound", 0)
                if page * rows >= num_found:
                    break

                page += 1
                time.sleep(self._rate_limit_delay)

        return items

    def _get_item_files(self, identifier: str) -> list[dict]:
        """Fetch the file list for a single Archive.org item."""
        url = f"{_METADATA_URL}/{identifier}/files"

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()

            data = resp.json()
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            if isinstance(data, list):
                return data
            return []

    def _download_file(self, url: str, dest: Path) -> int:
        """Download a single file with resume support.

        Returns the number of bytes written, or 0 on failure.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        headers: dict[str, str] = {}
        mode = "wb"
        existing_size = 0

        # Resume partial downloads
        if dest.exists():
            existing_size = dest.stat().st_size
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as stream:
                    if stream.status_code == 416:
                        # Range not satisfiable = file already complete
                        return existing_size

                    stream.raise_for_status()

                    with open(dest, mode) as f:
                        written = existing_size
                        for chunk in stream.iter_bytes(chunk_size=65_536):
                            f.write(chunk)
                            written += len(chunk)

                    return written

        except httpx.HTTPError as exc:
            self._console.print(f"[red]Failed to download {dest.name}: {exc}[/red]")
            return 0
