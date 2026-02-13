"""HuggingFace dataset downloader for Epstein-related datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table


@dataclass(frozen=True)
class _HFDataset:
    """Metadata for a known HuggingFace dataset."""

    repo_id: str
    description: str
    format: str  # "parquet", "json", "csv", etc.
    approx_size_mb: float


_KNOWN_DATASETS: list[_HFDataset] = [
    _HFDataset(
        repo_id="qanon-research/epstein-emails",
        description="Structured email corpus from Epstein case files (5,258 emails)",
        format="parquet",
        approx_size_mb=12.0,
    ),
    _HFDataset(
        repo_id="qanon-research/epstein-documents",
        description="EFTA document metadata and text extracts",
        format="parquet",
        approx_size_mb=45.0,
    ),
    _HFDataset(
        repo_id="qanon-research/epstein-flight-logs",
        description="Parsed flight log entries from Lolita Express manifests",
        format="parquet",
        approx_size_mb=2.5,
    ),
    _HFDataset(
        repo_id="qanon-research/epstein-black-book",
        description="Contact entries from the Epstein/Maxwell black book",
        format="parquet",
        approx_size_mb=1.0,
    ),
]


class HuggingFaceDownloader:
    """Download Epstein-related datasets from HuggingFace Hub.

    Prefers the ``huggingface_hub`` library for authenticated downloads
    with resume support.  Falls back to direct HTTP downloads via ``httpx``
    if ``huggingface_hub`` is not installed.
    """

    DATASETS = _KNOWN_DATASETS

    def __init__(self) -> None:
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, dataset_name: str, output_dir: Path) -> Path:
        """Download a HuggingFace dataset to *output_dir*.

        Parameters
        ----------
        dataset_name:
            Either a full repo ID (e.g. ``"qanon-research/epstein-emails"``)
            or a short name (e.g. ``"epstein-emails"``).  Short names are
            resolved against the known datasets list.
        output_dir:
            Directory to save downloaded files.

        Returns
        -------
        Path
            The directory containing the downloaded dataset files.
        """
        repo_id = self._resolve_dataset_name(dataset_name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._console.print(f"[cyan]Downloading dataset:[/cyan] [bold]{repo_id}[/bold]")
        self._console.print(f"[cyan]Output directory:[/cyan] {output_dir.resolve()}")
        self._console.print()

        # Try huggingface_hub first, fall back to httpx
        try:
            return self._download_with_hf_hub(repo_id, output_dir)
        except ImportError:
            self._console.print(
                "[yellow]huggingface_hub not installed, falling back to "
                "direct HTTP download.[/yellow]"
            )
            self._console.print(
                "[dim]Install huggingface_hub for better download support: "
                "pip install huggingface_hub[/dim]"
            )
            self._console.print()
            return self._download_with_httpx(repo_id, output_dir)

    def list_datasets(self) -> None:
        """Print a formatted table of known HuggingFace datasets."""
        table = Table(
            title="Known HuggingFace Epstein Datasets",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Repository", style="bold", min_width=35)
        table.add_column("Description", min_width=40)
        table.add_column("Format", justify="center", style="green")
        table.add_column("Size", justify="right", style="yellow")

        for ds in self.DATASETS:
            table.add_row(
                ds.repo_id,
                ds.description,
                ds.format,
                f"{ds.approx_size_mb:.1f} MB",
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        self._console.print("[dim]To download: epstein-pipeline download hf <repo-id>[/dim]")
        self._console.print()

    # ------------------------------------------------------------------
    # Download strategies
    # ------------------------------------------------------------------

    def _download_with_hf_hub(self, repo_id: str, output_dir: Path) -> Path:
        """Download using the huggingface_hub library (preferred)."""
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

        self._console.print("[cyan]Using huggingface_hub for download...[/cyan]")
        local_dir = output_dir / repo_id.split("/")[-1]

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )

        file_count = sum(1 for f in local_dir.rglob("*") if f.is_file())
        self._console.print(
            f"[bold green]Downloaded {file_count:,} files to {local_dir.resolve()}[/bold green]"
        )
        return local_dir

    def _download_with_httpx(self, repo_id: str, output_dir: Path) -> Path:
        """Download via the HuggingFace HTTP API as a fallback."""
        local_dir = output_dir / repo_id.split("/")[-1]
        local_dir.mkdir(parents=True, exist_ok=True)

        # List files in the dataset repo via the API
        api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
        self._console.print(f"[dim]Listing files from {api_url}[/dim]")

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(api_url)

            if response.status_code == 404:
                raise RuntimeError(
                    f"Dataset '{repo_id}' not found on HuggingFace. "
                    f"Check the repository ID and try again."
                )

            response.raise_for_status()
            files = response.json()

            if not files:
                raise RuntimeError(f"Dataset '{repo_id}' appears to be empty.")

            self._console.print(f"[cyan]Found {len(files)} file(s) to download.[/cyan]")

            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=self._console,
            ) as progress:
                for file_info in files:
                    rfilename = file_info.get("path", file_info.get("rfilename", ""))
                    if not rfilename:
                        continue

                    # Skip directories
                    if file_info.get("type") == "directory":
                        continue

                    file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rfilename}"
                    dest = local_dir / rfilename
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    # Stream download with progress
                    task_id = progress.add_task(
                        "download",
                        filename=rfilename,
                        total=None,
                    )

                    with client.stream("GET", file_url) as stream:
                        stream.raise_for_status()
                        total = stream.headers.get("content-length")
                        if total is not None:
                            progress.update(task_id, total=int(total))

                        with open(dest, "wb") as f:
                            for chunk in stream.iter_bytes(chunk_size=65_536):
                                f.write(chunk)
                                progress.advance(task_id, len(chunk))

        file_count = sum(1 for f in local_dir.rglob("*") if f.is_file())
        self._console.print()
        self._console.print(
            f"[bold green]Downloaded {file_count:,} files to {local_dir.resolve()}[/bold green]"
        )
        return local_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_name(self, name: str) -> str:
        """Resolve a short name or full repo ID to a full repo ID.

        If *name* contains a ``/``, it is assumed to be a full repo ID.
        Otherwise, we search the known datasets for a matching suffix.
        """
        if "/" in name:
            return name

        for ds in self.DATASETS:
            short = ds.repo_id.split("/")[-1]
            if short == name or short.replace("-", "") == name.replace("-", ""):
                return ds.repo_id

        # Not found in known list; assume it's a valid repo path under a
        # default namespace.
        self._console.print(
            f"[yellow]'{name}' not in known datasets list. Trying as full repo ID...[/yellow]"
        )
        return name
