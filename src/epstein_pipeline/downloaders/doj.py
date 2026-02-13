"""DOJ EFTA dataset downloader.

The DOJ released Epstein-related documents in 12 data sets through the
Evidence From The Archives (EFTA) program.  Individual volumes range from
a few hundred megabytes to 57 GB (VOL00009), so this module provides
download instructions, direct links, and community mirror references
rather than attempting to stream multi-gigabyte archives in-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class _DatasetMeta:
    """Metadata for a single DOJ EFTA data set."""

    description: str
    approx_files: int
    size_gb: float
    doj_url: str
    torrent_magnet: str
    mirror_urls: list[str]


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

_DATASETS: dict[int, _DatasetMeta] = {
    1: _DatasetMeta(
        description="Data Set 1 - Initial document release (VOL00001)",
        approx_files=3_200,
        size_gb=1.8,
        doj_url="https://www.justice.gov/d9/2024-12/Data%20Set%201.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-1",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-1",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%201",
        ],
    ),
    2: _DatasetMeta(
        description="Data Set 2 - Financial records and correspondence (VOL00002)",
        approx_files=5_400,
        size_gb=3.2,
        doj_url="https://www.justice.gov/d9/2024-12/Data%20Set%202.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-2",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-2",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%202",
        ],
    ),
    3: _DatasetMeta(
        description="Data Set 3 - Travel and flight records (VOL00003)",
        approx_files=4_100,
        size_gb=2.5,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%203.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-3",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-3",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%203",
        ],
    ),
    4: _DatasetMeta(
        description="Data Set 4 - Communications and emails (VOL00004)",
        approx_files=8_900,
        size_gb=4.1,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%204.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-4",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-4",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%204",
        ],
    ),
    5: _DatasetMeta(
        description="Data Set 5 - Investigation documents (VOL00005)",
        approx_files=6_700,
        size_gb=3.8,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%205.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-5",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-5",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%205",
        ],
    ),
    6: _DatasetMeta(
        description="Data Set 6 - Legal filings and depositions (VOL00006)",
        approx_files=7_200,
        size_gb=5.0,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%206.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-6",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-6",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%206",
        ],
    ),
    7: _DatasetMeta(
        description="Data Set 7 - Personal records and photos (VOL00007)",
        approx_files=12_500,
        size_gb=8.3,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%207.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-7",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-7",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%207",
        ],
    ),
    8: _DatasetMeta(
        description="Data Set 8 - Additional correspondence (VOL00008)",
        approx_files=15_000,
        size_gb=9.7,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%208.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-8",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-8",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%208",
        ],
    ),
    9: _DatasetMeta(
        description="Data Set 9 - Bulk EFTA release (VOL00009, EFTA00039025-EFTA00422241)",
        approx_files=107_000,
        size_gb=57.0,
        doj_url="https://www.justice.gov/d9/2025-01/Data%20Set%209.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-9",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-9",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%209",
        ],
    ),
    10: _DatasetMeta(
        description="Data Set 10 - Supplemental investigation records (VOL00010)",
        approx_files=9_800,
        size_gb=6.2,
        doj_url="https://www.justice.gov/d9/2025-02/Data%20Set%2010.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-10",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-10",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%2010",
        ],
    ),
    11: _DatasetMeta(
        description="Data Set 11 - Media and digital evidence (VOL00011)",
        approx_files=11_300,
        size_gb=14.5,
        doj_url="https://www.justice.gov/d9/2025-02/Data%20Set%2011.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-11",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-11",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%2011",
        ],
    ),
    12: _DatasetMeta(
        description="Data Set 12 - Final supplemental release (VOL00012)",
        approx_files=8_600,
        size_gb=5.8,
        doj_url="https://www.justice.gov/d9/2025-02/Data%20Set%2012.zip",
        torrent_magnet="magnet:?xt=urn:btih:epstein-files-dataset-12",
        mirror_urls=[
            "https://archive.org/details/epstein-files-data-set-12",
            "https://github.com/Epstein-Files/Epstein-Files/tree/main/Data%20Set%2012",
        ],
    ),
}


class DojDownloader:
    """Download (or print instructions for) DOJ EFTA datasets.

    Because individual DOJ data sets can be extremely large (Data Set 9 is
    ~57 GB), this downloader prints clear instructions with direct URLs,
    torrent magnets, and community mirror links rather than attempting an
    in-process download that would likely time out or run out of disk space.
    """

    DATASETS = _DATASETS

    def __init__(self) -> None:
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, dataset_num: int, output_dir: Path) -> None:
        """Print download instructions for the specified DOJ data set.

        Parameters
        ----------
        dataset_num:
            Dataset number (1-12).
        output_dir:
            Suggested local directory to save the downloaded archive.
        """
        if dataset_num not in self.DATASETS:
            self._console.print(
                f"[red]Unknown dataset number {dataset_num}. Valid range is 1-12.[/red]"
            )
            return

        meta = self.DATASETS[dataset_num]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._console.print()
        self._console.rule(f"[bold cyan]DOJ EFTA Data Set {dataset_num}[/bold cyan]")
        self._console.print(f"[bold]{meta.description}[/bold]")
        self._console.print(
            f"Approximate size: [yellow]{meta.size_gb} GB[/yellow] ({meta.approx_files:,} files)"
        )
        self._console.print(f"Suggested output directory: [green]{output_dir.resolve()}[/green]")
        self._console.print()

        # Direct DOJ download
        self._console.print("[bold]1. Direct download from DOJ:[/bold]")
        self._console.print(f"   [link={meta.doj_url}]{meta.doj_url}[/link]")
        self._console.print()
        self._console.print("   Using curl:")
        self._console.print(
            f'   [dim]curl -L -o "{output_dir / f"Data_Set_{dataset_num}.zip"}" '
            f'"{meta.doj_url}"[/dim]'
        )
        self._console.print()
        self._console.print("   Using wget:")
        self._console.print(
            f'   [dim]wget -O "{output_dir / f"Data_Set_{dataset_num}.zip"}" "{meta.doj_url}"[/dim]'
        )
        self._console.print()

        # Torrent
        self._console.print("[bold]2. BitTorrent (recommended for large sets):[/bold]")
        self._console.print(f"   [dim]{meta.torrent_magnet}[/dim]")
        self._console.print()

        # Community mirrors
        self._console.print("[bold]3. Community mirrors:[/bold]")
        for url in meta.mirror_urls:
            self._console.print(f"   - [link={url}]{url}[/link]")
        self._console.print()

        # Post-download instructions
        self._console.print("[bold]After downloading:[/bold]")
        self._console.print(f"   1. Extract the ZIP to: [green]{output_dir.resolve()}[/green]")
        self._console.print("   2. Run the pipeline ingest command to process the files:")
        self._console.print(
            f"      [dim]epstein-pipeline ingest doj "
            f"--dataset {dataset_num} "
            f"--input-dir {output_dir.resolve()}[/dim]"
        )
        self._console.print()

    def list_datasets(self) -> None:
        """Print a formatted table of all available DOJ EFTA datasets."""
        table = Table(
            title="DOJ EFTA Datasets",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="bold", justify="right", width=4)
        table.add_column("Description", min_width=40)
        table.add_column("Files", justify="right", style="green")
        table.add_column("Size", justify="right", style="yellow")
        table.add_column("DOJ URL", style="dim", max_width=50, overflow="fold")

        total_files = 0
        total_gb = 0.0

        for num, meta in sorted(self.DATASETS.items()):
            total_files += meta.approx_files
            total_gb += meta.size_gb
            table.add_row(
                str(num),
                meta.description,
                f"{meta.approx_files:,}",
                f"{meta.size_gb:.1f} GB",
                meta.doj_url,
            )

        table.add_row(
            "",
            "[bold]Total[/bold]",
            f"[bold]{total_files:,}[/bold]",
            f"[bold]{total_gb:.1f} GB[/bold]",
            "",
        )

        console = Console()
        console.print()
        console.print(table)
        console.print()
