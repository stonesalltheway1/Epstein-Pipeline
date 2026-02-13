"""Kaggle dataset downloader for epstein-ranker and related datasets."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from rich.console import Console


class KaggleDownloader:
    """Download the epstein-ranker dataset from Kaggle.

    Requires the ``kaggle`` CLI to be installed and configured with an API
    token (``~/.kaggle/kaggle.json``).

    The primary dataset is ``jamesgrantz/epstein-ranker`` which contains
    ~23,700 documents with AI-generated summaries and pre-computed person
    linkages.
    """

    DATASET_SLUG = "jamesgrantz/epstein-ranker"

    def __init__(self) -> None:
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, output_dir: Path) -> Path:
        """Download the epstein-ranker dataset from Kaggle.

        Parameters
        ----------
        output_dir:
            Directory where the dataset will be saved and extracted.

        Returns
        -------
        Path
            The directory containing the extracted dataset files.

        Raises
        ------
        RuntimeError
            If the kaggle CLI is not installed or the download fails.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if kaggle CLI is available
        if not self._check_kaggle_cli():
            raise RuntimeError(
                "The kaggle CLI is not installed or not on PATH.\n\n"
                "Install it with:\n"
                "  pip install kaggle\n\n"
                "Then configure your API token:\n"
                "  1. Go to https://www.kaggle.com/settings\n"
                "  2. Click 'Create New Token' under the API section\n"
                "  3. Save the downloaded kaggle.json to:\n"
                "     - Linux/Mac: ~/.kaggle/kaggle.json\n"
                "     - Windows:   %USERPROFILE%\\.kaggle\\kaggle.json\n"
                "  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)\n"
            )

        self._console.print(f"[cyan]Downloading dataset:[/cyan] [bold]{self.DATASET_SLUG}[/bold]")
        self._console.print(f"[cyan]Output directory:[/cyan] {output_dir.resolve()}")
        self._console.print()

        # Run kaggle download
        try:
            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.DATASET_SLUG,
                    "-p",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                self._console.print(result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr.strip() if exc.stderr else str(exc)

            if "403" in error_msg or "Forbidden" in error_msg:
                raise RuntimeError(
                    "Kaggle API returned 403 Forbidden. Check that:\n"
                    "  1. Your kaggle.json API token is valid\n"
                    "  2. You have accepted the dataset's terms on kaggle.com\n"
                    f"  3. The dataset '{self.DATASET_SLUG}' is accessible\n\n"
                    f"Error details: {error_msg}"
                ) from exc

            raise RuntimeError(f"Kaggle download failed: {error_msg}") from exc

        # Extract the ZIP if present
        zip_name = self.DATASET_SLUG.split("/")[-1] + ".zip"
        zip_path = output_dir / zip_name
        extract_dir = output_dir / "epstein-ranker"

        if zip_path.exists():
            self._console.print(f"[cyan]Extracting:[/cyan] {zip_path.name}")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            self._console.print(f"[green]Extracted to:[/green] {extract_dir.resolve()}")

            # Count extracted files
            file_count = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
            self._console.print(f"[green]Files extracted:[/green] {file_count:,}")

            # Remove the ZIP to save disk space
            zip_path.unlink()
            self._console.print("[dim]Removed ZIP archive to save space.[/dim]")
        else:
            extract_dir = output_dir
            self._console.print(
                "[yellow]No ZIP file found; dataset may have been "
                "downloaded as individual files.[/yellow]"
            )

        self._console.print()
        self._console.print("[bold green]Download complete.[/bold green]")
        return extract_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_kaggle_cli(self) -> bool:
        """Return True if the kaggle CLI is installed and on PATH."""
        return shutil.which("kaggle") is not None
