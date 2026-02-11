"""CSV exporter for documents in the Epstein Pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.document import Document

# Column definitions for the CSV export.
_CSV_COLUMNS = [
    "id",
    "title",
    "date",
    "source",
    "category",
    "summary",
    "person_count",
    "page_count",
    "bates_range",
    "pdf_url",
    "tags",
]


class CsvExporter:
    """Export documents to CSV format."""

    def __init__(self) -> None:
        self._console = Console()

    def export(self, documents: list[Document], output_path: Path) -> Path:
        """Export documents to a CSV file.

        Parameters
        ----------
        documents:
            List of Document models to export.
        output_path:
            Full path for the output CSV file.  Parent directories are
            created automatically if they do not exist.

        Returns
        -------
        Path
            The path to the written CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_CSV_COLUMNS,
                extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()

            for doc in documents:
                writer.writerow(self._document_to_row(doc))

        size_kb = output_path.stat().st_size / 1024
        self._console.print(
            f"[green]Exported {len(documents):,} documents to "
            f"{output_path.resolve()} ({size_kb:.1f} KB)[/green]"
        )
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _document_to_row(doc: Document) -> dict[str, str]:
        """Convert a Document model to a flat CSV row dict."""
        return {
            "id": doc.id,
            "title": doc.title,
            "date": doc.date or "",
            "source": doc.source,
            "category": doc.category,
            "summary": doc.summary or "",
            "person_count": str(len(doc.personIds)),
            "page_count": str(doc.pageCount) if doc.pageCount is not None else "",
            "bates_range": doc.batesRange or "",
            "pdf_url": doc.pdfUrl or "",
            "tags": "; ".join(doc.tags),
        }
