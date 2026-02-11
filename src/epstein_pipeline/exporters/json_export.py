"""JSON exporter for documents in the Epstein Pipeline.

Exports documents to JSON format, either as a generic dump or in the exact
structure expected by the Epstein Exposed main site (``data/*.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.document import Document


class JsonExporter:
    """Export documents to JSON format."""

    def __init__(self) -> None:
        self._console = Console()

    def export(
        self,
        documents: list[Document],
        output_dir: Path,
        filename: str = "documents.json",
    ) -> Path:
        """Export documents to a single JSON file.

        Uses Pydantic's ``model_dump(by_alias=True)`` to produce JSON keys
        matching the TypeScript interfaces on the main site (e.g.
        ``personIds`` instead of ``person_ids``).

        Parameters
        ----------
        documents:
            List of Document models to export.
        output_dir:
            Directory to write the JSON file.
        filename:
            Output filename (default ``"documents.json"``).

        Returns
        -------
        Path
            The path to the written JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename

        serialized = [doc.model_dump(by_alias=True, exclude_none=True) for doc in documents]

        out_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        self._console.print(
            f"[green]Exported {len(documents):,} documents to "
            f"{out_path.resolve()} ({size_mb:.1f} MB)[/green]"
        )
        return out_path

    def export_for_site(
        self,
        documents: list[Document],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Export in the exact format the Epstein Exposed main site expects.

        The main site organises documents into separate JSON files by source:

        - ``data/kaggle-documents.json`` -- Kaggle-sourced docs
        - ``data/hf-efta-documents.json`` -- HuggingFace EFTA docs
        - ``data/epstein-docs-documents.json`` -- epstein-docs.github.io docs
        - ``data/efta-documents.json`` -- Base EFTA OCR docs
        - A fallback file for anything that doesn't match a known source

        Documents are grouped by their ``source`` and ``tags`` fields.
        The OCR text field is excluded from site exports to keep file sizes
        manageable (OCR text is stored separately in ``ocr-text.json``).

        Parameters
        ----------
        documents:
            List of Document models to export.
        output_dir:
            The ``data/`` directory of the main site repository.

        Returns
        -------
        dict[str, Path]
            Mapping of filename to written path for each output file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Classify documents into site buckets
        buckets: dict[str, list[Document]] = {
            "kaggle-documents.json": [],
            "hf-efta-documents.json": [],
            "epstein-docs-documents.json": [],
            "efta-documents.json": [],
            "pipeline-documents.json": [],
        }

        for doc in documents:
            tags_lower = {t.lower() for t in doc.tags}

            if "kaggle" in tags_lower or "epstein-ranker" in tags_lower:
                buckets["kaggle-documents.json"].append(doc)
            elif "huggingface" in tags_lower or "hf-efta" in tags_lower:
                buckets["hf-efta-documents.json"].append(doc)
            elif "epstein-docs" in tags_lower:
                buckets["epstein-docs-documents.json"].append(doc)
            elif doc.source == "efta" or (
                doc.batesRange and doc.batesRange.startswith("EFTA")
            ):
                buckets["efta-documents.json"].append(doc)
            else:
                buckets["pipeline-documents.json"].append(doc)

        # Write each non-empty bucket
        written: dict[str, Path] = {}

        for filename, docs in buckets.items():
            if not docs:
                continue

            serialized = [
                doc.model_dump(
                    by_alias=True,
                    exclude_none=True,
                    exclude={"ocrText"},  # OCR text stored separately
                )
                for doc in docs
            ]

            out_path = output_dir / filename
            out_path.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            size_mb = out_path.stat().st_size / (1024 * 1024)
            self._console.print(
                f"[green]{filename}:[/green] {len(docs):,} documents ({size_mb:.1f} MB)"
            )
            written[filename] = out_path

        # Write OCR text separately for documents that have it
        docs_with_ocr = [doc for doc in documents if doc.ocrText]
        if docs_with_ocr:
            ocr_data = {doc.id: doc.ocrText for doc in docs_with_ocr}
            ocr_path = output_dir / "ocr-text.json"
            ocr_path.write_text(
                json.dumps(ocr_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            size_mb = ocr_path.stat().st_size / (1024 * 1024)
            self._console.print(
                f"[green]ocr-text.json:[/green] {len(docs_with_ocr):,} "
                f"documents with OCR text ({size_mb:.1f} MB)"
            )
            written["ocr-text.json"] = ocr_path

        self._console.print()
        self._console.print(
            f"[bold green]Exported {len(documents):,} documents "
            f"across {len(written)} files.[/bold green]"
        )
        return written
