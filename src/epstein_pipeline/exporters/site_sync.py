"""Export pipeline data to the epstein-index site format.

Generates JSON data files and optionally seeds a SQLite database
matching the site's expected schema (see scripts/seed-sqlite.mjs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.document import Document, Person
from epstein_pipeline.models.forensics import (
    ExtractedEntity,
    ExtractedImage,
    RecoveredText,
    RedactionScore,
    Transcript,
)

logger = logging.getLogger(__name__)
console = Console()


class SiteSyncer:
    """Sync pipeline output to the epstein-index site."""

    def __init__(self, site_dir: Path) -> None:
        self.site_dir = Path(site_dir)
        self.data_dir = self.site_dir / "data"

        if not self.site_dir.exists():
            raise FileNotFoundError(f"Site directory not found: {self.site_dir}")

    def export_json(
        self,
        documents: list[Document],
        *,
        output_name: str = "pipeline-documents.json",
    ) -> Path:
        """Export documents as a JSON file in the site's data directory.

        The site expects JSON arrays of document objects that can be imported
        via ``import docs from "@/data/pipeline-documents.json"``.
        """
        out_path = self.data_dir / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = [doc.model_dump(exclude_none=True, exclude={"ocrText"}) for doc in documents]

        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        console.print(
            f"  [green]Exported {len(documents):,} documents"
            f" to {out_path}[/green] ({size_mb:.1f} MB)"
        )
        return out_path

    def export_sqlite(
        self,
        documents: list[Document],
        persons: list[Person],
        *,
        db_name: str = "epstein.sqlite",
        redaction_scores: list[RedactionScore] | None = None,
        recovered_texts: list[RecoveredText] | None = None,
        transcripts: list[Transcript] | None = None,
        entities: list[ExtractedEntity] | None = None,
        images: list[ExtractedImage] | None = None,
    ) -> Path:
        """Export to a SQLite database matching the site's schema."""
        from epstein_pipeline.exporters.sqlite_export import SqliteExporter

        db_path = self.data_dir / db_name
        exporter = SqliteExporter()
        return exporter.export(
            documents=documents,
            persons=persons,
            db_path=db_path,
            redaction_scores=redaction_scores,
            recovered_texts=recovered_texts,
            transcripts=transcripts,
            entities=entities,
            images=images,
        )

    def sync(
        self,
        documents: list[Document],
        persons: list[Person],
        *,
        output_sqlite: bool = True,
        output_json: bool = True,
        redaction_scores: list[RedactionScore] | None = None,
        recovered_texts: list[RecoveredText] | None = None,
        transcripts: list[Transcript] | None = None,
        entities: list[ExtractedEntity] | None = None,
        images: list[ExtractedImage] | None = None,
    ) -> None:
        """Full sync: export both JSON and SQLite to the site."""
        console.print()
        console.rule("[bold cyan]Site Sync[/bold cyan]")
        console.print(f"  Site directory: {self.site_dir}")
        console.print(f"  Documents: {len(documents):,}")
        console.print(f"  Persons: {len(persons):,}")
        console.print()

        if output_json:
            self.export_json(documents)

        if output_sqlite:
            self.export_sqlite(
                documents=documents,
                persons=persons,
                redaction_scores=redaction_scores,
                recovered_texts=recovered_texts,
                transcripts=transcripts,
                entities=entities,
                images=images,
            )

        console.print()
        console.rule("[bold green]Sync Complete[/bold green]")
