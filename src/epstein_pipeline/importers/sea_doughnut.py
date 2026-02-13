"""Import data from Sea_Doughnut's SQLite databases.

Sea_Doughnut (rhowardstone/Epstein-research-data) provides 5 SQLite databases
totalling ~6.1GB with 1.38M documents, 638K redaction scores, 39.5K recovered
text pages, 38.9K images, 1,530 transcripts, and 107K extracted entities.

Expected directory layout::

    data-dir/
        full_text_corpus.db          (6.1 GB - documents, transcripts, entities)
        redaction_analysis_v2.db     (redaction scores + recovered text)
        image_analysis.db            (extracted images metadata)
        ocr_database.db              (OCR text fallback)
        persons_registry.json        (1,536 persons)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from epstein_pipeline.models.forensics import (
    ExtractedEntity,
    ExtractedImage,
    RecoveredText,
    RedactionScore,
    SeaDoughnutCorpus,
    Transcript,
)

logger = logging.getLogger(__name__)
console = Console()


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


class SeaDoughnutImporter:
    """Import data from Sea_Doughnut's research databases."""

    # Database filenames we look for
    CORPUS_DB = "full_text_corpus.db"
    REDACTION_DB = "redaction_analysis_v2.db"
    IMAGE_DB = "image_analysis.db"
    OCR_DB = "ocr_database.db"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _open_db(self, filename: str) -> sqlite3.Connection | None:
        """Open a SQLite database if it exists."""
        db_path = self.data_dir / filename
        if not db_path.exists():
            console.print(f"  [yellow]Database not found: {db_path}[/yellow]")
            return None
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def import_documents(
        self,
        output_dir: Path | None = None,
        limit: int | None = None,
    ) -> int:
        """Import documents from full_text_corpus.db.

        Returns the number of documents imported.  If *output_dir* is given,
        writes NDJSON chunks (one JSON object per line, 10K docs per file).
        """
        conn = self._open_db(self.CORPUS_DB)
        if conn is None:
            return 0

        try:
            # Discover the schema
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            console.print(f"  [dim]Tables in corpus DB: {', '.join(tables)}[/dim]")

            # Try common table names
            doc_table = None
            for candidate in ["documents", "corpus", "full_text", "entries"]:
                if candidate in tables:
                    doc_table = candidate
                    break

            if doc_table is None:
                # Use the first non-system table
                user_tables = [t for t in tables if not t.startswith("sqlite_")]
                if user_tables:
                    doc_table = user_tables[0]
                else:
                    console.print("  [red]No document table found[/red]")
                    return 0

            # Count rows
            count_result = conn.execute(f"SELECT COUNT(*) FROM [{doc_table}]").fetchone()
            total = count_result[0] if count_result else 0
            console.print(f"  Found [bold]{total:,}[/bold] rows in [{doc_table}]")

            if limit:
                total = min(total, limit)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

            # Get column names
            col_cursor = conn.execute(f"PRAGMA table_info([{doc_table}])")
            columns = {row[1] for row in col_cursor.fetchall()}

            # Stream rows
            query = f"SELECT * FROM [{doc_table}]"
            if limit:
                query += f" LIMIT {limit}"

            imported = 0
            chunk_size = 10_000
            current_chunk: list[dict] = []
            chunk_num = 0

            with _progress() as progress:
                task = progress.add_task("Importing documents", total=total)

                for row in conn.execute(query):
                    row_dict = dict(row)

                    # Normalize to our Document schema
                    doc_data = self._normalize_document_row(row_dict, columns)
                    current_chunk.append(doc_data)
                    imported += 1

                    if output_dir and len(current_chunk) >= chunk_size:
                        self._write_chunk(output_dir, chunk_num, current_chunk)
                        chunk_num += 1
                        current_chunk = []

                    progress.advance(task)

            # Write remaining chunk
            if output_dir and current_chunk:
                self._write_chunk(output_dir, chunk_num, current_chunk)

            console.print(f"  [green]Imported {imported:,} documents[/green]")
            return imported

        finally:
            conn.close()

    def _normalize_document_row(self, row: dict, columns: set[str]) -> dict:
        """Normalize a Sea_Doughnut row into our document format."""
        # Map common column names
        doc_id = (
            row.get("doc_id")
            or row.get("document_id")
            or row.get("id")
            or row.get("bates_number")
            or f"sd-{hash(str(row)) & 0xFFFFFFFF:08x}"
        )

        title = row.get("title") or row.get("filename") or row.get("file_name") or str(doc_id)

        text = (
            row.get("text")
            or row.get("full_text")
            or row.get("content")
            or row.get("ocr_text")
            or ""
        )

        # Determine source from dataset info
        dataset = row.get("dataset", row.get("data_set", ""))
        source = "efta"
        if dataset:
            ds_str = str(dataset).lower()
            for i in range(1, 13):
                if f"ds{i}" in ds_str or f"data set {i}" in ds_str or ds_str == str(i):
                    source = f"efta-ds{i}"
                    break

        return {
            "id": str(doc_id),
            "title": str(title)[:500],
            "source": source,
            "category": "other",
            "ocrText": str(text) if text else None,
            "tags": ["sea-doughnut"],
            "batesRange": row.get("bates_range") or row.get("bates_number"),
            "pageCount": row.get("page_count") or row.get("num_pages"),
        }

    def _write_chunk(self, output_dir: Path, chunk_num: int, data: list[dict]) -> None:
        """Write a chunk of documents as JSON."""
        path = output_dir / f"documents_{chunk_num:04d}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Redaction scores
    # ------------------------------------------------------------------

    def import_redaction_scores(self) -> list[RedactionScore]:
        """Import redaction analysis scores from redaction_analysis_v2.db."""
        conn = self._open_db(self.REDACTION_DB)
        if conn is None:
            return []

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Find the scores table
            score_table = None
            for candidate in ["redaction_scores", "scores", "analysis", "redactions"]:
                if candidate in tables:
                    score_table = candidate
                    break
            if score_table is None:
                user_tables = [t for t in tables if not t.startswith("sqlite_")]
                score_table = user_tables[0] if user_tables else None

            if not score_table:
                return []

            count = conn.execute(f"SELECT COUNT(*) FROM [{score_table}]").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] redaction scores")

            scores: list[RedactionScore] = []
            with _progress() as progress:
                task = progress.add_task("Importing redaction scores", total=count)
                for row in conn.execute(f"SELECT * FROM [{score_table}]"):
                    row_dict = dict(row)
                    scores.append(
                        RedactionScore(
                            document_id=str(
                                row_dict.get("doc_id")
                                or row_dict.get("document_id")
                                or row_dict.get("id", "")
                            ),
                            total_redactions=int(row_dict.get("total_redactions", 0)),
                            proper_redactions=int(row_dict.get("proper_redactions", 0)),
                            improper_redactions=int(row_dict.get("improper_redactions", 0)),
                            redaction_density=float(row_dict.get("redaction_density", 0)),
                            page_count=row_dict.get("page_count"),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(scores):,} redaction scores[/green]")
            return scores

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Recovered text
    # ------------------------------------------------------------------

    def import_recovered_text(self) -> list[RecoveredText]:
        """Import text recovered from under redactions."""
        conn = self._open_db(self.REDACTION_DB)
        if conn is None:
            return []

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            text_table = None
            for candidate in ["recovered_text", "recovered", "text_under_redactions"]:
                if candidate in tables:
                    text_table = candidate
                    break

            if not text_table:
                console.print("  [yellow]No recovered text table found[/yellow]")
                return []

            count = conn.execute(f"SELECT COUNT(*) FROM [{text_table}]").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] recovered text entries")

            results: list[RecoveredText] = []
            with _progress() as progress:
                task = progress.add_task("Importing recovered text", total=count)
                for row in conn.execute(f"SELECT * FROM [{text_table}]"):
                    row_dict = dict(row)
                    results.append(
                        RecoveredText(
                            document_id=str(
                                row_dict.get("doc_id")
                                or row_dict.get("document_id")
                                or row_dict.get("id", "")
                            ),
                            page_number=int(row_dict.get("page_number", row_dict.get("page", 0))),
                            text=str(row_dict.get("text", row_dict.get("recovered_text", ""))),
                            confidence=float(row_dict.get("confidence", 0)),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} recovered text pages[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def import_images(self) -> list[ExtractedImage]:
        """Import extracted image metadata from image_analysis.db."""
        conn = self._open_db(self.IMAGE_DB)
        if conn is None:
            return []

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            img_table = None
            for candidate in ["images", "extracted_images", "image_catalog"]:
                if candidate in tables:
                    img_table = candidate
                    break
            if not img_table:
                user_tables = [t for t in tables if not t.startswith("sqlite_")]
                img_table = user_tables[0] if user_tables else None

            if not img_table:
                return []

            count = conn.execute(f"SELECT COUNT(*) FROM [{img_table}]").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] images")

            results: list[ExtractedImage] = []
            with _progress() as progress:
                task = progress.add_task("Importing images", total=count)
                for row in conn.execute(f"SELECT * FROM [{img_table}]"):
                    row_dict = dict(row)
                    results.append(
                        ExtractedImage(
                            document_id=str(
                                row_dict.get("doc_id")
                                or row_dict.get("document_id")
                                or row_dict.get("id", "")
                            ),
                            page_number=int(row_dict.get("page_number", row_dict.get("page", 0))),
                            image_index=int(row_dict.get("image_index", row_dict.get("idx", 0))),
                            width=int(row_dict.get("width", 0)),
                            height=int(row_dict.get("height", 0)),
                            format=str(row_dict.get("format", row_dict.get("ext", "png"))),
                            file_path=row_dict.get("file_path") or row_dict.get("path"),
                            description=row_dict.get("description") or row_dict.get("caption"),
                            size_bytes=int(row_dict.get("size_bytes", row_dict.get("size", 0))),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} images[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Transcripts
    # ------------------------------------------------------------------

    def import_transcripts(self) -> list[Transcript]:
        """Import transcripts from full_text_corpus.db."""
        conn = self._open_db(self.CORPUS_DB)
        if conn is None:
            return []

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            tx_table = None
            for candidate in ["transcripts", "transcript", "audio_transcripts"]:
                if candidate in tables:
                    tx_table = candidate
                    break

            if not tx_table:
                console.print("  [yellow]No transcripts table found[/yellow]")
                return []

            count = conn.execute(f"SELECT COUNT(*) FROM [{tx_table}]").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] transcripts")

            results: list[Transcript] = []
            with _progress() as progress:
                task = progress.add_task("Importing transcripts", total=count)
                for row in conn.execute(f"SELECT * FROM [{tx_table}]"):
                    row_dict = dict(row)
                    results.append(
                        Transcript(
                            source_path=str(
                                row_dict.get("source_path", row_dict.get("file_path", ""))
                            ),
                            document_id=str(
                                row_dict.get("doc_id")
                                or row_dict.get("document_id")
                                or row_dict.get("id", "")
                            ),
                            text=str(row_dict.get("text", row_dict.get("transcript", ""))),
                            language=str(row_dict.get("language", "en")),
                            duration_seconds=float(
                                row_dict.get("duration", row_dict.get("duration_seconds", 0))
                            ),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} transcripts[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def import_entities(self) -> list[ExtractedEntity]:
        """Import extracted entities from full_text_corpus.db."""
        conn = self._open_db(self.CORPUS_DB)
        if conn is None:
            return []

        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            ent_table = None
            for candidate in ["entities", "extracted_entities", "named_entities"]:
                if candidate in tables:
                    ent_table = candidate
                    break

            if not ent_table:
                console.print("  [yellow]No entities table found[/yellow]")
                return []

            count = conn.execute(f"SELECT COUNT(*) FROM [{ent_table}]").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] entities")

            results: list[ExtractedEntity] = []
            with _progress() as progress:
                task = progress.add_task("Importing entities", total=count)
                for row in conn.execute(f"SELECT * FROM [{ent_table}]"):
                    row_dict = dict(row)
                    results.append(
                        ExtractedEntity(
                            document_id=str(
                                row_dict.get("doc_id")
                                or row_dict.get("document_id")
                                or row_dict.get("id", "")
                            ),
                            entity_type=str(
                                row_dict.get("entity_type", row_dict.get("type", "UNKNOWN"))
                            ),
                            text=str(row_dict.get("text", row_dict.get("entity_text", ""))),
                            confidence=float(row_dict.get("confidence", 0)),
                            person_id=row_dict.get("person_id"),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} entities[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Import all
    # ------------------------------------------------------------------

    def import_all(self, output_dir: Path | None = None) -> SeaDoughnutCorpus:
        """Import all data from Sea_Doughnut's databases.

        Returns a SeaDoughnutCorpus container with all imported data.
        """
        console.print()
        console.rule("[bold cyan]Sea_Doughnut Data Import[/bold cyan]")
        console.print(f"  Data directory: {self.data_dir}")
        console.print()

        doc_count = self.import_documents(
            output_dir=output_dir / "documents" if output_dir else None
        )

        console.print()
        redaction_scores = self.import_redaction_scores()

        console.print()
        recovered_texts = self.import_recovered_text()

        console.print()
        images = self.import_images()

        console.print()
        transcripts = self.import_transcripts()

        console.print()
        entities = self.import_entities()

        corpus = SeaDoughnutCorpus(
            document_count=doc_count,
            redaction_scores=redaction_scores,
            recovered_texts=recovered_texts,
            images=images,
            transcripts=transcripts,
            entities=entities,
        )

        console.print()
        console.rule("[bold green]Import Complete[/bold green]")
        console.print(f"  Documents:        {doc_count:,}")
        console.print(f"  Redaction scores: {len(redaction_scores):,}")
        console.print(f"  Recovered texts:  {len(recovered_texts):,}")
        console.print(f"  Images:           {len(images):,}")
        console.print(f"  Transcripts:      {len(transcripts):,}")
        console.print(f"  Entities:         {len(entities):,}")

        return corpus
