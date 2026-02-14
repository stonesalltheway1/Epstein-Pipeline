"""Import data from Sea_Doughnut's SQLite databases.

Sea_Doughnut (rhowardstone/Epstein-research-data) provides processed research
databases covering all 12 DOJ EFTA dataset releases:

- full_text_corpus.db   ~6.1 GB  1,380,941 docs, 2,731,825 pages, FTS5 index
- transcripts.db        ~50 MB   1,530 entries, 375 with speech (92K words)
- redaction_analysis_v2.db       849,655 docs, 2,587,102 redactions
- concordance_metadata.db        OPT/DAT, SDNY_GM bridge, provenance map
- persons_registry.json          1,538 persons, 203 with aliases

Expected directory layout::

    data-dir/
        full_text_corpus.db
        transcripts.db               (separate from corpus; NOT inside it)
        redaction_analysis_v2.db
        concordance_metadata.db      (optional — provenance + SDNY bridge)
        persons_registry.json        (optional — 1,538 persons)
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
    ConcordanceSummary,
    ExtractedEntity,
    ExtractedImage,
    ProvenanceRange,
    RecoveredText,
    RedactionScore,
    SeaDoughnutCorpus,
    Transcript,
)

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# EFTA → Dataset mapping (verified from DOJ filesystem)
# ---------------------------------------------------------------------------

DATASET_RANGES: list[tuple[int, int, int]] = [
    # (dataset, efta_start, efta_end)
    (1,  1, 3158),
    (2,  3159, 3857),
    (3,  3858, 5586),
    (4,  5705, 8320),
    (5,  8409, 8528),
    (6,  8529, 8998),
    (7,  9016, 9664),
    (8,  9676, 39023),
    (9,  39025, 1262781),
    (10, 1262782, 2212882),
    (11, 2212883, 2730262),
    (12, 2730265, 2731783),
]

# Provenance category → DocumentCategory mapping
_CATEGORY_MAP: dict[str, str] = {
    "prosecution": "legal",
    "prosecution_admin": "legal",
    "financial_subpoena": "financial",
    "telecom_subpoena": "communications",
    "device_extraction": "communications",
    "investigation": "investigation",
    "mixed_prosecution": "legal",
}


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def efta_to_dataset(efta_num: int) -> int | None:
    """Map an EFTA number to its DOJ dataset."""
    for ds, start, end in DATASET_RANGES:
        if start <= efta_num <= end:
            return ds
    # Fall back: nearest lower dataset
    for ds, start, _ in reversed(DATASET_RANGES):
        if efta_num >= start:
            return ds
    return None


def efta_to_doj_url(efta_number: str, dataset: int | None = None) -> str | None:
    """Generate the DOJ PDF URL for an EFTA document."""
    efta_num = int(efta_number[4:]) if efta_number.startswith("EFTA") else None
    if efta_num is None:
        return None
    ds = dataset or efta_to_dataset(efta_num)
    if ds is None:
        return None
    return f"https://www.justice.gov/epstein/files/DataSet%20{ds}/{efta_number}.pdf"


class SeaDoughnutImporter:
    """Import data from Sea_Doughnut's research databases.

    This importer reads the exact schemas produced by the Sea_Doughnut
    processing pipeline, rather than guessing column names.
    """

    CORPUS_DB = "full_text_corpus.db"
    TRANSCRIPTS_DB = "transcripts.db"
    REDACTION_DB = "redaction_analysis_v2.db"
    CONCORDANCE_DB = "concordance_metadata.db"
    PERSONS_JSON = "persons_registry.json"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load provenance map if concordance DB exists
        self._provenance: list[dict] | None = None
        self._load_provenance()

    def _open_db(self, filename: str) -> sqlite3.Connection | None:
        """Open a SQLite database if it exists."""
        db_path = self.data_dir / filename
        if not db_path.exists():
            console.print(f"  [yellow]Not found: {db_path}[/yellow]")
            return None
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA cache_size = -500000")  # 500MB cache
        return conn

    def _load_provenance(self) -> None:
        """Load provenance map from concordance_metadata.db if available."""
        conn = self._open_db(self.CONCORDANCE_DB)
        if conn is None:
            return
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "provenance_map" not in tables:
                return

            self._provenance = []
            for row in conn.execute(
                "SELECT efta_start_num, efta_end_num, source_category, "
                "source_description FROM provenance_map ORDER BY efta_start_num"
            ):
                self._provenance.append({
                    "start": row[0], "end": row[1],
                    "category": row[2], "description": row[3],
                })
            console.print(
                f"  [dim]Loaded {len(self._provenance)} provenance ranges[/dim]"
            )
        finally:
            conn.close()

    def _categorize(self, efta_num: int) -> tuple[str, str | None]:
        """Return (DocumentCategory, description) from provenance map."""
        if self._provenance:
            for p in self._provenance:
                if p["start"] <= efta_num <= p["end"]:
                    cat = _CATEGORY_MAP.get(p["category"], "other")
                    return cat, p["description"]
        return "other", None

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def import_documents(
        self,
        output_dir: Path | None = None,
        limit: int | None = None,
    ) -> int:
        """Import documents from full_text_corpus.db.

        Schema::

            documents(efta_number, dataset, total_pages, file_size, ...)
            pages(efta_number, page_number, text_content, char_count)

        Returns the number of documents imported.
        """
        conn = self._open_db(self.CORPUS_DB)
        if conn is None:
            return 0

        try:
            total_q = "SELECT COUNT(*) FROM documents"
            total = conn.execute(total_q).fetchone()[0]
            console.print(f"  Corpus: [bold]{total:,}[/bold] documents")

            if limit:
                total = min(total, limit)
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

            imported = 0
            chunk_size = 10_000
            current_chunk: list[dict] = []
            chunk_num = 0
            batch_size = 500  # docs per page-text batch

            with _progress() as progress:
                task = progress.add_task("Importing documents", total=total)

                # Stream documents in order
                doc_query = "SELECT efta_number, dataset, total_pages, file_size FROM documents ORDER BY efta_number"
                if limit:
                    doc_query += f" LIMIT {limit}"

                doc_cursor = conn.execute(doc_query)
                doc_batch: list[tuple] = []

                while True:
                    row = doc_cursor.fetchone()
                    if row is not None:
                        doc_batch.append(tuple(row))

                    # Process batch when full or at end
                    if len(doc_batch) >= batch_size or (row is None and doc_batch):
                        # Fetch page text for this batch
                        efta_list = [d[0] for d in doc_batch]
                        placeholders = ",".join("?" * len(efta_list))
                        page_rows = conn.execute(
                            f"SELECT efta_number, page_number, text_content "
                            f"FROM pages WHERE efta_number IN ({placeholders}) "
                            f"ORDER BY efta_number, page_number",
                            efta_list,
                        ).fetchall()

                        # Group pages by efta_number
                        pages_by_efta: dict[str, list[str]] = {}
                        for pr in page_rows:
                            efta = pr[0]
                            text = pr[2] or ""
                            pages_by_efta.setdefault(efta, []).append(text)

                        for efta_number, dataset, total_pages, _ in doc_batch:
                            page_texts = pages_by_efta.get(efta_number, [])
                            full_text = "\n".join(page_texts) if page_texts else None

                            efta_num = int(efta_number[4:])
                            ds = dataset or efta_to_dataset(efta_num)
                            category, prov_desc = self._categorize(efta_num)

                            bates_end_num = efta_num + (total_pages or 1) - 1
                            bates_range = (
                                f"{efta_number}-EFTA{bates_end_num:08d}"
                                if total_pages and total_pages > 1
                                else efta_number
                            )

                            doc_data = {
                                "id": efta_number,
                                "title": efta_number,
                                "source": f"efta-ds{ds}" if ds else "efta",
                                "category": category,
                                "ocrText": full_text,
                                "tags": ["sea-doughnut"],
                                "batesRange": bates_range,
                                "pageCount": total_pages,
                                "pdfUrl": efta_to_doj_url(efta_number, ds),
                            }

                            if prov_desc:
                                doc_data["summary"] = prov_desc

                            current_chunk.append(doc_data)
                            imported += 1
                            progress.advance(task)

                            if output_dir and len(current_chunk) >= chunk_size:
                                self._write_chunk(output_dir, chunk_num, current_chunk)
                                chunk_num += 1
                                current_chunk = []

                        doc_batch = []

                    if row is None:
                        break

            if output_dir and current_chunk:
                self._write_chunk(output_dir, chunk_num, current_chunk)

            console.print(f"  [green]Imported {imported:,} documents[/green]")
            return imported

        finally:
            conn.close()

    def _write_chunk(self, output_dir: Path, chunk_num: int, data: list[dict]) -> None:
        """Write a chunk of documents as NDJSON (one JSON object per line)."""
        path = output_dir / f"documents_{chunk_num:04d}.ndjson"
        with open(path, "w", encoding="utf-8") as f:
            for doc in data:
                f.write(json.dumps(doc, ensure_ascii=False))
                f.write("\n")

    # ------------------------------------------------------------------
    # Redaction scores
    # ------------------------------------------------------------------

    def import_redaction_scores(self) -> list[RedactionScore]:
        """Import redaction analysis from redaction_analysis_v2.db.

        Schema::

            document_summary(efta_number, total_redactions, bad_redactions,
                             proper_redactions, has_recoverable_text)
        """
        conn = self._open_db(self.REDACTION_DB)
        if conn is None:
            return []

        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "document_summary" not in tables:
                console.print("  [yellow]No document_summary table found[/yellow]")
                return []

            count = conn.execute("SELECT COUNT(*) FROM document_summary").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] redaction summaries")

            scores: list[RedactionScore] = []
            with _progress() as progress:
                task = progress.add_task("Importing redaction scores", total=count)
                for row in conn.execute("SELECT * FROM document_summary"):
                    rd = dict(row)
                    total_r = int(rd.get("total_redactions", 0))
                    bad_r = int(rd.get("bad_redactions", 0))
                    proper_r = int(rd.get("proper_redactions", 0))

                    scores.append(
                        RedactionScore(
                            document_id=str(rd.get("efta_number", "")),
                            total_redactions=total_r,
                            proper_redactions=proper_r,
                            improper_redactions=bad_r,
                            redaction_density=0.0,  # not in our schema
                            page_count=None,
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
        """Import text recovered from under redactions.

        Schema::

            redactions(efta_number, page_number, hidden_text, confidence,
                       redaction_type)
        """
        conn = self._open_db(self.REDACTION_DB)
        if conn is None:
            return []

        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM redactions "
                "WHERE hidden_text IS NOT NULL AND LENGTH(hidden_text) > 3"
            ).fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] recovered text entries")

            results: list[RecoveredText] = []
            with _progress() as progress:
                task = progress.add_task("Importing recovered text", total=count)
                for row in conn.execute(
                    "SELECT efta_number, page_number, hidden_text, confidence "
                    "FROM redactions "
                    "WHERE hidden_text IS NOT NULL AND LENGTH(hidden_text) > 3"
                ):
                    results.append(
                        RecoveredText(
                            document_id=str(row[0]),
                            page_number=int(row[1] or 0),
                            text=str(row[2]),
                            confidence=float(row[3] or 0),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} recovered text entries[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Transcripts (separate transcripts.db)
    # ------------------------------------------------------------------

    def import_transcripts(self) -> list[Transcript]:
        """Import transcripts from transcripts.db (separate file).

        Schema::

            transcripts(efta_number, file_path, file_type, duration_secs,
                        language, transcript, word_count, dataset_source)
        """
        conn = self._open_db(self.TRANSCRIPTS_DB)
        if conn is None:
            # Fall back to looking in corpus DB
            conn = self._open_db(self.CORPUS_DB)
            if conn is None:
                return []
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "transcripts" not in tables:
                conn.close()
                console.print("  [yellow]No transcripts table found[/yellow]")
                return []

        try:
            count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
            console.print(f"  Found [bold]{count:,}[/bold] transcripts")

            results: list[Transcript] = []
            with _progress() as progress:
                task = progress.add_task("Importing transcripts", total=count)
                for row in conn.execute("SELECT * FROM transcripts"):
                    rd = dict(row)
                    transcript_text = rd.get("transcript", "") or ""
                    if not transcript_text.strip():
                        progress.advance(task)
                        continue

                    results.append(
                        Transcript(
                            source_path=str(rd.get("file_path", "")),
                            document_id=str(rd.get("efta_number", "")),
                            text=transcript_text,
                            language=str(rd.get("language", "en")),
                            duration_seconds=float(rd.get("duration_secs", 0) or 0),
                        )
                    )
                    progress.advance(task)

            console.print(f"  [green]Imported {len(results):,} transcripts with content[/green]")
            return results

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Images (not in our current pipeline, but kept for compatibility)
    # ------------------------------------------------------------------

    def import_images(self) -> list[ExtractedImage]:
        """Import extracted image metadata if available."""
        # Our pipeline stores extracted images on disk, not in a DB.
        # Return empty list — images can be processed from PDFs directly.
        console.print("  [dim]Image metadata: not in DB (images are on-disk)[/dim]")
        return []

    # ------------------------------------------------------------------
    # Entities (from persons_registry.json cross-referenced with corpus)
    # ------------------------------------------------------------------

    def import_entities(self) -> list[ExtractedEntity]:
        """Import entity data.

        We don't have a pre-extracted entities table. The Pipeline's own
        entity extraction processor should be run after import.
        """
        console.print(
            "  [dim]Entities: use 'extract-entities' after import "
            "(no pre-extracted entity table)[/dim]"
        )
        return []

    # ------------------------------------------------------------------
    # Persons registry
    # ------------------------------------------------------------------

    def import_persons(self) -> list[dict]:
        """Import persons from persons_registry.json.

        Our format::

            [{"name": "...", "aliases": [...], "description": "...",
              "source": "...", "category": "..."}, ...]

        Converts to Pipeline format (id, slug, name, aliases, category).
        """
        json_path = self.data_dir / self.PERSONS_JSON
        if not json_path.exists():
            console.print(f"  [yellow]Not found: {json_path}[/yellow]")
            return []

        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        persons = []
        for i, entry in enumerate(raw):
            name = entry.get("name", "")
            if not name:
                continue
            slug = name.lower().replace(" ", "-").replace(".", "").replace(",", "")
            person: dict = {
                "id": f"p-{i:04d}",
                "slug": slug,
                "name": name,
                "aliases": entry.get("aliases", []),
                "category": entry.get("category", "unknown"),
            }
            desc = entry.get("description")
            if desc:
                person["shortBio"] = desc
            persons.append(person)

        console.print(f"  [green]Loaded {len(persons):,} persons from registry[/green]")
        return persons

    # ------------------------------------------------------------------
    # Concordance / Provenance
    # ------------------------------------------------------------------

    def import_concordance_summary(self) -> ConcordanceSummary | None:
        """Import concordance metadata summary.

        Returns a ConcordanceSummary with provenance ranges and bridge stats.
        """
        conn = self._open_db(self.CONCORDANCE_DB)
        if conn is None:
            return None

        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            provenance_ranges: list[ProvenanceRange] = []
            sdny_bridge_count = 0
            production_count = 0
            opt_document_count = 0

            if "provenance_map" in tables:
                for row in conn.execute(
                    "SELECT dataset, efta_start, efta_end, sdny_gm_start, sdny_gm_end, "
                    "source_description, source_category, doc_count, page_count, confidence "
                    "FROM provenance_map ORDER BY dataset, efta_start_num"
                ):
                    rd = dict(row)
                    provenance_ranges.append(ProvenanceRange(
                        dataset=int(rd["dataset"]),
                        efta_start=str(rd["efta_start"]),
                        efta_end=str(rd["efta_end"]),
                        source_description=str(rd.get("source_description", "")),
                        source_category=str(rd.get("source_category", "")),
                        doc_count=int(rd.get("doc_count", 0) or 0),
                        page_count=int(rd.get("page_count", 0) or 0),
                        sdny_gm_start=rd.get("sdny_gm_start"),
                        sdny_gm_end=rd.get("sdny_gm_end"),
                        confidence=str(rd.get("confidence", "high")),
                    ))

            if "sdny_efta_bridge" in tables:
                sdny_bridge_count = conn.execute(
                    "SELECT COUNT(*) FROM sdny_efta_bridge"
                ).fetchone()[0]

            if "productions" in tables:
                production_count = conn.execute(
                    "SELECT COUNT(*) FROM productions"
                ).fetchone()[0]

            if "opt_documents" in tables:
                opt_document_count = conn.execute(
                    "SELECT COUNT(*) FROM opt_documents"
                ).fetchone()[0]

            summary = ConcordanceSummary(
                provenance_ranges=provenance_ranges,
                sdny_bridge_count=sdny_bridge_count,
                production_count=production_count,
                opt_document_count=opt_document_count,
            )

            console.print(
                f"  [green]Concordance: {len(tables)} tables, "
                f"{len(provenance_ranges)} provenance ranges[/green]"
            )
            return summary

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Import all
    # ------------------------------------------------------------------

    def import_all(self, output_dir: Path | None = None) -> SeaDoughnutCorpus:
        """Import all data from Sea_Doughnut's databases.

        Returns a SeaDoughnutCorpus with all imported data.
        """
        console.print()
        console.rule("[bold cyan]Sea_Doughnut Data Import (v2)[/bold cyan]")
        console.print(f"  Data directory: {self.data_dir}")
        console.print()

        # List available databases
        for db_name in [self.CORPUS_DB, self.TRANSCRIPTS_DB, self.REDACTION_DB,
                        self.CONCORDANCE_DB, self.PERSONS_JSON]:
            path = self.data_dir / db_name
            if path.exists():
                size_mb = path.stat().st_size / 1e6
                console.print(f"  [green]Found[/green] {db_name} ({size_mb:.1f} MB)")
            else:
                console.print(f"  [dim]Missing[/dim] {db_name}")
        console.print()

        # Import documents
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

        console.print()
        persons = self.import_persons()

        console.print()
        concordance = self.import_concordance_summary()

        corpus = SeaDoughnutCorpus(
            document_count=doc_count,
            redaction_scores=redaction_scores,
            recovered_texts=recovered_texts,
            images=images,
            transcripts=transcripts,
            entities=entities,
            concordance=concordance,
        )

        console.print()
        console.rule("[bold green]Import Complete[/bold green]")
        console.print(f"  Documents:        {doc_count:,}")
        console.print(f"  Redaction scores: {len(redaction_scores):,}")
        console.print(f"  Recovered texts:  {len(recovered_texts):,}")
        console.print(f"  Transcripts:      {len(transcripts):,}")
        console.print(f"  Persons:          {len(persons):,}")
        if concordance:
            console.print(
                f"  Concordance:      {len(concordance.provenance_ranges)} "
                f"provenance ranges"
            )
            if concordance.sdny_bridge_count:
                console.print(
                    f"  SDNY_GM bridge:   {concordance.sdny_bridge_count:,} mappings"
                )

        # Save persons to output
        if output_dir and persons:
            persons_out = output_dir / "persons-registry.json"
            persons_out.parent.mkdir(parents=True, exist_ok=True)
            persons_out.write_text(
                json.dumps(persons, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"  [green]Persons saved to {persons_out}[/green]")

        # Save concordance summary
        if output_dir and concordance:
            conc_out = output_dir / "concordance-summary.json"
            conc_out.write_text(
                concordance.model_dump_json(indent=2),
                encoding="utf-8",
            )

        return corpus
