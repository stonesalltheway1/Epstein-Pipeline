"""Click CLI for the Epstein Pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from epstein_pipeline.config import Settings

console = Console()

BANNER = """
[bold cyan]Epstein Pipeline[/bold cyan] - Open Source Document Processing
[dim]https://epsteinexposed.com | https://github.com/stonesalltheway1/Epstein-Pipeline[/dim]
"""


def _load_settings() -> Settings:
    """Load settings from environment, creating dirs as needed."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="epstein-pipeline")
def cli() -> None:
    """Epstein Pipeline -- Open Source Document Processing.

    Process, OCR, deduplicate, and export Epstein case file documents.

    \b
    https://epsteinexposed.com
    https://github.com/stonesalltheway1/Epstein-Pipeline
    """
    console.print(BANNER)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# migrate (NEW — Phase 8)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--database-url",
    envvar="EPSTEIN_NEON_DATABASE_URL",
    required=True,
    help="Neon Postgres connection URL.",
)
def migrate(database_url: str) -> None:
    """Run schema migrations against a Neon Postgres database.

    Creates all tables, indexes, and extensions needed by the pipeline.
    Safe to run repeatedly (idempotent).

    \b
    Examples:
      epstein-pipeline migrate --database-url postgresql://...@...neon.tech/epstein
      EPSTEIN_NEON_DATABASE_URL=... epstein-pipeline migrate
    """
    from epstein_pipeline.exporters.neon_schema import run_migration_sync

    console.print("[bold]Running Neon schema migration...[/bold]")
    try:
        run_migration_sync(database_url)
        console.print("[green]Schema migration complete.[/green]")
    except Exception as exc:
        console.print(f"[red]Migration failed: {exc}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# export neon (NEW — Phase 1)
# ---------------------------------------------------------------------------


@cli.command("export-neon")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--database-url",
    envvar="EPSTEIN_NEON_DATABASE_URL",
    required=True,
    help="Neon Postgres connection URL.",
)
@click.option("--batch-size", type=int, default=100, help="Rows per upsert batch.")
def export_neon(input_dir: Path, database_url: str, batch_size: int) -> None:
    """Export processed documents to Neon Postgres with pgvector.

    Reads processed JSON files and upserts them into the Neon database.

    \b
    Examples:
      epstein-pipeline export-neon ./output/entities --database-url postgresql://...
      EPSTEIN_NEON_DATABASE_URL=... epstein-pipeline export-neon ./output/entities
    """
    import asyncio

    from epstein_pipeline.exporters.neon_export import NeonExporter
    from epstein_pipeline.models.document import Document, ProcessingResult

    settings = _load_settings()
    settings.neon_database_url = database_url
    settings.neon_batch_size = batch_size

    json_files = sorted(input_dir.rglob("*.json"))
    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No documents found to export.[/yellow]")
        return

    console.print(f"Exporting [bold]{len(documents)}[/bold] documents to Neon Postgres")

    exporter = NeonExporter(settings)
    asyncio.run(exporter.upsert_documents(documents))
    console.print("[green]Export complete.[/green]")


# ---------------------------------------------------------------------------
# classify (NEW — Phase 6)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=str, default=None, help="Classification model.")
@click.option("--threshold", type=float, default=0.6, help="Confidence threshold.")
def classify(input_dir: Path, output: Path | None, model: str | None, threshold: float) -> None:
    """Classify documents into categories using zero-shot classification.

    \b
    Examples:
      epstein-pipeline classify ./output/ocr --output ./output/classified
      epstein-pipeline classify ./output/ocr --threshold 0.7
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "classified"
    out_dir.mkdir(parents=True, exist_ok=True)

    from epstein_pipeline.models.document import Document, ProcessingResult
    from epstein_pipeline.processors.classifier import DocumentClassifier

    if model:
        settings.classifier_model = model
    settings.classifier_confidence_threshold = threshold

    json_files = sorted(input_dir.glob("*.json"))
    documents: list[Document] = []
    results_map: dict[str, ProcessingResult] = {}

    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
                    results_map[jf.name] = result
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No documents found.[/yellow]")
        return

    console.print(f"Classifying [bold]{len(documents)}[/bold] documents")

    classifier = DocumentClassifier(settings)
    classifications = classifier.classify_batch(documents)

    for cls_result in classifications:
        console.print(
            f"  {cls_result.document_id}: "
            f"[cyan]{cls_result.predicted_category}[/cyan] "
            f"({cls_result.confidence:.0%})"
        )

    console.print(
        f"[green]Classification complete. {len(classifications)} documents classified.[/green]"
    )


# ---------------------------------------------------------------------------
# search (NEW — semantic search demo)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--database-url",
    envvar="EPSTEIN_NEON_DATABASE_URL",
    required=True,
    help="Neon Postgres connection URL.",
)
@click.option("--limit", "-n", type=int, default=10, help="Number of results.")
@click.option("--threshold", type=float, default=0.7, help="Similarity threshold.")
def search(query: str, database_url: str, limit: int, threshold: float) -> None:
    """Semantic search across document embeddings in Neon Postgres.

    \b
    Examples:
      epstein-pipeline search "flight logs to Virgin Islands" --limit 5
      epstein-pipeline search "financial transactions" --threshold 0.8
    """
    import asyncio

    from epstein_pipeline.exporters.neon_export import NeonExporter

    settings = _load_settings()
    settings.neon_database_url = database_url

    console.print(f'Searching for: [bold cyan]"{query}"[/bold cyan]')

    exporter = NeonExporter(settings)
    results = asyncio.run(exporter.semantic_search(query, limit=limit, threshold=threshold))

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} Results", show_lines=True)
    table.add_column("Document", style="cyan", max_width=30)
    table.add_column("Chunk", justify="right")
    table.add_column("Similarity", justify="right", style="bold")
    table.add_column("Text", max_width=60)

    for r in results:
        table.add_row(
            r["document_id"],
            str(r["chunk_index"]),
            f"{r['similarity']:.2%}",
            r["chunk_text"][:100] + "...",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# download (existing)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("source", type=click.Choice(["doj", "kaggle", "huggingface", "archive"]))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory."
)
def download(source: str, output: Path | None) -> None:
    """Download documents from a supported source.

    SOURCE must be one of: doj, kaggle, huggingface, archive.
    """
    settings = _load_settings()
    out_dir = output or settings.data_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "doj":
        _download_doj(out_dir)
    elif source == "kaggle":
        _download_kaggle(out_dir)
    elif source == "huggingface":
        _download_huggingface(out_dir)
    elif source == "archive":
        _download_archive(out_dir)


def _download_doj(out_dir: Path) -> None:
    console.print("[bold]Downloading DOJ EFTA documents...[/bold]")
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx is required. Install with: pip install httpx[/red]")
        sys.exit(1)

    index_url = "https://www.justice.gov/d9/2024-12/epstein_index.json"
    console.print(f"  Fetching index from {index_url}")

    try:
        resp = httpx.get(index_url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        index_data = resp.json()
    except Exception as exc:
        console.print(f"[red]Failed to fetch DOJ index: {exc}[/red]")
        sys.exit(1)

    if isinstance(index_data, list):
        items = index_data
    elif isinstance(index_data, dict):
        items = index_data.get("documents", index_data.get("files", []))
    else:
        items = []

    console.print(f"  Found {len(items)} items in index")
    index_path = out_dir / "doj_index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    console.print(f"  [green]Saved index to {index_path}[/green]")


def _download_kaggle(out_dir: Path) -> None:
    console.print("[bold]Downloading Kaggle epstein-ranker dataset...[/bold]")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("jamesgallagher/epstein-ranker", path=str(out_dir), unzip=True)
        console.print(f"  [green]Downloaded to {out_dir}[/green]")
    except ImportError:
        console.print("[yellow]kaggle package not installed.[/yellow]")
    except Exception as exc:
        console.print(f"[red]Kaggle download failed: {exc}[/red]")


def _download_huggingface(out_dir: Path) -> None:
    console.print("[bold]Downloading HuggingFace Epstein datasets...[/bold]")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="danieldux/jeffrey-epstein-document-corpus",
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        console.print(f"  [green]Downloaded to {out_dir}[/green]")
    except ImportError:
        console.print("[yellow]huggingface_hub not installed.[/yellow]")
    except Exception as exc:
        console.print(f"[red]HuggingFace download failed: {exc}[/red]")


def _download_archive(out_dir: Path) -> None:
    console.print("[bold]Searching Internet Archive...[/bold]")
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx is required.[/red]")
        sys.exit(1)

    search_url = "https://archive.org/advancedsearch.php"
    params = {
        "q": "jeffrey epstein",
        "fl[]": "identifier,title,mediatype,date",
        "sort[]": "date desc",
        "rows": "500",
        "output": "json",
    }
    try:
        resp = httpx.get(search_url, params=params, timeout=60.0)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        console.print(f"  Found {len(docs)} items")
        manifest_path = out_dir / "archive_manifest.json"
        manifest_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
        console.print(f"  [green]Saved to {manifest_path}[/green]")
    except Exception as exc:
        console.print(f"[red]Archive.org search failed: {exc}[/red]")


# ---------------------------------------------------------------------------
# ocr
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory."
)
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers.")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["auto", "pymupdf", "surya", "olmocr", "docling"]),
    default=None,
    help="OCR backend (default: auto = pymupdf → surya → docling fallback chain).",
)
def ocr(input_dir: Path, output: Path | None, workers: int | None, backend: str | None) -> None:
    """OCR PDF files using multiple backends.

    Backends:
      auto    - PyMuPDF → Surya → Docling fallback chain (default)
      pymupdf - Extract existing text layers only (fastest)
      surya   - Surya OCR with confidence scoring (CPU/GPU)
      olmocr  - Allen AI olmOCR 2 VLM (GPU required, highest quality)
      docling - IBM Docling (fallback)

    \b
    Examples:
      epstein-pipeline ocr ./data/pdfs --output ./output/ocr
      epstein-pipeline ocr ./pdfs --backend surya --workers 8
      epstein-pipeline ocr ./pdfs --backend olmocr
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "ocr"
    ocr_backend = backend or settings.ocr_backend

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return

    console.print(f"Found [bold]{len(pdfs)}[/bold] PDFs, backend={ocr_backend}")

    from epstein_pipeline.processors.ocr import OcrProcessor

    processor = OcrProcessor(settings, backend=ocr_backend)
    results = processor.process_batch(pdfs, out_dir, max_workers=workers)

    successes = sum(1 for r in results if r.document is not None)
    failures = sum(1 for r in results if r.errors)
    total_time = sum(r.processing_time_ms for r in results)

    console.print()
    console.print(f"[green]Completed:[/green] {successes} succeeded, {failures} failed")
    console.print(f"[dim]Total time: {total_time / 1000:.1f}s | Output: {out_dir}[/dim]")


# ---------------------------------------------------------------------------
# extract-entities
# ---------------------------------------------------------------------------


@cli.command("extract-entities")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--registry", "-r", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--workers", "-w", type=int, default=None)
@click.option(
    "--entity-types",
    type=str,
    default="PERSON",
    help="Comma-separated entity types (PERSON,ORG,GPE,DATE,MONEY,PHONE,EMAIL_ADDR,all).",
)
def extract_entities(
    input_dir: Path,
    output: Path | None,
    registry: Path | None,
    workers: int | None,
    entity_types: str,
) -> None:
    """Extract named entities from OCR JSON files.

    \b
    Examples:
      epstein-pipeline extract-entities ./output/ocr --entity-types all
      epstein-pipeline extract-entities ./output/ocr -r ./data/persons-registry.json -w 4
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "entities"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry or settings.persons_registry_path

    if not registry_path.exists():
        console.print(f"[red]Person registry not found at {registry_path}[/red]")
        sys.exit(1)

    from epstein_pipeline.models.document import ProcessingResult
    from epstein_pipeline.models.registry import PersonRegistry
    from epstein_pipeline.processors.entities import EntityExtractor

    person_registry = PersonRegistry.from_json(registry_path)
    console.print(f"Loaded [bold]{len(person_registry)}[/bold] persons")

    types_set = set(t.strip() for t in entity_types.split(","))
    extractor = EntityExtractor(settings, person_registry, entity_types=types_set)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    console.print(f"Processing [bold]{len(json_files)}[/bold] files")

    total_entities = 0
    for jf in json_files:
        try:
            result = ProcessingResult.model_validate_json(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if result.document is None:
            continue

        text_parts = [
            t
            for t in [result.document.title, result.document.summary, result.document.ocrText]
            if t
        ]
        combined = "\n".join(text_parts)
        extraction = extractor.extract_all(combined)
        result.document.personIds = extraction.person_ids
        total_entities += len(extraction.person_ids)

        out_path = out_dir / jf.name
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    console.print(
        f"[green]Done.[/green] {total_entities} person links across {len(json_files)} files."
    )


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", "-t", type=float, default=0.90)
@click.option(
    "--mode",
    type=click.Choice(["exact", "minhash", "semantic", "all"]),
    default="all",
    help="Dedup strategy: exact, minhash, semantic, or all (default).",
)
@click.option("--clusters", is_flag=True, default=False, help="Output duplicate clusters.")
def dedup(
    input_dir: Path, output: Path | None, threshold: float, mode: str, clusters: bool
) -> None:
    """Find duplicate documents using multiple strategies.

    Strategies:
      exact    - Content hash + title fuzzy + Bates overlap
      minhash  - MinHash/LSH near-duplicate detection (O(n))
      semantic - Embedding cosine similarity
      all      - All three passes (recommended)

    \b
    Examples:
      epstein-pipeline dedup ./output/ocr --mode all
      epstein-pipeline dedup ./output/ocr --mode minhash --clusters
    """
    settings = _load_settings()
    report_path = output or settings.output_dir / "dedup-report.json"

    from epstein_pipeline.models.document import Document, ProcessingResult
    from epstein_pipeline.processors.dedup import Deduplicator

    # Apply CLI overrides to settings
    settings.dedup_threshold = threshold
    from epstein_pipeline.config import DedupMode

    settings.dedup_mode = DedupMode(mode)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    console.print(f"Loaded {len(documents)} documents, mode={mode}")
    deduplicator = Deduplicator(settings=settings)

    if clusters:
        dup_clusters = deduplicator.find_clusters(documents)
        console.print(f"Found [bold]{len(dup_clusters)}[/bold] duplicate clusters")
        pairs = deduplicator.find_duplicates(documents)
    else:
        pairs = deduplicator.find_duplicates(documents)
    console.print(f"Found [bold]{len(pairs)}[/bold] duplicate pairs (threshold={threshold})")

    if pairs:
        table = Table(title="Top Duplicate Pairs", show_lines=True)
        table.add_column("Doc 1", style="cyan")
        table.add_column("Doc 2", style="cyan")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("Reason")
        for pair in pairs[:20]:
            table.add_row(pair.doc_id_1, pair.doc_id_2, f"{pair.score:.2%}", pair.reason)
        console.print(table)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps([p.model_dump() for p in pairs], indent=2), encoding="utf-8")
    console.print(f"[green]Report saved to {report_path}[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def validate(input_dir: Path) -> None:
    """Validate JSON files against the Document schema."""
    from pydantic import ValidationError

    from epstein_pipeline.models.document import Document, ProcessingResult

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    console.print(f"Validating [bold]{len(json_files)}[/bold] JSON files...")

    valid = 0
    invalid = 0
    errors_log: list[tuple[str, str]] = []

    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid += 1
            errors_log.append((jf.name, f"Invalid JSON: {exc}"))
            continue

        try:
            if "document" in raw:
                ProcessingResult.model_validate(raw)
            elif "id" in raw and "title" in raw:
                Document.model_validate(raw)
            else:
                invalid += 1
                errors_log.append((jf.name, "Unrecognised schema"))
                continue
            valid += 1
        except ValidationError as exc:
            invalid += 1
            first_error = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
            errors_log.append((jf.name, f"Validation: {first_error.get('msg', str(first_error))}"))

    console.print(f"\n  [green]Valid:[/green]   {valid}")
    console.print(f"  [red]Invalid:[/red] {invalid}")

    if errors_log:
        table = Table(title="Validation Errors", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Error", style="red")
        for name, err in errors_log[:30]:
            table.add_row(name, err)
        console.print(table)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("format", type=click.Choice(["json", "csv", "sqlite"]))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def export(format: str, input_dir: Path, output: Path | None) -> None:
    """Export processed documents to JSON, CSV, or SQLite."""
    settings = _load_settings()

    from epstein_pipeline.models.document import Document, ProcessingResult

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No valid documents found.[/yellow]")
        return

    console.print(f"Exporting [bold]{len(documents)}[/bold] documents as {format.upper()}")

    if format == "json":
        out_path = output or settings.output_dir / "export.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [d.model_dump(exclude_none=True) for d in documents]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]Exported to {out_path}[/green]")

    elif format == "csv":
        import csv

        out_path = output or settings.output_dir / "export.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "title",
            "date",
            "source",
            "category",
            "summary",
            "personIds",
            "tags",
            "pdfUrl",
            "pageCount",
            "batesRange",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for doc in documents:
                row = doc.model_dump(exclude_none=True, exclude={"ocrText"})
                if "personIds" in row:
                    row["personIds"] = ";".join(row["personIds"])
                if "tags" in row:
                    row["tags"] = ";".join(row["tags"])
                writer.writerow(row)
        console.print(f"[green]Exported to {out_path}[/green]")

    elif format == "sqlite":
        from epstein_pipeline.exporters.sqlite_export import SqliteExporter

        out_path = output or settings.output_dir / "epstein.db"
        exporter = SqliteExporter()
        exporter.export(documents=documents, persons=[], db_path=out_path)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def stats(input_dir: Path) -> None:
    """Print statistics about processed data."""
    from collections import Counter

    from epstein_pipeline.models.document import Document, ProcessingResult

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    documents: list[Document] = []
    total_errors = 0

    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw:
                result = ProcessingResult.model_validate(raw)
                total_errors += len(result.errors)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No valid documents found.[/yellow]")
        return

    source_counts = Counter(d.source for d in documents)
    Counter(d.category for d in documents)
    has_ocr = sum(1 for d in documents if d.ocrText and d.ocrText.strip())
    has_summary = sum(1 for d in documents if d.summary)
    has_persons = sum(1 for d in documents if d.personIds)
    total_person_links = sum(len(d.personIds) for d in documents)
    unique_persons = len({pid for d in documents for pid in d.personIds})
    total_ocr_chars = sum(len(d.ocrText) for d in documents if d.ocrText)
    total = len(documents)

    console.print(f"\n[bold]Dataset Statistics[/bold] -- {input_dir}")
    console.print(f"  Files: {len(json_files)} | Documents: {total} | Errors: {total_errors}")
    console.print(
        f"  OCR text: {has_ocr} ({has_ocr / total:.0%})"
        f" | Summaries: {has_summary} | Person links: {has_persons}"
    )
    console.print(
        f"  Person links total: {total_person_links:,} | Unique persons: {unique_persons}"
    )
    console.print(f"  OCR chars: {total_ocr_chars:,} ({total_ocr_chars / 1_000_000:.1f} MB)")

    table = Table(title="Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right")
    for src, count in source_counts.most_common():
        table.add_row(src, str(count))
    console.print(table)


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./output/embeddings).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Embedding model (default: nomic-ai/nomic-embed-text-v2-moe).",
)
@click.option(
    "--dimensions",
    "-d",
    type=int,
    default=768,
    help="Embedding dimensions: 768 (full) or 256 (Matryoshka).",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=None,
    help="Batch size (auto-detects for GPU vs CPU).",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default=None,
    help="Force compute device.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["ndjson", "sqlite", "neon"]),
    default="ndjson",
    help="Output format: ndjson, sqlite, or neon (pgvector).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=3200,
    help="Chunk size in characters (~800 tokens).",
)
@click.option(
    "--overlap",
    type=int,
    default=800,
    help="Chunk overlap in characters (~200 tokens).",
)
def embed(
    input_dir: Path,
    output: Path | None,
    model: str | None,
    dimensions: int,
    batch_size: int | None,
    device: str | None,
    fmt: str,
    chunk_size: int,
    overlap: int,
) -> None:
    """Generate vector embeddings for documents.

    Reads processed JSON files from INPUT_DIR, chunks the text, and
    generates embeddings using sentence-transformers.

    \b
    Examples:
      epstein-pipeline embed ./output/ocr
      epstein-pipeline embed ./output/ocr --format sqlite --dimensions 256
      epstein-pipeline embed ./output/ocr --device cuda --batch-size 128
    """
    from epstein_pipeline.models.document import Document, ProcessingResult
    from epstein_pipeline.processors.embeddings import EmbeddingProcessor

    settings = _load_settings()
    if chunk_size != 3200:
        settings.embedding_chunk_size = chunk_size
    if overlap != 800:
        settings.embedding_chunk_overlap = overlap

    output_dir = output or settings.output_dir / "embeddings"
    console.print(BANNER)
    console.print(f"[bold]Generating embeddings[/bold] from {input_dir}")
    console.print(f"  Model: {model or settings.embedding_model}")
    console.print(f"  Dimensions: {dimensions}")
    console.print(f"  Format: {fmt}")

    # Load documents from JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        console.print("[red]No JSON files found in input directory[/red]")
        sys.exit(1)

    documents: list[Document] = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    documents.append(Document.model_validate(item))
            elif isinstance(data, dict):
                if "document" in data:
                    result = ProcessingResult.model_validate(data)
                    if result.document:
                        documents.append(result.document)
                else:
                    documents.append(Document.model_validate(data))
        except Exception as exc:
            console.print(f"  [dim]Skipped {jf.name}: {exc}[/dim]")

    console.print(f"  Loaded {len(documents)} documents")

    processor = EmbeddingProcessor(
        settings=settings,
        model_name=model,
        dimensions=dimensions,
        batch_size=batch_size,
        device=device,
    )

    results = processor.process_batch(documents, output_dir, fmt=fmt)

    total_chunks = sum(len(r.chunks) for r in results)
    console.print(
        f"\n[green]Embedded {len(results)} documents ({total_chunks} chunks) → {output_dir}[/green]"
    )


# ---------------------------------------------------------------------------
# import sea-doughnut
# ---------------------------------------------------------------------------


@cli.group("import")
def import_group() -> None:
    """Import data from external sources."""
    pass


@import_group.command("sea-doughnut")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Sea_Doughnut data directory.",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory."
)
@click.option("--limit", "-l", type=int, default=None, help="Limit documents imported.")
def import_sea_doughnut(data_dir: Path, output: Path | None, limit: int | None) -> None:
    """Import Sea_Doughnut's research databases (1.38M docs).

    \b
    Examples:
      epstein-pipeline import sea-doughnut --data-dir E:/epstein-data/sea-doughnut-v2
      epstein-pipeline import sea-doughnut -d ./sea-doughnut -o ./output/sea-doughnut
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "sea-doughnut"

    from epstein_pipeline.importers.sea_doughnut import SeaDoughnutImporter

    importer = SeaDoughnutImporter(data_dir)
    corpus = importer.import_all(output_dir=out_dir)

    # Save summary
    summary_path = (
        out_dir / "import-summary.json"
        if out_dir
        else settings.output_dir / "sea-doughnut-summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "documents": corpus.document_count,
                "redaction_scores": len(corpus.redaction_scores),
                "recovered_texts": len(corpus.recovered_texts),
                "images": len(corpus.images),
                "transcripts": len(corpus.transcripts),
                "entities": len(corpus.entities),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[green]Summary saved to {summary_path}[/green]")


# ---------------------------------------------------------------------------
# sync-registry
# ---------------------------------------------------------------------------


@cli.command("sync-registry")
@click.option(
    "--from-site-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to epstein-index/data/persons.ts",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def sync_registry(from_site_path: Path, output: Path | None) -> None:
    """Sync persons registry from the epstein-index site."""
    settings = _load_settings()
    out_path = output or settings.persons_registry_path

    # Import the sync script's logic
    import re

    content = from_site_path.read_text(encoding="utf-8")
    persons = []

    pattern = re.compile(
        r'\{\s*id:\s*"(p-\d+)".*?slug:\s*"([^"]+)".*?name:\s*"([^"]+)".*?'
        r'(?:aliases:\s*\[(.*?)\].*?)?category:\s*"([^"]+)"',
        re.DOTALL,
    )

    for match in pattern.finditer(content):
        aliases_raw = match.group(4) or ""
        aliases = re.findall(r'"([^"]+)"', aliases_raw) if aliases_raw.strip() else []
        persons.append(
            {
                "id": match.group(1),
                "slug": match.group(2),
                "name": match.group(3),
                "aliases": aliases,
                "category": match.group(5),
            }
        )

    if not persons:
        console.print("[red]No persons found in the file.[/red]")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(persons, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Synced {len(persons)} persons to {out_path}[/green]")


# ---------------------------------------------------------------------------
# analyze-redactions
# ---------------------------------------------------------------------------


@cli.command("analyze-redactions")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--workers", "-w", type=int, default=1)
def analyze_redactions(input_dir: Path, output: Path | None, workers: int) -> None:
    """Analyze PDFs for redactions and attempt text recovery.

    \b
    Examples:
      epstein-pipeline analyze-redactions ./pdfs --output ./output/redactions
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "redactions"

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return

    console.print(f"Analyzing [bold]{len(pdfs)}[/bold] PDFs for redactions")

    from epstein_pipeline.processors.redaction import RedactionAnalyzer

    analyzer = RedactionAnalyzer()
    analyzer.analyze_batch(pdfs, out_dir, max_workers=workers)


# ---------------------------------------------------------------------------
# extract-images
# ---------------------------------------------------------------------------


@cli.command("extract-images")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--describe", is_flag=True, help="Use AI vision to describe images.")
@click.option("--vision-model", type=str, default=None)
def extract_images(
    input_dir: Path, output: Path | None, describe: bool, vision_model: str | None
) -> None:
    """Extract images from PDF files.

    \b
    Examples:
      epstein-pipeline extract-images ./pdfs --output ./output/images --describe
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "images"

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {input_dir}[/yellow]")
        return

    console.print(f"Extracting images from [bold]{len(pdfs)}[/bold] PDFs")

    from epstein_pipeline.processors.image_extractor import ImageExtractor

    extractor = ImageExtractor(vision_model=vision_model or settings.vision_model)
    extractor.process_batch(pdfs, out_dir, describe=describe)


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=str, default=None, help="Whisper model size (e.g. large-v3).")
def transcribe(input_dir: Path, output: Path | None, model: str | None) -> None:
    """Transcribe audio/video files using faster-whisper.

    \b
    Examples:
      epstein-pipeline transcribe ./media --output ./output/transcripts --model large-v3
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "transcripts"

    from epstein_pipeline.processors.transcriber import SUPPORTED_EXTENSIONS

    media_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not media_files:
        console.print(f"[yellow]No media files found in {input_dir}[/yellow]")
        return

    console.print(f"Found [bold]{len(media_files)}[/bold] media files")

    from epstein_pipeline.processors.transcriber import MediaTranscriber

    transcriber = MediaTranscriber(model_size=model or settings.whisper_model)
    transcriber.transcribe_batch(media_files, out_dir)


# ---------------------------------------------------------------------------
# build-graph
# ---------------------------------------------------------------------------


@cli.command("build-graph")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "gexf", "both"]), default="json")
def build_graph(input_dir: Path, output: Path | None, fmt: str) -> None:
    """Build a knowledge graph from entity extraction results.

    \b
    Examples:
      epstein-pipeline build-graph ./output/entities --output ./output/graph.json
      epstein-pipeline build-graph ./output/entities --format both
    """
    settings = _load_settings()
    out_path = output or settings.output_dir / "graph.json"

    from epstein_pipeline.models.document import Document, ProcessingResult
    from epstein_pipeline.processors.knowledge_graph import KnowledgeGraphBuilder

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    console.print(f"Building graph from {len(documents)} documents")

    builder = KnowledgeGraphBuilder()
    builder.add_documents(documents)
    graph = builder.build()

    console.print(f"  Nodes: {graph.node_count} | Edges: {graph.edge_count}")

    if fmt in ("json", "both"):
        json_path = out_path if fmt == "json" else out_path.with_suffix(".json")
        KnowledgeGraphBuilder.export_json(graph, json_path)
        console.print(f"  [green]JSON: {json_path}[/green]")

    if fmt in ("gexf", "both"):
        gexf_path = out_path.with_suffix(".gexf")
        KnowledgeGraphBuilder.export_gexf(graph, gexf_path)
        console.print(f"  [green]GEXF: {gexf_path}[/green]")


# ---------------------------------------------------------------------------
# forensics plist
# ---------------------------------------------------------------------------


@cli.group()
def forensics() -> None:
    """Forensic analysis tools."""
    pass


@forensics.command("plist")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def forensics_plist(input_dir: Path, output: Path | None) -> None:
    """Scan PDFs for embedded Apple Mail PLIST metadata.

    \b
    Examples:
      epstein-pipeline forensics plist ./pdfs --output ./output/plist
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "plist"

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {input_dir}[/yellow]")
        return

    console.print(f"Scanning [bold]{len(pdfs)}[/bold] PDFs for PLIST metadata")

    from epstein_pipeline.processors.plist_forensics import PlistForensicsProcessor

    processor = PlistForensicsProcessor()
    processor.process_batch(pdfs, out_dir)


# ---------------------------------------------------------------------------
# sync-site
# ---------------------------------------------------------------------------


@cli.command("sync-site")
@click.option(
    "--site-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to epstein-index.",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Input directory with processed JSON.",
)
@click.option("--output-sqlite", is_flag=True, default=True)
@click.option("--output-json", is_flag=True, default=True)
def sync_site(
    site_dir: Path, input_dir: Path | None, output_sqlite: bool, output_json: bool
) -> None:
    """Export processed data to the epstein-index site format.

    \b
    Examples:
      epstein-pipeline sync-site --site-dir ../epstein-index --input-dir ./output/entities
    """
    settings = _load_settings()
    in_dir = input_dir or settings.output_dir / "entities"

    from epstein_pipeline.exporters.site_sync import SiteSyncer
    from epstein_pipeline.models.document import Document, ProcessingResult

    json_files = sorted(in_dir.rglob("*.json"))
    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No documents found to sync.[/yellow]")
        return

    syncer = SiteSyncer(site_dir)
    syncer.sync(documents, persons=[], output_sqlite=output_sqlite, output_json=output_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
