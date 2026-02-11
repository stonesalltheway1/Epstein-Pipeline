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


@cli.command()
@click.argument("source", type=click.Choice(["doj", "kaggle", "huggingface", "archive"]))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory for downloaded files.")
def download(source: str, output: Path | None) -> None:
    """Download documents from a supported source.

    SOURCE must be one of: doj, kaggle, huggingface, archive.

    \b
    Examples:
      epstein-pipeline download doj --output ./data/doj
      epstein-pipeline download kaggle
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
    """Download documents from the DOJ EFTA releases."""
    console.print("[bold]Downloading DOJ EFTA documents...[/bold]")
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx is required for downloads. Install with: pip install httpx[/red]")
        sys.exit(1)

    # The DOJ provides a known index of released Epstein documents.
    index_url = "https://www.justice.gov/d9/2024-12/epstein_index.json"
    console.print(f"  Fetching index from {index_url}")

    try:
        resp = httpx.get(index_url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        index_data = resp.json()
    except Exception as exc:
        console.print(f"[red]Failed to fetch DOJ index: {exc}[/red]")
        console.print("[yellow]The DOJ index URL may have changed. Check https://www.justice.gov for updates.[/yellow]")
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
    """Download the Epstein-ranker dataset from Kaggle."""
    console.print("[bold]Downloading Kaggle epstein-ranker dataset...[/bold]")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "jamesgallagher/epstein-ranker",
            path=str(out_dir),
            unzip=True,
        )
        console.print(f"  [green]Downloaded and extracted to {out_dir}[/green]")
    except ImportError:
        console.print("[yellow]kaggle package not installed. Install with: pip install kaggle[/yellow]")
        console.print("[yellow]Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables.[/yellow]")
        console.print(f"[dim]Alternative: manually download from https://www.kaggle.com/datasets/jamesgallagher/epstein-ranker and extract to {out_dir}[/dim]")
    except Exception as exc:
        console.print(f"[red]Kaggle download failed: {exc}[/red]")


def _download_huggingface(out_dir: Path) -> None:
    """Download datasets from HuggingFace."""
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
        console.print("[yellow]huggingface_hub package not installed. Install with: pip install huggingface_hub[/yellow]")
    except Exception as exc:
        console.print(f"[red]HuggingFace download failed: {exc}[/red]")


def _download_archive(out_dir: Path) -> None:
    """Download Epstein-related items from the Internet Archive."""
    console.print("[bold]Searching Internet Archive for Epstein materials...[/bold]")
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx is required for downloads. Install with: pip install httpx[/red]")
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
        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        console.print(f"  Found {len(docs)} items on Archive.org")

        manifest_path = out_dir / "archive_manifest.json"
        manifest_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
        console.print(f"  [green]Saved manifest to {manifest_path}[/green]")
    except Exception as exc:
        console.print(f"[red]Archive.org search failed: {exc}[/red]")


# ---------------------------------------------------------------------------
# ocr
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory for OCR results (JSON files).")
def ocr(input_dir: Path, output: Path | None) -> None:
    """OCR PDF files using IBM Docling.

    Processes all .pdf files in INPUT_DIR and writes JSON results to the
    output directory.  Already-processed files are skipped automatically.

    \b
    Examples:
      epstein-pipeline ocr ./data/pdfs --output ./output/ocr
      epstein-pipeline ocr /mnt/epstein/VOL00009
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "ocr"

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return

    console.print(f"Found [bold]{len(pdfs)}[/bold] PDF files in {input_dir}")

    from epstein_pipeline.processors.ocr import OcrProcessor

    processor = OcrProcessor(settings)
    results = processor.process_batch(pdfs, out_dir)

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
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory for entity extraction results.")
@click.option("--registry", "-r", type=click.Path(exists=True, path_type=Path), default=None, help="Path to persons-registry.json.")
def extract_entities(input_dir: Path, output: Path | None, registry: Path | None) -> None:
    """Extract named entities (persons) from OCR JSON files.

    Reads JSON files produced by the 'ocr' command from INPUT_DIR and
    writes enriched copies with personIds populated.

    \b
    Examples:
      epstein-pipeline extract-entities ./output/ocr --output ./output/entities
      epstein-pipeline extract-entities ./output/ocr -r ./data/persons-registry.json
    """
    settings = _load_settings()
    out_dir = output or settings.output_dir / "entities"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry or settings.persons_registry_path

    if not registry_path.exists():
        console.print(f"[red]Person registry not found at {registry_path}[/red]")
        console.print("[yellow]Generate it first or specify --registry path.[/yellow]")
        sys.exit(1)

    from epstein_pipeline.models.document import ProcessingResult
    from epstein_pipeline.models.registry import PersonRegistry
    from epstein_pipeline.processors.entities import EntityExtractor

    console.print(f"Loading person registry from {registry_path}")
    person_registry = PersonRegistry.from_json(registry_path)
    console.print(f"  Loaded [bold]{len(person_registry)}[/bold] persons")

    extractor = EntityExtractor(settings, person_registry)

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
            console.print(f"  [yellow]Skipping invalid file: {jf.name}[/yellow]")
            continue

        if result.document is None:
            continue

        text_parts = []
        if result.document.title:
            text_parts.append(result.document.title)
        if result.document.summary:
            text_parts.append(result.document.summary)
        if result.document.ocrText:
            text_parts.append(result.document.ocrText)

        combined = "\n".join(text_parts)
        person_ids = extractor.extract(combined)
        result.document.personIds = person_ids
        total_entities += len(person_ids)

        out_path = out_dir / jf.name
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    console.print(f"[green]Done.[/green] Extracted {total_entities} person links across {len(json_files)} files.")
    console.print(f"[dim]Output: {out_dir}[/dim]")


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output path for dedup report (JSON).")
@click.option("--threshold", "-t", type=float, default=0.90, help="Similarity threshold (0.0 - 1.0). Default: 0.90.")
def dedup(input_dir: Path, output: Path | None, threshold: float) -> None:
    """Find duplicate documents.

    Reads JSON files from INPUT_DIR and generates a deduplication report
    listing all candidate duplicate pairs with similarity scores.

    \b
    Examples:
      epstein-pipeline dedup ./output/ocr --output ./output/dedup-report.json
      epstein-pipeline dedup ./output/entities --threshold 0.85
    """
    settings = _load_settings()
    report_path = output or settings.output_dir / "dedup-report.json"

    from epstein_pipeline.models.document import Document, ProcessingResult
    from epstein_pipeline.processors.dedup import Deduplicator

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    console.print(f"Loading [bold]{len(json_files)}[/bold] documents...")

    documents: list[Document] = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            # Support both raw Document JSON and ProcessingResult wrappers.
            if "document" in raw and raw["document"] is not None:
                result = ProcessingResult.model_validate(raw)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    console.print(f"  Loaded {len(documents)} valid documents")

    deduplicator = Deduplicator(threshold=threshold)
    pairs = deduplicator.find_duplicates(documents)

    console.print(f"  Found [bold]{len(pairs)}[/bold] duplicate pairs (threshold={threshold})")

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
    report_data = [p.model_dump() for p in pairs]
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"[green]Report saved to {report_path}[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def validate(input_dir: Path) -> None:
    """Validate JSON files against the Document schema.

    Checks every .json file in INPUT_DIR to verify it conforms to the
    pipeline's Document or ProcessingResult model.

    \b
    Examples:
      epstein-pipeline validate ./output/ocr
      epstein-pipeline validate ./data
    """
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

        # Try ProcessingResult first, then bare Document.
        try:
            if "document" in raw:
                ProcessingResult.model_validate(raw)
            elif "id" in raw and "title" in raw:
                Document.model_validate(raw)
            else:
                invalid += 1
                errors_log.append((jf.name, "Unrecognised schema (no 'document' or 'id'+'title' fields)"))
                continue
            valid += 1
        except ValidationError as exc:
            invalid += 1
            first_error = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
            errors_log.append((jf.name, f"Validation: {first_error.get('msg', str(first_error))}"))

    console.print()
    console.print(f"  [green]Valid:[/green]   {valid}")
    console.print(f"  [red]Invalid:[/red] {invalid}")

    if errors_log:
        console.print()
        table = Table(title="Validation Errors", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Error", style="red")
        for name, err in errors_log[:30]:
            table.add_row(name, err)
        if len(errors_log) > 30:
            console.print(f"  [dim]...and {len(errors_log) - 30} more errors[/dim]")
        console.print(table)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("format", type=click.Choice(["json", "csv", "sqlite"]))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path.")
def export(format: str, input_dir: Path, output: Path | None) -> None:
    """Export processed documents to JSON, CSV, or SQLite.

    Reads JSON files from INPUT_DIR and writes a consolidated export file.

    \b
    Examples:
      epstein-pipeline export json ./output/ocr --output ./export/documents.json
      epstein-pipeline export csv ./output/entities --output ./export/documents.csv
      epstein-pipeline export sqlite ./output/entities --output ./export/epstein.db
    """
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
        console.print("[yellow]No valid documents found to export.[/yellow]")
        return

    console.print(f"Exporting [bold]{len(documents)}[/bold] documents as {format.upper()}")

    if format == "json":
        out_path = output or settings.output_dir / "export.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [d.model_dump(exclude_none=True) for d in documents]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]Exported to {out_path}[/green] ({out_path.stat().st_size / 1024:.1f} KB)")

    elif format == "csv":
        import csv

        out_path = output or settings.output_dir / "export.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "id", "title", "date", "source", "category", "summary",
            "personIds", "tags", "pdfUrl", "pageCount", "batesRange",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for doc in documents:
                row = doc.model_dump(exclude_none=True, exclude={"ocrText"})
                # Flatten lists to semicolon-separated strings for CSV.
                if "personIds" in row:
                    row["personIds"] = ";".join(row["personIds"])
                if "tags" in row:
                    row["tags"] = ";".join(row["tags"])
                writer.writerow(row)

        console.print(f"[green]Exported to {out_path}[/green] ({out_path.stat().st_size / 1024:.1f} KB)")

    elif format == "sqlite":
        import sqlite3

        out_path = output or settings.output_dir / "epstein.db"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(out_path))
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                date TEXT,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                summary TEXT,
                person_ids TEXT,
                tags TEXT,
                pdf_url TEXT,
                page_count INTEGER,
                bates_range TEXT,
                ocr_text TEXT
            )
        """)

        cur.execute("DELETE FROM documents")

        for doc in documents:
            cur.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, title, date, source, category, summary, person_ids, tags,
                 pdf_url, page_count, bates_range, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.title,
                    doc.date,
                    doc.source,
                    doc.category,
                    doc.summary,
                    ";".join(doc.personIds),
                    ";".join(doc.tags),
                    doc.pdfUrl,
                    doc.pageCount,
                    doc.batesRange,
                    doc.ocrText,
                ),
            )

        # Create full-text search index on OCR text.
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id, title, summary, ocr_text,
                content='documents',
                content_rowid='rowid'
            )
        """)
        cur.execute("DELETE FROM documents_fts")
        cur.execute("""
            INSERT INTO documents_fts (id, title, summary, ocr_text)
            SELECT id, title, summary, ocr_text FROM documents
        """)

        conn.commit()
        conn.close()

        console.print(f"[green]Exported to {out_path}[/green] ({out_path.stat().st_size / 1024:.1f} KB)")
        console.print("[dim]Includes FTS5 full-text search on id, title, summary, ocr_text[/dim]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def stats(input_dir: Path) -> None:
    """Print statistics about processed data.

    Scans JSON files in INPUT_DIR and displays a summary of document counts,
    sources, categories, person links, and OCR coverage.

    \b
    Examples:
      epstein-pipeline stats ./output/ocr
      epstein-pipeline stats ./output/entities
    """
    from collections import Counter

    from epstein_pipeline.models.document import Document, ProcessingResult

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON files found in {input_dir}[/yellow]")
        return

    documents: list[Document] = []
    total_errors = 0
    total_warnings = 0

    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if "document" in raw:
                result = ProcessingResult.model_validate(raw)
                total_errors += len(result.errors)
                total_warnings += len(result.warnings)
                if result.document:
                    documents.append(result.document)
            elif "id" in raw and "title" in raw:
                documents.append(Document.model_validate(raw))
        except Exception:
            continue

    if not documents:
        console.print("[yellow]No valid documents found.[/yellow]")
        return

    # Compute statistics
    source_counts = Counter(d.source for d in documents)
    category_counts = Counter(d.category for d in documents)
    has_ocr = sum(1 for d in documents if d.ocrText and d.ocrText.strip())
    has_summary = sum(1 for d in documents if d.summary)
    has_persons = sum(1 for d in documents if d.personIds)
    has_pdf = sum(1 for d in documents if d.pdfUrl)
    has_bates = sum(1 for d in documents if d.batesRange)
    total_person_links = sum(len(d.personIds) for d in documents)
    unique_persons = len({pid for d in documents for pid in d.personIds})

    total_ocr_chars = sum(len(d.ocrText) for d in documents if d.ocrText)

    # Display
    console.print()
    console.print(f"[bold]Dataset Statistics[/bold] -- {input_dir}")
    console.print(f"  Files scanned:  {len(json_files)}")
    console.print(f"  Valid documents: {len(documents)}")
    console.print(f"  Processing errors: {total_errors}")
    console.print(f"  Processing warnings: {total_warnings}")
    console.print()

    # Coverage
    coverage_table = Table(title="Coverage", show_lines=False)
    coverage_table.add_column("Field", style="cyan")
    coverage_table.add_column("Count", justify="right")
    coverage_table.add_column("Percent", justify="right")
    total = len(documents)
    coverage_table.add_row("OCR text", str(has_ocr), f"{has_ocr / total:.0%}")
    coverage_table.add_row("Summary", str(has_summary), f"{has_summary / total:.0%}")
    coverage_table.add_row("Person links", str(has_persons), f"{has_persons / total:.0%}")
    coverage_table.add_row("PDF URL", str(has_pdf), f"{has_pdf / total:.0%}")
    coverage_table.add_row("Bates range", str(has_bates), f"{has_bates / total:.0%}")
    console.print(coverage_table)
    console.print()

    # Sources
    source_table = Table(title="Sources", show_lines=False)
    source_table.add_column("Source", style="cyan")
    source_table.add_column("Count", justify="right")
    for src, count in source_counts.most_common():
        source_table.add_row(src, str(count))
    console.print(source_table)
    console.print()

    # Categories
    cat_table = Table(title="Categories", show_lines=False)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right")
    for cat, count in category_counts.most_common():
        cat_table.add_row(cat, str(count))
    console.print(cat_table)
    console.print()

    # Person linkage
    console.print(f"  Total person links: {total_person_links}")
    console.print(f"  Unique persons referenced: {unique_persons}")
    console.print(f"  Total OCR text: {total_ocr_chars:,} characters ({total_ocr_chars / 1_000_000:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
