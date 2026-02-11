"""OCR processor using IBM Docling for PDF-to-text extraction."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Document, ProcessingResult


class OcrProcessor:
    """Process PDF files through IBM Docling for OCR text extraction."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.converter = DocumentConverter()

    def process_file(self, path: Path) -> ProcessingResult:
        """OCR a single PDF and return a ProcessingResult with extracted text.

        The returned ProcessingResult contains a Document whose ``ocrText``
        field is populated with the Docling markdown export.  If conversion
        fails, the result will carry the error message and a ``None`` document.
        """
        start_ms = time.monotonic_ns() // 1_000_000
        errors: list[str] = []
        warnings: list[str] = []
        document: Document | None = None

        try:
            result = self.converter.convert(str(path))
            md_text = result.document.export_to_markdown()

            if not md_text or not md_text.strip():
                warnings.append(f"Docling produced empty text for {path.name}")
                md_text = ""

            # Compute a content hash from the raw file bytes for dedup.
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            # Build a Document with the extracted text.
            doc_id = f"ocr-{content_hash[:12]}"
            document = Document(
                id=doc_id,
                title=path.stem.replace("_", " ").replace("-", " ").strip(),
                source="efta",
                category="other",
                ocrText=md_text,
                tags=["ocr", "docling"],
            )

        except Exception as exc:
            errors.append(f"Docling conversion failed for {path.name}: {exc}")

        elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
        return ProcessingResult(
            source_path=str(path),
            document=document,
            errors=errors,
            warnings=warnings,
            processing_time_ms=elapsed,
        )

    def process_batch(
        self, paths: list[Path], output_dir: Path
    ) -> list[ProcessingResult]:
        """Process multiple PDFs with a Rich progress bar.

        For each file a JSON result is written to *output_dir* using the
        SHA-256 content hash as filename (``{hash}.json``).  Files whose
        output JSON already exists in *output_dir* are skipped automatically,
        making the batch resumable.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ProcessingResult] = []

        # Pre-compute hashes so we can skip already-processed files.
        to_process: list[tuple[Path, str]] = []
        for p in paths:
            content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            out_path = output_dir / f"{content_hash}.json"
            if out_path.exists():
                # Already processed -- load previous result.
                try:
                    prev = ProcessingResult.model_validate_json(
                        out_path.read_text(encoding="utf-8")
                    )
                    results.append(prev)
                except Exception:
                    # Corrupted cache -- reprocess.
                    to_process.append((p, content_hash))
            else:
                to_process.append((p, content_hash))

        if not to_process:
            return results

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task("OCR processing", total=len(to_process))
            for file_path, content_hash in to_process:
                progress.update(task, description=f"OCR: {file_path.name[:40]}")
                result = self.process_file(file_path)
                results.append(result)

                # Persist result to disk.
                out_path = output_dir / f"{content_hash}.json"
                out_path.write_text(
                    result.model_dump_json(indent=2), encoding="utf-8"
                )

                progress.advance(task)

        return results
