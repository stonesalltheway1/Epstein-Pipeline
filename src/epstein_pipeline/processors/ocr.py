"""OCR processor using IBM Docling and/or PyMuPDF for PDF-to-text extraction."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level function for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------


def _process_single_ocr(args: tuple[str, str, str]) -> ProcessingResult:
    """Process a single PDF file for OCR.

    This is a module-level function so it can be pickled for use with
    ProcessPoolExecutor.  Accepts a tuple of (file_path, backend, spacy_model)
    to keep the signature simple.
    """
    file_path_str, backend, _ = args
    path = Path(file_path_str)
    start_ms = time.monotonic_ns() // 1_000_000
    errors: list[str] = []
    warnings: list[str] = []
    md_text = ""

    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = f"ocr-{content_hash[:12]}"

    # Try PyMuPDF first if requested
    if backend in ("pymupdf", "both"):
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()
            md_text = "\n\n".join(pages_text).strip()

            if md_text:
                if backend == "both":
                    warnings.append("PyMuPDF extracted text successfully")
            else:
                warnings.append("PyMuPDF produced empty text")
                md_text = ""
        except ImportError:
            if backend == "pymupdf":
                errors.append("PyMuPDF not installed. Install with: pip install pymupdf")
            # Fall through to Docling if backend is "both"
        except Exception as exc:
            warnings.append(f"PyMuPDF extraction failed: {exc}")

    # Try Docling if PyMuPDF didn't produce text, or if backend is "docling"
    if not md_text and backend in ("docling", "both"):
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(path))
            md_text = result.document.export_to_markdown()

            if not md_text or not md_text.strip():
                warnings.append(f"Docling produced empty text for {path.name}")
                md_text = ""
        except ImportError:
            errors.append("Docling not installed. Install with: pip install docling")
        except Exception as exc:
            errors.append(f"Docling conversion failed for {path.name}: {exc}")

    document = (
        Document(
            id=doc_id,
            title=path.stem.replace("_", " ").replace("-", " ").strip(),
            source="efta",
            category="other",
            ocrText=md_text or None,
            tags=["ocr"],
        )
        if md_text or not errors
        else None
    )

    elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
    return ProcessingResult(
        source_path=str(path),
        document=document,
        errors=errors,
        warnings=warnings,
        processing_time_ms=elapsed,
    )


class OcrProcessor:
    """Process PDF files through OCR text extraction.

    Supports three backends:
    - ``docling`` (default) -- IBM Docling
    - ``pymupdf`` -- PyMuPDF (better for invisible OCR text layers)
    - ``both`` -- Try PyMuPDF first, fall back to Docling
    """

    def __init__(
        self,
        config: Settings,
        backend: str = "docling",
    ) -> None:
        self.config = config
        self.backend = backend

    def process_file(self, path: Path) -> ProcessingResult:
        """OCR a single PDF and return a ProcessingResult."""
        return _process_single_ocr((str(path), self.backend, self.config.spacy_model))

    def process_batch(
        self,
        paths: list[Path],
        output_dir: Path,
        max_workers: int | None = None,
    ) -> list[ProcessingResult]:
        """Process multiple PDFs with optional parallelism.

        Files whose output JSON already exists are skipped (resumable).
        Uses ProcessPoolExecutor for CPU-bound OCR when max_workers > 1.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ProcessingResult] = []
        workers = max_workers or self.config.max_workers

        # Pre-compute hashes to skip already-processed files
        to_process: list[tuple[Path, str]] = []
        for p in paths:
            content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            out_path = output_dir / f"{content_hash}.json"
            if out_path.exists():
                try:
                    prev = ProcessingResult.model_validate_json(
                        out_path.read_text(encoding="utf-8")
                    )
                    results.append(prev)
                except Exception:
                    to_process.append((p, content_hash))
            else:
                to_process.append((p, content_hash))

        if not to_process:
            return results

        if workers > 1 and len(to_process) > 1:
            results.extend(self._process_parallel(to_process, output_dir, workers))
        else:
            results.extend(self._process_sequential(to_process, output_dir))

        return results

    def _process_parallel(
        self,
        to_process: list[tuple[Path, str]],
        output_dir: Path,
        max_workers: int,
    ) -> list[ProcessingResult]:
        """Process files in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: list[ProcessingResult] = []
        args_list = [(str(p), self.backend, self.config.spacy_model) for p, _ in to_process]
        hash_map = {str(p): h for p, h in to_process}

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

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_process_single_ocr, args): args for args in args_list
                }

                for future in as_completed(future_map):
                    args = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                        # Persist
                        content_hash = hash_map[args[0]]
                        out_path = output_dir / f"{content_hash}.json"
                        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                    except Exception as exc:
                        logger.error("OCR failed for %s: %s", args[0], exc)
                    progress.advance(task)

        return results

    def _process_sequential(
        self,
        to_process: list[tuple[Path, str]],
        output_dir: Path,
    ) -> list[ProcessingResult]:
        """Process files sequentially with progress bar."""
        results: list[ProcessingResult] = []

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

                out_path = output_dir / f"{content_hash}.json"
                out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                progress.advance(task)

        return results
