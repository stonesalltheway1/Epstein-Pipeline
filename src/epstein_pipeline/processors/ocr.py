"""OCR processor with multi-backend support and confidence scoring.

Backends (ordered by speed):
- ``pymupdf``  -- PyMuPDF text-layer extraction (instant, no OCR)
- ``surya``    -- Surya OCR (fast, structured, bounding boxes + confidence)
- ``olmocr``   -- olmOCR / Allen AI VLM-based (GPU-heavy, handwriting support)
- ``docling``  -- IBM Docling (table-aware, layout analysis)
- ``auto``     -- Fallback chain: pymupdf -> surya -> docling

The ``auto`` mode tries backends in order until acceptable text is produced.
olmOCR is excluded from auto mode due to high GPU cost; select it explicitly
with ``--ocr-backend=olmocr``.
"""

from __future__ import annotations

import hashlib
import logging
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

from epstein_pipeline.config import OcrBackend, Settings
from epstein_pipeline.models.document import Document, ProcessingResult

logger = logging.getLogger(__name__)

# ── Default fallback chain for auto mode (olmocr excluded) ──────────────
_AUTO_CHAIN: list[str] = ["pymupdf", "surya", "docling"]


# ---------------------------------------------------------------------------
# OCR result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OcrResult:
    """Result from a single backend extraction attempt."""

    text: str
    confidence: float  # average confidence across all pages (0.0 - 1.0)
    backend_used: str
    page_confidences: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract backend base class
# ---------------------------------------------------------------------------


class OcrBackendBase(ABC):
    """Abstract base class for OCR backends."""

    name: str = "base"

    @abstractmethod
    def extract(self, path: Path) -> OcrResult:
        """Extract text from a PDF file.

        Returns an ``OcrResult`` with text content and confidence scores.
        Raises ``RuntimeError`` if the backend is unavailable or extraction
        fails unrecoverably.
        """

    @staticmethod
    def _heuristic_confidence(text: str) -> float:
        """Compute a simple heuristic confidence score for extracted text.

        Useful for backends that don't provide native confidence values
        (PyMuPDF, Docling).  Considers:
        - Ratio of printable characters to total characters
        - Average word length (too short or too long = suspicious)
        - Presence of common English stop words
        """
        if not text or not text.strip():
            return 0.0

        # Printable character ratio
        printable = set(string.printable)
        total_chars = len(text)
        printable_count = sum(1 for c in text if c in printable)
        printable_ratio = printable_count / total_chars if total_chars else 0.0

        # Average word length
        words = text.split()
        if not words:
            return 0.0
        avg_word_len = sum(len(w) for w in words) / len(words)
        # Penalize if average word length is outside typical range (2-12)
        word_len_score = 1.0
        if avg_word_len < 2.0:
            word_len_score = avg_word_len / 2.0
        elif avg_word_len > 12.0:
            word_len_score = max(0.3, 1.0 - (avg_word_len - 12.0) / 20.0)

        # Stop-word presence (rough check that text is real English)
        stop_words = {"the", "and", "of", "to", "in", "a", "is", "that", "for", "it"}
        lower_words = {w.lower().strip(string.punctuation) for w in words}
        stop_hits = len(stop_words & lower_words)
        stop_score = min(1.0, stop_hits / 3.0)  # 3+ stop words = max score

        # Weighted combination
        confidence = 0.40 * printable_ratio + 0.30 * word_len_score + 0.30 * stop_score
        return round(min(1.0, max(0.0, confidence)), 4)

    @staticmethod
    def _page_heuristic_confidences(pages_text: list[str]) -> list[float]:
        """Compute per-page heuristic confidence scores."""
        return [OcrBackendBase._heuristic_confidence(page) for page in pages_text]


# ---------------------------------------------------------------------------
# PyMuPDF backend
# ---------------------------------------------------------------------------


class PyMuPDFBackend(OcrBackendBase):
    """Extract existing text layers from PDF using PyMuPDF (fitz).

    This is the fastest backend -- it reads embedded text without performing
    actual OCR.  Returns empty text for scanned/image-only PDFs.
    """

    name = "pymupdf"

    def extract(self, path: Path) -> OcrResult:
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Install with: pip install pymupdf")

        doc = fitz.open(str(path))
        pages_text: list[str] = []
        try:
            for page in doc:
                pages_text.append(page.get_text())
        finally:
            doc.close()

        full_text = "\n\n".join(pages_text).strip()
        page_confs = self._page_heuristic_confidences(pages_text)
        avg_conf = sum(page_confs) / len(page_confs) if page_confs else 0.0

        return OcrResult(
            text=full_text,
            confidence=round(avg_conf, 4),
            backend_used=self.name,
            page_confidences=page_confs,
        )


# ---------------------------------------------------------------------------
# Surya backend
# ---------------------------------------------------------------------------


class SuryaBackend(OcrBackendBase):
    """OCR using Surya (surya-ocr).

    Handles scanned PDFs, poor-quality images, and multi-column layouts.
    Returns structured text with bounding boxes and character-level
    confidence scores.  Works on CPU (slower) or GPU (fast with CUDA).
    """

    name = "surya"

    def extract(self, path: Path) -> OcrResult:
        try:
            from surya.detection import DetectionPredictor  # type: ignore[import-untyped]
            from surya.ocr import run_ocr  # type: ignore[import-untyped]
            from surya.recognition import RecognitionPredictor  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("surya-ocr not installed. Install with: pip install surya-ocr")

        # Convert PDF pages to images
        try:
            from pdf2image import convert_from_path  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "pdf2image not installed. Install with: pip install pdf2image "
                "(requires poppler system dependency)"
            )

        images = convert_from_path(str(path), dpi=200)
        if not images:
            return OcrResult(
                text="",
                confidence=0.0,
                backend_used=self.name,
                page_confidences=[],
            )

        # Initialize Surya predictors
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        # Run OCR on all page images
        languages = [["en"]] * len(images)
        ocr_results = run_ocr(
            images,
            languages,
            det_predictor,
            rec_predictor,
        )

        pages_text: list[str] = []
        page_confidences: list[float] = []

        for page_result in ocr_results:
            lines: list[str] = []
            line_confs: list[float] = []

            for text_line in page_result.text_lines:
                lines.append(text_line.text)
                # Surya provides per-line confidence
                if hasattr(text_line, "confidence") and text_line.confidence is not None:
                    line_confs.append(float(text_line.confidence))

            page_text = "\n".join(lines)
            pages_text.append(page_text)

            if line_confs:
                page_confidences.append(round(sum(line_confs) / len(line_confs), 4))
            else:
                # Fall back to heuristic if Surya doesn't provide confidence
                page_confidences.append(self._heuristic_confidence(page_text))

        full_text = "\n\n".join(pages_text).strip()
        avg_conf = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0

        return OcrResult(
            text=full_text,
            confidence=round(avg_conf, 4),
            backend_used=self.name,
            page_confidences=page_confidences,
        )


# ---------------------------------------------------------------------------
# olmOCR backend
# ---------------------------------------------------------------------------


class OlmOcrBackend(OcrBackendBase):
    """VLM-based OCR using olmOCR (Allen AI).

    Uses a vision-language model to understand document context, including
    handwriting, complex layouts, and degraded scans.  Requires a GPU with
    at least 8 GB VRAM.  Only used when explicitly selected via
    ``--ocr-backend=olmocr`` (excluded from auto fallback chain).
    """

    name = "olmocr"

    def extract(self, path: Path) -> OcrResult:
        try:
            from olmocr.pipeline import OlmOcrPipeline  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "olmocr not installed. Install with: pip install olmocr "
                "(requires GPU with 8GB+ VRAM)"
            )

        try:
            pipeline = OlmOcrPipeline()
            result = pipeline.process(str(path))
        except Exception as exc:
            raise RuntimeError(f"olmOCR processing failed: {exc}") from exc

        # olmOCR returns structured page results
        pages_text: list[str] = []
        page_confidences: list[float] = []

        if hasattr(result, "pages"):
            for page in result.pages:
                text = page.text if hasattr(page, "text") else str(page)
                pages_text.append(text)
                # olmOCR may provide page-level confidence
                if hasattr(page, "confidence") and page.confidence is not None:
                    page_confidences.append(float(page.confidence))
                else:
                    page_confidences.append(self._heuristic_confidence(text))
        elif isinstance(result, str):
            # Fallback: result is just a string
            pages_text.append(result)
            page_confidences.append(self._heuristic_confidence(result))
        else:
            # Try to get text from the result object
            text = str(result)
            pages_text.append(text)
            page_confidences.append(self._heuristic_confidence(text))

        full_text = "\n\n".join(pages_text).strip()
        avg_conf = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0

        return OcrResult(
            text=full_text,
            confidence=round(avg_conf, 4),
            backend_used=self.name,
            page_confidences=page_confidences,
        )


# ---------------------------------------------------------------------------
# Docling backend
# ---------------------------------------------------------------------------


class DoclingBackend(OcrBackendBase):
    """OCR using IBM Docling.

    Good for table-aware extraction and complex document layouts.  Slower
    than PyMuPDF and Surya but handles structured documents well.
    """

    name = "docling"

    def extract(self, path: Path) -> OcrResult:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("Docling not installed. Install with: pip install docling")

        converter = DocumentConverter()
        result = converter.convert(str(path))
        md_text = result.document.export_to_markdown()

        if not md_text or not md_text.strip():
            return OcrResult(
                text="",
                confidence=0.0,
                backend_used=self.name,
                page_confidences=[],
            )

        # Docling doesn't provide per-page separation easily, so we treat
        # the whole document as one block for confidence estimation.
        conf = self._heuristic_confidence(md_text)
        return OcrResult(
            text=md_text.strip(),
            confidence=conf,
            backend_used=self.name,
            page_confidences=[conf],
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_CLASSES: dict[str, type[OcrBackendBase]] = {
    "pymupdf": PyMuPDFBackend,
    "surya": SuryaBackend,
    "olmocr": OlmOcrBackend,
    "docling": DoclingBackend,
}


def _get_backend(name: str) -> OcrBackendBase:
    """Instantiate a backend by name."""
    cls = _BACKEND_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend: {name!r}. Choose from: {', '.join(_BACKEND_CLASSES)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Module-level function for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------


def _process_single_ocr(args: tuple[str, str, str, float, list[str]]) -> ProcessingResult:
    """Process a single PDF file for OCR.

    This is a module-level function so it can be pickled for use with
    ProcessPoolExecutor.  Accepts a tuple of:
        (file_path, backend, spacy_model, confidence_threshold, fallback_chain)
    """
    file_path_str, backend, _, confidence_threshold, fallback_chain = args
    path = Path(file_path_str)
    start_ms = time.monotonic_ns() // 1_000_000
    errors: list[str] = []
    warnings: list[str] = []
    ocr_result: OcrResult | None = None

    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = f"ocr-{content_hash[:12]}"

    # Determine which backends to try
    if backend == OcrBackend.AUTO or backend == "auto":
        # Use fallback chain, excluding olmocr (too expensive for auto)
        chain = [b for b in fallback_chain if b != "olmocr"]
        if not chain:
            chain = list(_AUTO_CHAIN)
    else:
        # Explicit backend selection -- single backend, no fallback
        chain = [backend if isinstance(backend, str) else backend.value]

    # Walk the chain until we get acceptable text
    for backend_name in chain:
        try:
            be = _get_backend(backend_name)
            result = be.extract(path)

            if result.text and result.text.strip():
                # Check if quality is acceptable
                if result.confidence >= confidence_threshold:
                    ocr_result = result
                    break
                else:
                    # Text extracted but below confidence threshold
                    warnings.append(
                        f"{backend_name}: extracted text with low confidence "
                        f"({result.confidence:.2f} < {confidence_threshold:.2f})"
                    )
                    # Keep as candidate if nothing better found
                    if ocr_result is None or result.confidence > ocr_result.confidence:
                        ocr_result = result
            else:
                warnings.append(f"{backend_name}: produced empty text")
        except RuntimeError as exc:
            # Backend not installed or failed
            warnings.append(f"{backend_name}: {exc}")
        except Exception as exc:
            errors.append(f"{backend_name} failed for {path.name}: {exc}")

    # Flag low-confidence pages for manual review
    low_conf_pages: list[int] = []
    if ocr_result and ocr_result.page_confidences:
        low_conf_pages = [
            i + 1  # 1-indexed page numbers
            for i, conf in enumerate(ocr_result.page_confidences)
            if conf < confidence_threshold
        ]
        if low_conf_pages:
            warnings.append(
                f"Pages below confidence threshold ({confidence_threshold}): {low_conf_pages}"
            )

    md_text = ocr_result.text if ocr_result else ""
    ocr_confidence = ocr_result.confidence if ocr_result else None

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
        ocr_confidence=ocr_confidence,
    )


# ---------------------------------------------------------------------------
# OcrProcessor -- orchestrates backend selection and batch processing
# ---------------------------------------------------------------------------


class OcrProcessor:
    """Process PDF files through OCR text extraction.

    Supports multiple backends with automatic fallback:

    - ``pymupdf``  -- instant text-layer extraction (no OCR)
    - ``surya``    -- fast OCR with confidence scores, multi-column support
    - ``olmocr``   -- VLM-based OCR (GPU required, explicit selection only)
    - ``docling``  -- IBM Docling (tables, layout analysis)
    - ``auto``     -- fallback chain: pymupdf -> surya -> docling

    Provides per-page confidence scoring and flags low-quality pages for
    manual review.
    """

    def __init__(
        self,
        config: Settings,
        backend: str | OcrBackend | None = None,
    ) -> None:
        self.config = config
        # Allow override via constructor or fall back to config
        if backend is not None:
            self.backend = backend if isinstance(backend, str) else backend.value
        else:
            self.backend = config.ocr_backend.value
        self.confidence_threshold = config.ocr_confidence_threshold
        self.fallback_chain = list(config.ocr_fallback_chain)

    def process_file(self, path: Path) -> ProcessingResult:
        """OCR a single PDF and return a ProcessingResult."""
        return _process_single_ocr(
            (
                str(path),
                self.backend,
                self.config.spacy_model,
                self.confidence_threshold,
                self.fallback_chain,
            )
        )

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
        args_list = [
            (
                str(p),
                self.backend,
                self.config.spacy_model,
                self.confidence_threshold,
                self.fallback_chain,
            )
            for p, _ in to_process
        ]
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
                        # Persist result
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
