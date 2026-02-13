"""Redaction analysis processor.

Scans PDFs for redaction regions using PyMuPDF, classifies them as
proper (no text under), bad_overlay (text accessible), or recoverable
(text extractable), and produces RedactionAnalysisResult reports.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.forensics import (
    Redaction,
    RedactionAnalysisResult,
)

logger = logging.getLogger(__name__)
console = Console()


class RedactionAnalyzer:
    """Analyze PDFs for redactions and attempt text recovery."""

    def __init__(self) -> None:
        # Import check
        try:
            from epstein_pipeline.processors.pymupdf_extractor import PyMuPdfExtractor

            self._extractor = PyMuPdfExtractor()
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for redaction analysis. Install with: pip install pymupdf"
            )

    def analyze_file(self, path: Path) -> RedactionAnalysisResult:
        """Analyze a single PDF for redactions.

        Returns a RedactionAnalysisResult with classified redactions
        and any recovered text.
        """
        import fitz

        doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        doc_id = f"redact-{doc_hash}"

        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()

        regions = self._extractor.detect_redactions(path)

        redactions = []
        proper = 0
        bad_overlay = 0
        recoverable = 0
        recovered_fragments = []

        for region in regions:
            classification = region.classification
            if classification == "proper":
                proper += 1
            elif classification == "bad_overlay":
                bad_overlay += 1
            elif classification == "recoverable":
                recoverable += 1
                if region.text_under:
                    recovered_fragments.append(region.text_under)

            redactions.append(
                Redaction(
                    page=region.page_number,
                    x0=region.x0,
                    y0=region.y0,
                    x1=region.x1,
                    y1=region.y1,
                    classification=classification,
                    recovered_text=region.text_under,
                )
            )

        return RedactionAnalysisResult(
            source_path=str(path),
            document_id=doc_id,
            page_count=page_count,
            redactions=redactions,
            total_redactions=len(redactions),
            proper=proper,
            bad_overlay=bad_overlay,
            recoverable=recoverable,
            recovered_text_fragments=recovered_fragments,
        )

    def analyze_batch(
        self,
        paths: list[Path],
        output_dir: Path,
        max_workers: int = 1,
    ) -> list[RedactionAnalysisResult]:
        """Analyze multiple PDFs for redactions.

        Results are written to *output_dir* as JSON files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[RedactionAnalysisResult] = []

        from epstein_pipeline.utils.parallel import run_parallel

        if max_workers > 1:
            results = run_parallel(
                self.analyze_file,
                paths,
                max_workers=max_workers,
                label="Analyzing redactions",
                use_processes=False,  # PyMuPDF objects aren't easily picklable
            )
        else:
            from epstein_pipeline.utils.progress import create_progress

            with create_progress() as progress:
                task = progress.add_task("Analyzing redactions", total=len(paths))
                for path in paths:
                    try:
                        result = self.analyze_file(path)
                        results.append(result)
                    except Exception as exc:
                        logger.error("Redaction analysis failed for %s: %s", path, exc)
                    progress.advance(task)

        # Write results
        for result in results:
            out_path = output_dir / f"{result.document_id}.json"
            out_path.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )

        # Print summary
        total_redactions = sum(r.total_redactions for r in results)
        total_recoverable = sum(r.recoverable for r in results)
        console.print(f"\n  [green]Analyzed {len(results)} files[/green]")
        console.print(f"  Total redactions found: {total_redactions:,}")
        console.print(f"  Recoverable: {total_recoverable:,}")
        console.print(f"  Results in: {output_dir}")

        return results
