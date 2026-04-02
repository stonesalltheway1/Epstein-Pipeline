"""LLM-powered temporal event extraction using Instructor + Pydantic schemas.

Extracts structured timeline events from legal documents, depositions,
and correspondence. Handles:
  - Explicit dates ("March 15, 2005")
  - Relative dates ("the following Tuesday", "three days later")
  - Date ranges ("between January and March 2006")
  - Imprecise dates ("sometime in early 2004")

Uses chunking to handle long documents, then merges/deduplicates events.

Usage:
    extractor = TemporalExtractor(settings)
    result = extractor.extract(document_text, document_id="doc-001")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from epstein_pipeline.config import Settings
from epstein_pipeline.models.temporal import (
    DocumentTemporalResult,
    TemporalEvent,
    TemporalExtractionBatch,
)

logger = logging.getLogger(__name__)


# ── Instructor response model (wraps TemporalEvent list) ────────────────


class _ChunkExtractionResponse(BaseModel):
    """LLM response model for a single chunk."""

    events: list[TemporalEvent] = Field(default_factory=list)


# ── Extraction prompt ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a timeline analyst specializing in the Jeffrey Epstein case files.
Extract every temporal event from the provided document chunk. For each event, provide:

1. **date**: In YYYY-MM-DD format when possible. For month-only precision use YYYY-MM.
   For year-only use YYYY. For date ranges use YYYY-MM-DD/YYYY-MM-DD.
   If a relative date is used ("three days later"), resolve it relative to the document
   date context if provided, otherwise note it as-is.

2. **date_raw**: The exact date text as written in the original document.

3. **event_type**: One of: meeting, flight, transaction, communication, legal_proceeding,
   arrest, testimony, deposition, court_filing, property_transaction, employment,
   travel, social_event, abuse_allegation, investigation, media_report, other.

4. **description**: 1-2 sentence description of what happened.

5. **participants**: Names of people involved.

6. **locations**: Where the event occurred.

7. **confidence**: 0.0-1.0 indicating your certainty about the date and event.
   Use 0.9+ for explicit dates with clear events.
   Use 0.5-0.8 for inferred/approximate dates.
   Use 0.3-0.5 for vague temporal references.

Extract ONLY events explicitly mentioned or clearly implied by the text.
Do NOT infer events not supported by the document content."""


class TemporalExtractor:
    """Extract temporal events from legal documents using LLM + Pydantic.

    Follows the same Instructor pattern as StructuredExtractor.
    Chunks long documents, extracts events per chunk, then deduplicates.

    Parameters
    ----------
    settings : Settings
        Pipeline settings.
    backend : str | None
        LLM backend override. Default: from settings.temporal_llm_provider.
    model : str | None
        Model name override. Default: from settings.temporal_llm_model.
    """

    def __init__(
        self,
        settings: Settings,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or settings.temporal_llm_provider
        self.model = model or settings.temporal_llm_model
        self.chunk_size = settings.temporal_chunk_size
        self.chunk_overlap = settings.temporal_chunk_overlap
        self.max_events_per_chunk = settings.temporal_max_events_per_chunk
        self.confidence_threshold = settings.temporal_confidence_threshold
        self._client = None

    def _ensure_client(self):
        """Lazy-load the Instructor-wrapped client (same pattern as StructuredExtractor)."""
        if self._client is not None:
            return

        if self.backend not in ("ollama", "openai", "anthropic"):
            raise ValueError(f"Unknown backend: {self.backend}")

        try:
            import instructor
        except ImportError:
            raise ImportError(
                "instructor is required for temporal extraction. "
                "Install with: pip install 'epstein-pipeline[structured]'"
            )

        if self.backend == "ollama":
            import openai

            base_client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            self._client = instructor.from_openai(base_client)
            self.model = self.model or "llama3.2"

        elif self.backend == "openai":
            import os

            import openai

            base_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            self._client = instructor.from_openai(base_client)
            self.model = self.model or "gpt-4o-mini"

        elif self.backend == "anthropic":
            import os

            import anthropic

            base_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            self._client = instructor.from_anthropic(base_client)
            self.model = self.model or "claude-sonnet-4-20250514"

        logger.info("Temporal extractor initialized: %s / %s", self.backend, self.model)

    # ── Chunking ─────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks, respecting sentence boundaries."""
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                search_zone = text[max(start, end - 200) : end]
                last_period = search_zone.rfind(". ")
                if last_period != -1:
                    end = max(start, end - 200) + last_period + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    # ── Single-chunk extraction ──────────────────────────────────────────

    def _extract_chunk(
        self,
        chunk_text: str,
        document_date: str | None = None,
    ) -> list[TemporalEvent]:
        """Extract temporal events from a single text chunk."""
        self._ensure_client()

        context_note = ""
        if document_date:
            context_note = f"\n\nDocument date context: {document_date}"

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                response_model=_ChunkExtractionResponse,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Extract temporal events from this document chunk:"
                            f"{context_note}\n\n{chunk_text}"
                        ),
                    },
                ],
                max_retries=2,
            )

            # Filter by confidence and cap events
            events = [
                e
                for e in result.events
                if e.confidence >= self.confidence_threshold
            ][: self.max_events_per_chunk]

            # Attach source chunk reference (truncated)
            for event in events:
                event.source_chunk = chunk_text[:200]

            return events

        except Exception as exc:
            logger.warning("Temporal extraction failed for chunk: %s", exc)
            return []

    # ── Date normalization ───────────────────────────────────────────────

    @staticmethod
    def normalize_date(date_str: str) -> str:
        """Normalize extracted dates to YYYY-MM-DD where possible.

        Handles:
        - "March 15, 2005" -> "2005-03-15"
        - "2005" -> "2005"
        - "January 2005" -> "2005-01"
        - Already formatted dates pass through
        """
        # Already in YYYY-MM-DD format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # Already in YYYY-MM format
        if re.match(r"^\d{4}-\d{2}$", date_str):
            return date_str

        # Year only
        if re.match(r"^\d{4}$", date_str):
            return date_str

        # Date range — normalize each side
        if "/" in date_str and not date_str.startswith("http"):
            parts = date_str.split("/", 1)
            left = TemporalExtractor.normalize_date(parts[0].strip())
            right = TemporalExtractor.normalize_date(parts[1].strip())
            return f"{left}/{right}"

        # Try dateutil parsing
        try:
            from dateutil import parser as dateutil_parser

            dt = dateutil_parser.parse(date_str, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        return date_str  # Return as-is if unparseable

    # ── Deduplication of events ──────────────────────────────────────────

    @staticmethod
    def _dedup_events(events: list[TemporalEvent]) -> list[TemporalEvent]:
        """Remove near-duplicate events from overlapping chunks.

        Two events are considered duplicates if they have the same date,
        same event_type, and >80% word overlap in their descriptions.
        """
        if len(events) <= 1:
            return events

        deduped: list[TemporalEvent] = []
        seen_keys: set[tuple[str, str]] = set()

        for event in events:
            key = (event.date, event.event_type)
            if key in seen_keys:
                # Check description similarity with existing events of same key
                is_dup = False
                for existing in deduped:
                    if (
                        existing.date == event.date
                        and existing.event_type == event.event_type
                    ):
                        words_a = set(event.description.lower().split())
                        words_b = set(existing.description.lower().split())
                        if words_a and words_b:
                            overlap = len(words_a & words_b) / max(
                                len(words_a), len(words_b)
                            )
                            if overlap > 0.8:
                                # Keep the one with higher confidence
                                if event.confidence > existing.confidence:
                                    deduped.remove(existing)
                                    deduped.append(event)
                                is_dup = True
                                break
                if not is_dup:
                    deduped.append(event)
            else:
                seen_keys.add(key)
                deduped.append(event)

        return deduped

    # ── Full document extraction ─────────────────────────────────────────

    def extract(
        self,
        text: str,
        document_id: str = "",
        document_date: str | None = None,
    ) -> DocumentTemporalResult:
        """Extract temporal events from a full document.

        Chunks the document, extracts events per chunk, normalizes dates,
        and deduplicates across overlapping chunks.

        Parameters
        ----------
        text : str
            Full document text.
        document_id : str
            Document identifier for logging and output.
        document_date : str | None
            Known document date (YYYY-MM-DD) for resolving relative references.

        Returns
        -------
        DocumentTemporalResult
            All extracted and deduplicated events.
        """
        if not text or not text.strip():
            return DocumentTemporalResult(document_id=document_id)

        chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
        logger.info(
            "Extracting temporal events from %s (%d chunks)",
            document_id,
            len(chunks),
        )

        all_events: list[TemporalEvent] = []
        for chunk in chunks:
            events = self._extract_chunk(chunk, document_date)
            all_events.extend(events)

        # Normalize dates
        for event in all_events:
            event.date = self.normalize_date(event.date)

        # Dedup across chunks
        deduped = self._dedup_events(all_events)

        # Sort by date
        deduped.sort(key=lambda e: e.date)

        logger.info(
            "Extracted %d events from %s (%d raw, %d after dedup)",
            len(deduped),
            document_id,
            len(all_events),
            len(deduped),
        )

        return DocumentTemporalResult(
            document_id=document_id,
            events=deduped,
            document_date_context=document_date,
        )

    # ── Batch extraction ─────────────────────────────────────────────────

    def extract_batch(
        self,
        texts: list[tuple[str, str, str | None]],
        output_dir: Path | None = None,
    ) -> TemporalExtractionBatch:
        """Extract temporal events from multiple documents.

        Parameters
        ----------
        texts : list[tuple[str, str, str | None]]
            List of (text, document_id, document_date) tuples.
        output_dir : Path | None
            Optional directory to save per-document results.
        """
        from rich.console import Console
        from rich.progress import Progress

        con = Console()
        results: list[DocumentTemporalResult] = []

        con.print(f"  Extracting temporal events from {len(texts)} documents...")

        with Progress(console=con) as progress:
            task = progress.add_task("Extracting events", total=len(texts))
            for text, doc_id, doc_date in texts:
                result = self.extract(text, doc_id, doc_date)
                results.append(result)

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    out_path = output_dir / f"{doc_id}-events.json"
                    out_path.write_text(
                        result.model_dump_json(indent=2),
                        encoding="utf-8",
                    )

                progress.advance(task)

        total_events = sum(len(r.events) for r in results)

        con.print(
            f"\n  [green]Extracted {total_events} events "
            f"from {len(results)} documents[/green]"
        )

        # Event type breakdown
        type_counts: dict[str, int] = {}
        for r in results:
            for e in r.events:
                type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        for et, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            con.print(f"    {et}: {count}")

        return TemporalExtractionBatch(
            results=results,
            total_events=total_events,
            total_documents=len(results),
        )
