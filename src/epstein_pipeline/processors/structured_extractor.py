"""LLM-powered structured data extraction using Instructor + Pydantic schemas.

Extracts structured fields from legal documents that regex misses:
  - Case references (case number, court, parties)
  - Financial amounts with context
  - Dates with event descriptions
  - People with roles
  - Addresses and locations

Uses Instructor library with OpenAI-compatible APIs (Ollama, OpenAI, Anthropic).
Pydantic schemas guarantee type-safe, validated output.

Usage:
    extractor = StructuredExtractor(settings)
    result = extractor.extract(document_text)
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from epstein_pipeline.config import Settings

logger = logging.getLogger(__name__)


# ── Pydantic schemas for structured extraction ──────────────────────────


class CaseReference(BaseModel):
    """A legal case reference found in a document."""

    case_number: str = Field(description="Full case number (e.g., '15-cv-07433')")
    court: str | None = Field(default=None, description="Court name (e.g., 'SDNY')")
    short_name: str | None = Field(default=None, description="Short case name (e.g., 'Giuffre v. Maxwell')")
    parties: list[str] = Field(default_factory=list, description="Named parties")


class FinancialAmount(BaseModel):
    """A monetary amount with context."""

    amount: float = Field(description="Dollar amount")
    currency: str = Field(default="USD", description="Currency code")
    context: str = Field(description="What this amount is for (e.g., 'wire transfer to Deutsche Bank')")
    date: str | None = Field(default=None, description="Date associated with the amount")
    from_entity: str | None = Field(default=None, description="Who paid / source of funds")
    to_entity: str | None = Field(default=None, description="Who received / destination")


class PersonMention(BaseModel):
    """A person mentioned in the document with their role."""

    name: str = Field(description="Full name as written in the document")
    role: str | None = Field(default=None, description="Role or title (e.g., 'attorney', 'witness', 'defendant')")
    organization: str | None = Field(default=None, description="Associated organization")


class DateEvent(BaseModel):
    """A date with associated event description."""

    date: str = Field(description="Date in YYYY-MM-DD format if possible, otherwise as written")
    event: str = Field(description="What happened on this date")
    location: str | None = Field(default=None, description="Where the event occurred")


class LocationMention(BaseModel):
    """A location mentioned in the document."""

    name: str = Field(description="Location name")
    type: str | None = Field(default=None, description="Type: address, city, country, property, etc.")
    context: str | None = Field(default=None, description="Why this location is mentioned")


class DocumentExtractionResult(BaseModel):
    """Complete structured extraction from a document."""

    case_references: list[CaseReference] = Field(default_factory=list)
    financial_amounts: list[FinancialAmount] = Field(default_factory=list)
    persons: list[PersonMention] = Field(default_factory=list)
    dates: list[DateEvent] = Field(default_factory=list)
    locations: list[LocationMention] = Field(default_factory=list)
    document_type: str | None = Field(default=None, description="Detected document type")
    summary: str | None = Field(default=None, description="One-sentence summary of the document")


# ── Extraction prompts ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a legal document analyst specializing in the Jeffrey Epstein case files.
Extract structured data from the provided document text. Be precise and extract only what is
explicitly stated in the text. Do not infer or hallucinate information not present in the document.

For case references, extract the full case number, court, and parties.
For financial amounts, include the context of what the money was for.
For persons, note their role in the document (witness, attorney, defendant, etc.).
For dates, convert to YYYY-MM-DD format when possible.
For locations, note the type (address, city, property name, etc.)."""


class StructuredExtractor:
    """Extract structured data from legal documents using LLM + Pydantic.

    Uses the Instructor library to wrap OpenAI-compatible APIs with type-safe
    Pydantic output schemas. Works with Ollama (free, local), OpenAI, or
    Anthropic as the backend.

    Parameters
    ----------
    settings : Settings
        Pipeline settings (reads OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    backend : str
        LLM backend: "ollama" (default, free), "openai", or "anthropic"
    model : str | None
        Model name override (default: llama3.2 for Ollama, gpt-4o-mini for OpenAI)
    """

    def __init__(
        self,
        settings: Settings,
        backend: str = "ollama",
        model: str | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.model = model
        self._client = None

    def _ensure_client(self):
        """Lazy-load the Instructor-wrapped client."""
        if self._client is not None:
            return

        try:
            import instructor
        except ImportError:
            raise ImportError(
                "instructor is required for structured extraction. "
                "Install with: pip install instructor"
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
            import openai
            import os

            base_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY") or self.settings.openai_api_key,
            )
            self._client = instructor.from_openai(base_client)
            self.model = self.model or "gpt-4o-mini"

        elif self.backend == "anthropic":
            import anthropic
            import os

            base_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            self._client = instructor.from_anthropic(base_client)
            self.model = self.model or "claude-sonnet-4-20250514"

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        logger.info("Structured extractor initialized: %s / %s", self.backend, self.model)

    def extract(self, text: str, document_id: str = "") -> DocumentExtractionResult:
        """Extract structured data from a document text.

        Parameters
        ----------
        text : str
            Document text (OCR, transcript, or raw text). Truncated to 4000 chars.
        document_id : str
            Optional document ID for logging.

        Returns
        -------
        DocumentExtractionResult
            Pydantic model with all extracted structured fields.
        """
        self._ensure_client()

        # Truncate to fit context window
        truncated = text[:4000] if len(text) > 4000 else text

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                response_model=DocumentExtractionResult,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract structured data from this document:\n\n{truncated}"},
                ],
                max_retries=2,
            )
            return result
        except Exception as exc:
            logger.error("Structured extraction failed for %s: %s", document_id, exc)
            return DocumentExtractionResult()

    def extract_batch(
        self,
        texts: list[tuple[str, str]],  # [(text, document_id), ...]
        output_dir: Path | None = None,
    ) -> list[DocumentExtractionResult]:
        """Extract structured data from multiple documents.

        Parameters
        ----------
        texts : list[tuple[str, str]]
            List of (text, document_id) tuples.
        output_dir : Path | None
            Optional directory to save results as JSON files.
        """
        from rich.console import Console
        from epstein_pipeline.utils.progress import create_progress

        console = Console()
        results: list[DocumentExtractionResult] = []

        console.print(f"  Extracting structured data from {len(texts)} documents...")

        with create_progress() as progress:
            task = progress.add_task("Extracting", total=len(texts))
            for text, doc_id in texts:
                result = self.extract(text, doc_id)
                results.append(result)

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    out_path = output_dir / f"{doc_id}.json"
                    out_path.write_text(
                        result.model_dump_json(indent=2),
                        encoding="utf-8",
                    )

                progress.advance(task)

        # Summary
        total_cases = sum(len(r.case_references) for r in results)
        total_amounts = sum(len(r.financial_amounts) for r in results)
        total_persons = sum(len(r.persons) for r in results)
        total_dates = sum(len(r.dates) for r in results)

        console.print(f"\n  [green]Extracted from {len(results)} documents:[/green]")
        console.print(f"    Case references: {total_cases}")
        console.print(f"    Financial amounts: {total_amounts}")
        console.print(f"    Persons: {total_persons}")
        console.print(f"    Dates: {total_dates}")

        return results
