"""Pydantic models for temporal event extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "meeting",
    "flight",
    "transaction",
    "communication",
    "legal_proceeding",
    "arrest",
    "testimony",
    "deposition",
    "court_filing",
    "property_transaction",
    "employment",
    "travel",
    "social_event",
    "abuse_allegation",
    "investigation",
    "media_report",
    "other",
]


class TemporalEvent(BaseModel):
    """A structured temporal event extracted from document text."""

    date: str = Field(
        description=(
            "Date in YYYY-MM-DD format. Use YYYY-MM for month-only, "
            "YYYY for year-only. Use ISO 8601 ranges for spans: YYYY-MM-DD/YYYY-MM-DD."
        ),
    )
    date_raw: str | None = Field(
        default=None,
        description=(
            "The original date text as written in the document "
            "(e.g., 'March 15th', 'the following Tuesday')."
        ),
    )
    event_type: EventType = Field(description="Category of the event.")
    description: str = Field(
        description="One to two sentence description of what happened.",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="Names of people involved in this event.",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Locations where this event occurred.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the extraction accuracy (0.0-1.0).",
    )
    source_chunk: str | None = Field(
        default=None,
        description="The text chunk this event was extracted from (truncated).",
    )


class DocumentTemporalResult(BaseModel):
    """All temporal events extracted from a single document."""

    document_id: str
    events: list[TemporalEvent] = Field(default_factory=list)
    document_date_context: str | None = Field(
        default=None,
        description="Known document date for resolving relative references.",
    )


class TemporalExtractionBatch(BaseModel):
    """Batch result from temporal extraction across multiple documents."""

    results: list[DocumentTemporalResult] = Field(default_factory=list)
    total_events: int = 0
    total_documents: int = 0
