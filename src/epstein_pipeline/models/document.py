"""Pydantic v2 models matching the Epstein Exposed TypeScript interfaces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums as Literal unions (mirrors the TS string-union types)
# ---------------------------------------------------------------------------

DocumentSource = Literal[
    "court-filing",
    "fbi",
    "foia",
    "financial",
    "travel",
    "correspondence",
    "media",
    "testimony",
    "police",
    "estate",
    "efta",
    "epstein-docs",
    "kaggle",
    "huggingface",
    "sea-doughnut",
    # DOJ dataset-specific EFTA sources (Sea_Doughnut v2)
    "efta-ds1",
    "efta-ds2",
    "efta-ds3",
    "efta-ds4",
    "efta-ds5",
    "efta-ds6",
    "efta-ds7",
    "efta-ds8",
    "efta-ds9",
    "efta-ds10",
    "efta-ds11",
    "efta-ds12",
    "other",
]

DocumentCategory = Literal[
    "legal",
    "financial",
    "travel",
    "communications",
    "investigation",
    "media",
    "government",
    "personal",
    "medical",
    "property",
    "corporate",
    "intelligence",
    "other",
]

VerificationStatus = Literal[
    "verified",
    "unverified",
    "disputed",
    "redacted",
]


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """A single document in the Epstein case file corpus."""

    id: str
    title: str
    date: str | None = None  # YYYY-MM-DD format
    source: DocumentSource
    category: DocumentCategory
    summary: str | None = None
    personIds: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    pdfUrl: str | None = None
    sourceUrl: str | None = None
    archiveUrl: str | None = None
    pageCount: int | None = None
    batesRange: str | None = None  # e.g. "EFTA00039025-EFTA00039030"
    ocrText: str | None = None
    locationIds: list[str] = Field(default_factory=list)
    verificationStatus: VerificationStatus | None = None


class Person(BaseModel):
    """A person referenced in the Epstein case files."""

    id: str  # format: "p-NNNN"
    slug: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    category: str
    shortBio: str | None = None


class EmailContact(BaseModel):
    """An email participant (sender or recipient)."""

    name: str
    email: str | None = None
    personSlug: str | None = None


class Email(BaseModel):
    """An email message from the Epstein case files."""

    id: str
    subject: str
    from_: EmailContact = Field(alias="from")
    to: list[EmailContact]
    cc: list[EmailContact] = Field(default_factory=list)
    date: str | None = None
    body: str
    personIds: list[str] = Field(default_factory=list)
    folder: str | None = None

    model_config = {"populate_by_name": True}


class Flight(BaseModel):
    """A flight log entry (Lolita Express / other aircraft)."""

    id: str
    date: str | None = None
    aircraft: str | None = None
    tailNumber: str | None = None
    origin: str | None = None
    destination: str | None = None
    passengerIds: list[str] = Field(default_factory=list)
    pilotIds: list[str] = Field(default_factory=list)


class EntityResult(BaseModel):
    """A single extracted entity from NER processing."""

    entity_type: str  # PERSON, ORG, PHONE, EMAIL_ADDR, CASE_NUMBER, FLIGHT_ID, etc.
    value: str
    confidence: float | None = None
    source: str = "spacy"  # 'spacy', 'gliner', 'regex'
    span: str | None = None  # original text context


class ProcessingResult(BaseModel):
    """Result of processing a single source file through the pipeline."""

    source_path: str
    document: Document | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: int

    # v1.0 additions
    ocr_confidence: float | None = None  # per-page average confidence
    classified_category: str | None = None  # auto-classified document type
    entities: list[EntityResult] = Field(default_factory=list)
    duplicate_cluster_id: str | None = None  # assigned dedup cluster
