"""Shared test fixtures for the Epstein Pipeline test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Document, Person, ProcessingResult
from epstein_pipeline.models.registry import PersonRegistry


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create a Settings instance with temp directories."""
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / ".cache",
        persons_registry_path=tmp_path / "data" / "persons-registry.json",
    )


@pytest.fixture
def sample_persons() -> list[Person]:
    """A small set of test persons."""
    return [
        Person(
            id="p-0001",
            slug="jeffrey-epstein",
            name="Jeffrey Epstein",
            aliases=["Jeff Epstein", "JE"],
            category="convicted",
            shortBio="Convicted sex trafficker",
        ),
        Person(
            id="p-0002",
            slug="ghislaine-maxwell",
            name="Ghislaine Maxwell",
            aliases=["G. Maxwell"],
            category="convicted",
        ),
        Person(
            id="p-0003",
            slug="bill-clinton",
            name="Bill Clinton",
            aliases=["William Clinton", "President Clinton"],
            category="politician",
        ),
    ]


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small set of test documents."""
    return [
        Document(
            id="doc-001",
            title="EFTA Filing About Jeffrey Epstein Financial Records",
            source="efta",
            category="financial",
            summary="Financial records related to Jeffrey Epstein",
            personIds=["p-0001"],
            tags=["financial", "efta"],
            batesRange="EFTA00039025-EFTA00039030",
            ocrText="This document contains information about Jeffrey Epstein and Bill Clinton.",
        ),
        Document(
            id="doc-002",
            title="FBI Investigation Report",
            source="fbi",
            category="investigation",
            summary="FBI investigation into trafficking",
            personIds=["p-0001", "p-0002"],
            tags=["fbi", "investigation"],
            ocrText="Ghislaine Maxwell coordinated travel arrangements.",
        ),
        Document(
            id="doc-003",
            title="Court Filing - Testimony",
            source="court-filing",
            category="legal",
            personIds=["p-0001"],
            tags=["court", "testimony"],
        ),
    ]


@pytest.fixture
def registry_file(tmp_path: Path, sample_persons: list[Person]) -> Path:
    """Write a test persons registry JSON file."""
    path = tmp_path / "persons-registry.json"
    data = [p.model_dump() for p in sample_persons]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def person_registry(registry_file: Path) -> PersonRegistry:
    """Load a PersonRegistry from the test file."""
    return PersonRegistry.from_json(registry_file)


@pytest.fixture
def sample_processing_result() -> ProcessingResult:
    """A sample ProcessingResult."""
    return ProcessingResult(
        source_path="/tmp/test.pdf",
        document=Document(
            id="ocr-abc123",
            title="Test OCR Document",
            source="efta",
            category="other",
            ocrText="Jeffrey Epstein met with Bill Clinton in New York.",
            tags=["ocr"],
        ),
        errors=[],
        warnings=[],
        processing_time_ms=1500,
    )


@pytest.fixture
def json_output_dir(tmp_path: Path, sample_processing_result: ProcessingResult) -> Path:
    """Create a directory with sample processing result JSON files."""
    out_dir = tmp_path / "json_output"
    out_dir.mkdir()

    for i in range(3):
        result = sample_processing_result.model_copy()
        if result.document:
            result.document.id = f"doc-{i:03d}"
        path = out_dir / f"result_{i}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    return out_dir
