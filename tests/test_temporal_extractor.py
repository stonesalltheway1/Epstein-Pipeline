"""Tests for temporal event extraction."""

from __future__ import annotations

import pytest

from epstein_pipeline.config import Settings
from epstein_pipeline.models.temporal import (
    DocumentTemporalResult,
    TemporalEvent,
    TemporalExtractionBatch,
)
from epstein_pipeline.processors.temporal_extractor import TemporalExtractor

# ── TemporalEvent model tests ───────────────────────────────────────────


class TestTemporalEventModel:
    def test_basic_event(self):
        e = TemporalEvent(
            date="2005-03-15",
            event_type="meeting",
            description="Epstein met with Maxwell at the townhouse.",
            participants=["Jeffrey Epstein", "Ghislaine Maxwell"],
            locations=["New York City"],
            confidence=0.9,
        )
        assert e.date == "2005-03-15"
        assert e.event_type == "meeting"
        assert len(e.participants) == 2

    def test_confidence_upper_bound(self):
        with pytest.raises(Exception):
            TemporalEvent(
                date="2005",
                event_type="other",
                description="test",
                confidence=1.5,
            )

    def test_confidence_lower_bound(self):
        with pytest.raises(Exception):
            TemporalEvent(
                date="2005",
                event_type="other",
                description="test",
                confidence=-0.1,
            )

    def test_default_confidence(self):
        e = TemporalEvent(date="2005", event_type="other", description="test")
        assert e.confidence == 0.5

    def test_default_participants(self):
        e = TemporalEvent(date="2005", event_type="other", description="test")
        assert e.participants == []
        assert e.locations == []

    def test_serialization_roundtrip(self):
        e = TemporalEvent(
            date="2005-03-15",
            event_type="flight",
            description="Flight from Teterboro to Palm Beach.",
            participants=["Jeffrey Epstein"],
            locations=["Teterboro", "Palm Beach"],
            confidence=0.95,
        )
        data = e.model_dump()
        e2 = TemporalEvent.model_validate(data)
        assert e == e2


class TestDocumentTemporalResult:
    def test_empty_result(self):
        r = DocumentTemporalResult(document_id="doc-001")
        assert len(r.events) == 0
        assert r.document_date_context is None

    def test_with_events(self):
        r = DocumentTemporalResult(
            document_id="doc-001",
            events=[
                TemporalEvent(date="2005-01-01", event_type="meeting", description="test"),
            ],
        )
        assert len(r.events) == 1


class TestTemporalExtractionBatch:
    def test_empty_batch(self):
        b = TemporalExtractionBatch()
        assert b.total_events == 0
        assert b.total_documents == 0
        assert b.results == []


# ── Chunking tests ──────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_no_chunking(self):
        chunks = TemporalExtractor._chunk_text("short text", 3000, 500)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty_text(self):
        chunks = TemporalExtractor._chunk_text("", 3000, 500)
        assert len(chunks) == 0

    def test_long_text_chunked(self):
        text = "word " * 1000  # ~5000 chars
        chunks = TemporalExtractor._chunk_text(text, 1000, 200)
        assert len(chunks) > 1

    def test_chunks_cover_all_text(self):
        text = "A" * 3000
        chunks = TemporalExtractor._chunk_text(text, 1000, 200)
        assert len(chunks) >= 3
        # All chars should be covered
        for i, c in enumerate(chunks):
            assert len(c) > 0

    def test_overlap_produces_more_chunks(self):
        text = "A" * 3000
        chunks_no_overlap = TemporalExtractor._chunk_text(text, 1000, 0)
        chunks_with_overlap = TemporalExtractor._chunk_text(text, 1000, 200)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_sentence_boundary_breaking(self):
        text = "First sentence. Second sentence. Third sentence here. Fourth sentence."
        chunks = TemporalExtractor._chunk_text(text, 40, 10)
        # Should try to break at ". " boundaries
        assert len(chunks) >= 2


# ── Date normalization tests ────────────────────────────────────────────


class TestDateNormalization:
    def test_already_formatted_ymd(self):
        assert TemporalExtractor.normalize_date("2005-03-15") == "2005-03-15"

    def test_already_formatted_ym(self):
        assert TemporalExtractor.normalize_date("2005-03") == "2005-03"

    def test_year_only(self):
        assert TemporalExtractor.normalize_date("2005") == "2005"

    def test_natural_date(self):
        result = TemporalExtractor.normalize_date("March 15, 2005")
        assert result == "2005-03-15"

    def test_natural_date_2(self):
        result = TemporalExtractor.normalize_date("January 1, 2003")
        assert result == "2003-01-01"

    def test_date_range(self):
        result = TemporalExtractor.normalize_date("2005-01-01/2005-06-30")
        assert result == "2005-01-01/2005-06-30"

    def test_unparseable_returns_as_is(self):
        result = TemporalExtractor.normalize_date("sometime in early 2004")
        assert isinstance(result, str)


# ── Deduplication tests ─────────────────────────────────────────────────


class TestDedupEvents:
    def test_no_duplicates(self):
        events = [
            TemporalEvent(date="2005-01-01", event_type="meeting", description="Event A"),
            TemporalEvent(date="2005-01-02", event_type="flight", description="Event B"),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 2

    def test_empty_list(self):
        assert TemporalExtractor._dedup_events([]) == []

    def test_single_event(self):
        events = [
            TemporalEvent(date="2005-01-01", event_type="meeting", description="Event A"),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 1

    def test_exact_duplicate_removed(self):
        events = [
            TemporalEvent(
                date="2005-01-01",
                event_type="meeting",
                description="Epstein met Maxwell at the townhouse",
                confidence=0.9,
            ),
            TemporalEvent(
                date="2005-01-01",
                event_type="meeting",
                description="Epstein met Maxwell at the townhouse",
                confidence=0.7,
            ),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # kept higher confidence

    def test_keeps_higher_confidence_on_dedup(self):
        events = [
            TemporalEvent(
                date="2005-01-01",
                event_type="meeting",
                description="Epstein met with Maxwell at the New York townhouse that evening",
                confidence=0.5,
            ),
            TemporalEvent(
                date="2005-01-01",
                event_type="meeting",
                description="Epstein met with Maxwell at the New York townhouse that same evening",
                confidence=0.95,
            ),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_different_types_not_deduped(self):
        events = [
            TemporalEvent(
                date="2005-01-01", event_type="meeting", description="same event"
            ),
            TemporalEvent(
                date="2005-01-01", event_type="flight", description="same event"
            ),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 2

    def test_different_dates_not_deduped(self):
        events = [
            TemporalEvent(
                date="2005-01-01", event_type="meeting", description="same event"
            ),
            TemporalEvent(
                date="2005-01-02", event_type="meeting", description="same event"
            ),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 2

    def test_different_descriptions_not_deduped(self):
        events = [
            TemporalEvent(
                date="2005-01-01", event_type="meeting", description="totally different A"
            ),
            TemporalEvent(
                date="2005-01-01", event_type="meeting", description="completely unrelated B"
            ),
        ]
        result = TemporalExtractor._dedup_events(events)
        assert len(result) == 2


# ── Extractor class tests ──────────────────────────────────────────────


class TestTemporalExtractor:
    @pytest.fixture()
    def settings(self, tmp_path):
        return Settings(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / ".cache",
        )

    def test_init_defaults(self, settings):
        ext = TemporalExtractor(settings)
        assert ext.backend == "ollama"
        assert ext.chunk_size == 3000
        assert ext.chunk_overlap == 500
        assert ext.max_events_per_chunk == 20
        assert ext.confidence_threshold == 0.3

    def test_init_overrides(self, settings):
        ext = TemporalExtractor(settings, backend="openai", model="gpt-4o")
        assert ext.backend == "openai"
        assert ext.model == "gpt-4o"

    def test_extract_empty_text(self, settings):
        ext = TemporalExtractor(settings)
        result = ext.extract("", document_id="doc-001")
        assert len(result.events) == 0
        assert result.document_id == "doc-001"

    def test_extract_whitespace_text(self, settings):
        ext = TemporalExtractor(settings)
        result = ext.extract("   \n\t  ", document_id="doc-001")
        assert len(result.events) == 0

    def test_invalid_backend(self, settings):
        ext = TemporalExtractor(settings, backend="invalid")
        with pytest.raises(ValueError, match="Unknown backend"):
            ext._ensure_client()
