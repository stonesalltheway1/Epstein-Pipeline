"""Tests for redaction analysis models."""

from epstein_pipeline.models.forensics import (
    RecoveredText,
    Redaction,
    RedactionAnalysisResult,
    RedactionScore,
)


def test_redaction_score_model():
    score = RedactionScore(
        document_id="doc-001",
        total_redactions=50,
        proper_redactions=40,
        improper_redactions=10,
        redaction_density=0.35,
        page_count=20,
    )
    assert score.document_id == "doc-001"
    assert score.total_redactions == 50
    assert score.improper_redactions == 10


def test_recovered_text_model():
    rt = RecoveredText(
        document_id="doc-001",
        page_number=3,
        text="Secret financial transfer to account #12345",
        confidence=0.85,
    )
    assert rt.page_number == 3
    assert "financial" in rt.text


def test_redaction_model():
    r = Redaction(
        page=5,
        x0=100.0,
        y0=200.0,
        x1=400.0,
        y1=220.0,
        classification="recoverable",
        recovered_text="Funds transferred",
    )
    assert r.classification == "recoverable"
    assert r.recovered_text is not None


def test_redaction_analysis_result():
    result = RedactionAnalysisResult(
        source_path="/tmp/test.pdf",
        document_id="redact-abc",
        page_count=15,
        redactions=[
            Redaction(page=1, x0=0, y0=0, x1=100, y1=20, classification="proper"),
            Redaction(
                page=2,
                x0=0,
                y0=0,
                x1=100,
                y1=20,
                classification="recoverable",
                recovered_text="hidden",
            ),
        ],
        total_redactions=2,
        proper=1,
        bad_overlay=0,
        recoverable=1,
        recovered_text_fragments=["hidden"],
    )
    assert result.total_redactions == 2
    assert result.recoverable == 1
    assert len(result.recovered_text_fragments) == 1
