"""Tests for the document classifier."""

from __future__ import annotations

from unittest.mock import MagicMock

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Document
from epstein_pipeline.processors.classifier import (
    CLASSIFICATION_LABELS,
    ClassificationResult,
    DocumentClassifier,
)

# -- helpers --------------------------------------------------------


def _make_doc(
    id: str = "doc-test",
    title: str = "Test Doc",
    summary: str | None = None,
    ocr: str | None = None,
) -> Document:
    return Document(
        id=id,
        title=title,
        source="other",
        category="other",
        summary=summary,
        ocrText=ocr,
    )


def _mock_pipeline_result(
    top_label: str = "legal document or court filing",
    top_score: float = 0.85,
) -> dict:
    """Build a fake transformers pipeline result dict."""
    labels = [top_label] + [lbl for lbl in CLASSIFICATION_LABELS if lbl != top_label]
    scores = [top_score] + [
        round((1 - top_score) / (len(labels) - 1), 4) for _ in range(len(labels) - 1)
    ]
    return {"labels": labels, "scores": scores}


# -- tests ----------------------------------------------------------


class TestDocumentClassifierInit:
    """Test classifier instantiation."""

    def test_defaults_from_settings(self, settings: Settings):
        clf = DocumentClassifier(settings)
        assert clf.model_name == settings.classifier_model
        assert clf.confidence_threshold == 0.6
        assert clf._pipeline is None

    def test_override_model_and_threshold(self, settings: Settings):
        clf = DocumentClassifier(
            settings,
            model_name="custom/model",
            confidence_threshold=0.9,
        )
        assert clf.model_name == "custom/model"
        assert clf.confidence_threshold == 0.9


class TestBuildText:
    """Test the text-building logic inside classify()."""

    def test_title_only(self, settings: Settings):
        doc = _make_doc(title="Deposition Transcript")
        clf = DocumentClassifier(settings)

        mock_pipe = MagicMock(return_value=_mock_pipeline_result())
        clf._pipeline = mock_pipe

        clf.classify(doc)
        called_text = mock_pipe.call_args[0][0]
        assert "Title: Deposition Transcript" in called_text
        assert "Summary:" not in called_text

    def test_title_and_summary(self, settings: Settings):
        doc = _make_doc(
            title="FBI Report",
            summary="Summary of investigation",
        )
        clf = DocumentClassifier(settings)
        mock_pipe = MagicMock(return_value=_mock_pipeline_result())
        clf._pipeline = mock_pipe

        clf.classify(doc)
        text = mock_pipe.call_args[0][0]
        assert "Title: FBI Report" in text
        assert "Summary: Summary of investigation" in text

    def test_title_summary_and_ocr(self, settings: Settings):
        doc = _make_doc(
            title="Financial Record",
            summary="Bank statement",
            ocr="Wire transfer $50,000",
        )
        clf = DocumentClassifier(settings)
        mock_pipe = MagicMock(return_value=_mock_pipeline_result())
        clf._pipeline = mock_pipe

        clf.classify(doc)
        text = mock_pipe.call_args[0][0]
        assert "Wire transfer $50,000" in text

    def test_ocr_truncated_to_1000(self, settings: Settings):
        long_ocr = "x" * 2000
        doc = _make_doc(ocr=long_ocr)
        clf = DocumentClassifier(settings)
        mock_pipe = MagicMock(return_value=_mock_pipeline_result())
        clf._pipeline = mock_pipe

        clf.classify(doc)
        text = mock_pipe.call_args[0][0]
        # OCR portion should be at most 1000 chars
        assert "x" * 1000 in text
        assert "x" * 1001 not in text


class TestClassifyEmptyText:
    """Empty/blank documents should return ('other', 0.0)."""

    def test_no_title_no_summary_no_ocr(self, settings: Settings):
        doc = _make_doc(title="", summary=None, ocr=None)
        clf = DocumentClassifier(settings)
        result = clf.classify(doc)
        assert result.predicted_category == "other"
        assert result.confidence == 0.0
        assert result.all_scores == {}


class TestClassify:
    """Test classify() with mocked pipeline."""

    def test_high_confidence_assigns_category(self, settings: Settings):
        doc = _make_doc(title="Deposition of Jane Doe")
        clf = DocumentClassifier(settings)
        mock_pipe = MagicMock(
            return_value=_mock_pipeline_result("legal document or court filing", 0.92)
        )
        clf._pipeline = mock_pipe

        result = clf.classify(doc)
        assert result.predicted_category == "legal"
        assert result.confidence == 0.92
        assert isinstance(result.all_scores, dict)

    def test_below_threshold_returns_other(self, settings: Settings):
        doc = _make_doc(title="Some ambiguous doc")
        clf = DocumentClassifier(settings, confidence_threshold=0.8)
        mock_pipe = MagicMock(
            return_value=_mock_pipeline_result("legal document or court filing", 0.5)
        )
        clf._pipeline = mock_pipe

        result = clf.classify(doc)
        assert result.predicted_category == "other"
        assert result.confidence == 0.5


class TestClassifyBatch:
    """Test classify_batch() with mocked pipeline."""

    def test_batch_returns_all_results(self, settings: Settings):
        docs = [_make_doc(id=f"d-{i}") for i in range(3)]
        clf = DocumentClassifier(settings)
        clf._pipeline = MagicMock(return_value=_mock_pipeline_result())

        results = clf.classify_batch(docs)
        assert len(results) == 3
        assert [r.document_id for r in results] == [
            "d-0",
            "d-1",
            "d-2",
        ]

    def test_batch_handles_error_gracefully(self, settings: Settings):
        docs = [_make_doc(id="ok"), _make_doc(id="fail")]
        clf = DocumentClassifier(settings)
        call_count = 0

        def _side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return _mock_pipeline_result()

        clf._pipeline = MagicMock(side_effect=_side_effect)
        results = clf.classify_batch(docs)
        assert len(results) == 2
        assert results[0].predicted_category == "legal"
        assert results[1].predicted_category == "other"
        assert results[1].confidence == 0.0


class TestClassificationResultShape:
    """Verify result fields have correct types."""

    def test_result_fields(self, settings: Settings):
        doc = _make_doc()
        clf = DocumentClassifier(settings)
        clf._pipeline = MagicMock(return_value=_mock_pipeline_result())
        result = clf.classify(doc)

        assert isinstance(result, ClassificationResult)
        assert isinstance(result.confidence, float)
        assert result.predicted_category in (
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
        )
