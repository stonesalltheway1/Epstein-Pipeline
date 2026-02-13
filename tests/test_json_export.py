"""Tests for JSON export functionality."""

import json

from epstein_pipeline.models.document import Document


def test_document_json_export(sample_documents):
    """Test that documents serialize to valid JSON."""
    for doc in sample_documents:
        data = doc.model_dump(exclude_none=True)
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        assert loaded["id"] == doc.id
        assert loaded["source"] == doc.source


def test_document_json_roundtrip(sample_documents):
    """Test JSON serialization/deserialization roundtrip."""
    for doc in sample_documents:
        json_str = doc.model_dump_json()
        loaded = Document.model_validate_json(json_str)
        assert loaded.id == doc.id
        assert loaded.title == doc.title
        assert loaded.source == doc.source
        assert loaded.personIds == doc.personIds
