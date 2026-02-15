"""Tests for the embedding processor (mocks sentence-transformers)."""

import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from epstein_pipeline.models.document import Document
from epstein_pipeline.processors.embeddings import (
    EmbeddingProcessor,
    EmbeddingResult,
    float_list_to_f32_blob,
)


@pytest.fixture
def mock_settings(settings):
    """Settings with embedding config."""
    settings.embedding_model = "test-model"
    settings.embedding_dimensions = 4
    settings.embedding_chunk_size = 500
    settings.embedding_chunk_overlap = 100
    return settings


@pytest.fixture
def sample_doc():
    return Document(
        id="doc-test-001",
        title="Jeffrey Epstein Financial Records",
        source="efta",
        category="financial",
        ocrText=" ".join(
            f"This is sentence number {i} about financial transactions "
            "involving various parties and offshore accounts."
            for i in range(20)
        ),
    )


def _mock_encode(texts, **kwargs):
    """Return fake embeddings of the right shape."""
    return np.random.randn(len(texts), 4).astype(np.float32)


def test_float_list_to_f32_blob():
    """F32 blob should pack floats correctly."""
    values = [1.0, 2.0, 3.0]
    blob = float_list_to_f32_blob(values)
    assert len(blob) == 12  # 3 floats × 4 bytes
    unpacked = struct.unpack("3f", blob)
    assert unpacked == (1.0, 2.0, 3.0)


def test_float_list_to_f32_blob_empty():
    """Empty list should produce empty blob."""
    assert float_list_to_f32_blob([]) == b""


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor._load_model")
def test_embed_document(mock_load, mock_settings, sample_doc):
    """embed_document should chunk and produce embeddings."""
    mock_model = MagicMock()
    mock_model.encode = _mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 4
    mock_load.return_value = mock_model

    processor = EmbeddingProcessor(
        settings=mock_settings,
        model_name="test-model",
        dimensions=4,
    )

    result = processor.embed_document(sample_doc)
    assert isinstance(result, EmbeddingResult)
    assert result.document_id == "doc-test-001"
    assert len(result.chunks) > 0
    assert len(result.embeddings) == len(result.chunks)
    assert len(result.embeddings[0]) == 4


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor._load_model")
def test_embed_document_no_text(mock_load, mock_settings):
    """Document without text should produce empty result."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    processor = EmbeddingProcessor(settings=mock_settings, dimensions=4)
    doc = Document(
        id="empty-doc",
        title="Empty",
        source="other",
        category="other",
    )

    result = processor.embed_document(doc)
    assert result.document_id == "empty-doc"
    assert len(result.chunks) == 0
    assert len(result.embeddings) == 0


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor._load_model")
def test_embed_texts(mock_load, mock_settings):
    """embed_texts should return vectors for each input."""
    mock_model = MagicMock()
    mock_model.encode = _mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 4
    mock_load.return_value = mock_model

    processor = EmbeddingProcessor(settings=mock_settings, dimensions=4)

    texts = ["Hello world", "Another document"]
    embeddings = processor.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 4
    assert all(isinstance(v, float) for v in embeddings[0])


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor._load_model")
def test_write_ndjson(mock_load, mock_settings, tmp_path):
    """NDJSON output should have one JSON object per line."""
    from epstein_pipeline.processors.chunker import DocumentChunk

    results = [
        EmbeddingResult(
            document_id="doc-1",
            chunks=[
                DocumentChunk("doc-1", 0, "Chunk zero", 0, 2),
                DocumentChunk("doc-1", 1, "Chunk one", 100, 2),
            ],
            embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            model_name="test",
        )
    ]

    processor = EmbeddingProcessor(settings=mock_settings, dimensions=4)
    out_path = tmp_path / "test.ndjson"
    processor.write_ndjson(results, out_path)

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2

    obj = json.loads(lines[0])
    assert obj["document_id"] == "doc-1"
    assert obj["chunk_index"] == 0
    assert obj["chunk_text"] == "Chunk zero"
    assert len(obj["embedding"]) == 4


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor._load_model")
def test_write_sqlite(mock_load, mock_settings, tmp_path):
    """SQLite output should create document_chunks table."""
    import sqlite3

    from epstein_pipeline.processors.chunker import DocumentChunk

    results = [
        EmbeddingResult(
            document_id="doc-1",
            chunks=[DocumentChunk("doc-1", 0, "Test chunk", 0, 2)],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
            model_name="test",
        )
    ]

    processor = EmbeddingProcessor(settings=mock_settings, dimensions=4)
    db_path = tmp_path / "test.db"
    processor.write_sqlite(results, db_path)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT document_id, chunk_index, chunk_text FROM document_chunks"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "doc-1"
    assert rows[0][1] == 0
    assert rows[0][2] == "Test chunk"

    # Check embedding blob
    blob = conn.execute("SELECT embedding FROM document_chunks").fetchone()[0]
    assert len(blob) == 16  # 4 floats × 4 bytes
    conn.close()
