"""Tests for the document chunker."""

from epstein_pipeline.processors.chunker import Chunker, DocumentChunk


def test_short_text_single_chunk():
    """Short text should produce exactly one chunk."""
    chunker = Chunker(chunk_size=3200, overlap=800, min_chunk_size=10)
    text = "This is a short document about Jeffrey Epstein financial records."
    chunks = chunker.chunk_document("doc-1", text)
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert text in chunks[0].chunk_text


def test_empty_text_no_chunks():
    """Empty or whitespace-only text should produce no chunks."""
    chunker = Chunker()
    assert chunker.chunk_document("doc-1", "") == []
    assert chunker.chunk_document("doc-1", "   \n\n  ") == []
    assert chunker.chunk_document("doc-1", "ab") == []  # below min_chunk_size


def test_long_text_multiple_chunks():
    """Text longer than chunk_size should produce multiple overlapping chunks."""
    chunker = Chunker(chunk_size=100, overlap=20, min_chunk_size=30)
    text = " ".join(f"document word number {i}" for i in range(50))
    chunks = chunker.chunk_document("doc-2", text)
    assert len(chunks) >= 2
    # Check ordering
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.document_id == "doc-2"


def test_overlap_between_chunks():
    """Adjacent chunks should have overlapping content."""
    chunker = Chunker(
        chunk_size=100,
        overlap=30,
        min_chunk_size=20,
        respect_boundaries=False,
    )
    # Use distinct segments so we can verify overlap
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunker.chunk_document("doc-3", text)
    assert len(chunks) >= 2

    # The end of chunk 0 should overlap with start of chunk 1
    end_of_first = chunks[0].chunk_text[-30:]
    assert end_of_first in chunks[1].chunk_text


def test_prepend_title():
    """prepend_title should add title at the start of each chunk."""
    chunker = Chunker(chunk_size=3200, min_chunk_size=10)
    chunks = chunker.chunk_document(
        "doc-4",
        "Document body text here with enough content to pass minimum.",
        prepend_title="EFTA Document 12345",
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_text.startswith("EFTA Document 12345\n\n")


def test_ocr_noise_cleaning():
    """OCR noise should be cleaned before chunking."""
    chunker = Chunker(min_chunk_size=10)
    noisy = (
        "Hello world this is a real document with content"
        + "|" * 50
        + "\n\n\n\n\n"
        + "More text here with enough words to pass the minimum size check."
    )
    chunks = chunker.chunk_document("doc-5", noisy)
    assert len(chunks) == 1
    # Should not contain the pipe run
    assert "||||" not in chunks[0].chunk_text
    # Should collapse excessive newlines
    assert "\n\n\n" not in chunks[0].chunk_text


def test_section_boundary_respect():
    """Chunker should prefer breaking at section boundaries."""
    chunker = Chunker(
        chunk_size=200, overlap=40, min_chunk_size=50
    )
    # Build text with a clear section break
    section1 = "First section content. " * 8  # ~180 chars
    section2 = "\n\nSection 2\n\n" + "Second section content. " * 8
    text = section1 + section2
    chunks = chunker.chunk_document("doc-6", text)
    assert len(chunks) >= 2


def test_token_count_estimate():
    """Token count should be approximately chars / 4."""
    chunker = Chunker(min_chunk_size=10)
    text = " ".join(f"word{i}" for i in range(80))  # ~480 chars
    chunks = chunker.chunk_document("doc-7", text)
    assert len(chunks) == 1
    # ~480 chars → ~120 tokens estimate
    assert 80 < chunks[0].token_count_est < 160


def test_document_chunk_dataclass():
    """DocumentChunk should hold all expected fields."""
    chunk = DocumentChunk(
        document_id="doc-1",
        chunk_index=0,
        chunk_text="Hello",
        char_offset=0,
        token_count_est=1,
    )
    assert chunk.document_id == "doc-1"
    assert chunk.chunk_index == 0
    assert chunk.chunk_text == "Hello"
