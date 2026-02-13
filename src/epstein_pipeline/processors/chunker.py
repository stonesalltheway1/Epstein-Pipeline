"""Document chunking for embedding generation.

Supports semantic-aware sliding window chunking that respects
section/clause boundaries in legal documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Boundary patterns for legal documents
_SECTION_BREAK = re.compile(
    r"(?:\n\s*\n)"  # double newline (paragraph break)
    r"|(?:\n\s*(?:\d+\.|[A-Z]{2,}\.)\s)"  # numbered/lettered sections
    r"|(?:\n\s*[-=]{3,}\s*\n)",  # horizontal rules
    re.MULTILINE,
)

_SENTENCE_END = re.compile(r"[.!?]\s+")

# OCR noise patterns
_REPEATED_CHARS = re.compile(r"(.)\1{10,}")  # 10+ identical chars
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXCESSIVE_WHITESPACE = re.compile(r"[ \t]{5,}")
_DECORATIVE_LINES = re.compile(r"[|_=\-]{10,}")


@dataclass
class DocumentChunk:
    """A single chunk of a document ready for embedding."""

    document_id: str
    chunk_index: int
    chunk_text: str
    char_offset: int = 0
    token_count_est: int = 0


class Chunker:
    """Split document text into overlapping chunks for embedding.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in characters (default 3200 ~ 800 tokens).
    overlap : int
        Overlap between chunks in characters (default 800 ~ 200 tokens).
    min_chunk_size : int
        Minimum chunk size to emit (default 200 chars ~ 50 tokens).
    respect_boundaries : bool
        Try to break at section/paragraph boundaries (default True).
    """

    def __init__(
        self,
        chunk_size: int = 3200,
        overlap: int = 800,
        min_chunk_size: int = 200,
        respect_boundaries: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_boundaries = respect_boundaries

    def chunk_document(
        self,
        document_id: str,
        text: str,
        *,
        prepend_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Parameters
        ----------
        document_id : str
            The document identifier.
        text : str
            The full text to chunk.
        prepend_title : str | None
            If provided, prepend this title to each chunk for context.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of chunks with metadata.
        """
        text = self._clean_ocr_noise(text)

        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # For short texts that fit in one chunk, return as-is
        if len(text) <= self.chunk_size:
            chunk_text = text.strip()
            if prepend_title:
                chunk_text = f"{prepend_title}\n\n{chunk_text}"
            return [
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=0,
                    chunk_text=chunk_text,
                    char_offset=0,
                    token_count_est=len(chunk_text) // 4,
                )
            ]

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Find a good break point
            if end < len(text) and self.respect_boundaries:
                end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()
            if len(chunk_text) < self.min_chunk_size:
                break

            if prepend_title:
                chunk_text = f"{prepend_title}\n\n{chunk_text}"

            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    chunk_text=chunk_text,
                    char_offset=start,
                    token_count_est=len(chunk_text) // 4,
                )
            )
            chunk_idx += 1

            # Advance with overlap
            new_start = end - self.overlap
            if new_start <= start:
                # No forward progress — would loop forever
                break
            start = new_start
            if start >= len(text) - self.min_chunk_size:
                break

        return chunks

    def _find_break_point(self, text: str, start: int, target: int) -> int:
        """Find the best break point near the target position.

        Searches backwards from target for, in order of preference:
        1. Section breaks (double newlines, section headers)
        2. Sentence endings (. ! ?)
        3. Falls back to the original target
        """
        # Search window: look back up to 20% of chunk_size
        window_start = max(start, target - self.chunk_size // 5)
        window = text[window_start:target]

        # Try section breaks first
        for match in reversed(list(_SECTION_BREAK.finditer(window))):
            pos = window_start + match.end()
            if pos > start + self.min_chunk_size:
                return pos

        # Try sentence endings
        for match in reversed(list(_SENTENCE_END.finditer(window))):
            pos = window_start + match.end()
            if pos > start + self.min_chunk_size:
                return pos

        return target

    def _clean_ocr_noise(self, text: str) -> str:
        """Remove common OCR artifacts before chunking."""
        text = _CONTROL_CHARS.sub("", text)
        text = _REPEATED_CHARS.sub(r"\1\1\1", text)
        text = _DECORATIVE_LINES.sub("", text)
        text = _EXCESSIVE_WHITESPACE.sub("  ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
