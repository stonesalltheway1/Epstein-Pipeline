"""Document chunking for embedding generation.

Two chunking modes:
- **fixed** — Sliding window with character-based sizes (original approach)
- **semantic** — Respects paragraph/section/sentence boundaries with token targets

Both modes include OCR noise cleanup and legal document structure awareness.
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

# Structural patterns for semantic chunking
_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"(?:SECTION|ARTICLE|CHAPTER|PART|EXHIBIT|SCHEDULE|APPENDIX)\s+[\dIVXLCDM]+"
    r"|(?:\d+\.)+\s+[A-Z]"  # 1.2.3 Section Title
    r"|[A-Z][A-Z\s]{5,}$"  # ALL CAPS HEADERS
    r")",
    re.MULTILINE,
)

_LIST_ITEM = re.compile(r"^\s*(?:[a-z]\)|[ivx]+\)|•|[-*])\s", re.MULTILINE)


@dataclass
class DocumentChunk:
    """A single chunk of a document ready for embedding."""

    document_id: str
    chunk_index: int
    chunk_text: str
    char_offset: int = 0
    token_count_est: int = 0
    section_title: str | None = None
    page_number: int | None = None
    is_table: bool = False
    is_header: bool = False


class Chunker:
    """Split document text into overlapping chunks for embedding.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in characters (default 3200 ~ 800 tokens).
        Used in 'fixed' mode.
    overlap : int
        Overlap between chunks in characters (default 800 ~ 200 tokens).
        Used in 'fixed' mode.
    min_chunk_size : int
        Minimum chunk size to emit (default 200 chars ~ 50 tokens).
    respect_boundaries : bool
        Try to break at section/paragraph boundaries (default True).
    mode : str
        Chunking mode: 'fixed' or 'semantic' (default 'semantic').
    target_tokens : int
        Target chunk size in tokens for 'semantic' mode (default 512).
    min_tokens : int
        Minimum chunk size in tokens for 'semantic' mode (default 100).
    max_tokens : int
        Maximum chunk size in tokens for 'semantic' mode (default 1024).
    """

    def __init__(
        self,
        chunk_size: int = 3200,
        overlap: int = 800,
        min_chunk_size: int = 200,
        respect_boundaries: bool = True,
        mode: str = "semantic",
        target_tokens: int = 512,
        min_tokens: int = 100,
        max_tokens: int = 1024,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_boundaries = respect_boundaries
        self.mode = mode
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk_document(
        self,
        document_id: str,
        text: str,
        *,
        prepend_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Split text into chunks using the configured mode."""
        text = self._clean_ocr_noise(text)

        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        if self.mode == "semantic":
            return self._chunk_semantic(document_id, text, prepend_title=prepend_title)
        else:
            return self._chunk_fixed(document_id, text, prepend_title=prepend_title)

    # ------------------------------------------------------------------
    # Semantic chunking
    # ------------------------------------------------------------------

    def _chunk_semantic(
        self,
        document_id: str,
        text: str,
        *,
        prepend_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunk text respecting semantic boundaries.

        Strategy:
        1. Split into paragraphs (double newlines)
        2. Merge small paragraphs together up to target_tokens
        3. Split large paragraphs at sentence boundaries
        4. Respect document structure (headers, lists, tables)
        """
        # Estimate chars per token (roughly 4 chars/token for English)
        chars_per_token = 4
        target_chars = self.target_tokens * chars_per_token
        min_chars = self.min_tokens * chars_per_token
        max_chars = self.max_tokens * chars_per_token

        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return []

        # Short text: single chunk
        total_len = sum(len(p) for p in paragraphs)
        if total_len <= target_chars:
            chunk_text = "\n\n".join(paragraphs).strip()
            if prepend_title:
                chunk_text = f"{prepend_title}\n\n{chunk_text}"
            return [
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=0,
                    chunk_text=chunk_text,
                    char_offset=0,
                    token_count_est=len(chunk_text) // chars_per_token,
                )
            ]

        # Merge paragraphs into chunks
        chunks: list[DocumentChunk] = []
        current_parts: list[str] = []
        current_len = 0
        chunk_idx = 0
        char_offset = 0

        for para in paragraphs:
            para_len = len(para)

            # If this paragraph alone exceeds max, split it at sentences
            if para_len > max_chars:
                # Flush current buffer first
                if current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()
                    if len(chunk_text) >= min_chars:
                        if prepend_title:
                            chunk_text = f"{prepend_title}\n\n{chunk_text}"
                        chunks.append(
                            DocumentChunk(
                                document_id=document_id,
                                chunk_index=chunk_idx,
                                chunk_text=chunk_text,
                                char_offset=char_offset,
                                token_count_est=len(chunk_text) // chars_per_token,
                            )
                        )
                        chunk_idx += 1
                    current_parts = []
                    current_len = 0

                # Split large paragraph at sentences
                sentences = self._split_sentences(para)
                sent_buffer: list[str] = []
                sent_len = 0

                for sent in sentences:
                    if sent_len + len(sent) > target_chars and sent_buffer:
                        chunk_text = " ".join(sent_buffer).strip()
                        if len(chunk_text) >= min_chars:
                            if prepend_title:
                                chunk_text = f"{prepend_title}\n\n{chunk_text}"
                            chunks.append(
                                DocumentChunk(
                                    document_id=document_id,
                                    chunk_index=chunk_idx,
                                    chunk_text=chunk_text,
                                    char_offset=char_offset,
                                    token_count_est=len(chunk_text) // chars_per_token,
                                )
                            )
                            chunk_idx += 1
                        sent_buffer = []
                        sent_len = 0

                    sent_buffer.append(sent)
                    sent_len += len(sent)

                # Remaining sentences
                if sent_buffer:
                    current_parts = [" ".join(sent_buffer)]
                    current_len = sent_len
                continue

            # Would adding this paragraph exceed target?
            if current_len + para_len > target_chars and current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                if len(chunk_text) >= min_chars:
                    if prepend_title:
                        chunk_text = f"{prepend_title}\n\n{chunk_text}"
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            chunk_text=chunk_text,
                            char_offset=char_offset,
                            token_count_est=len(chunk_text) // chars_per_token,
                        )
                    )
                    chunk_idx += 1
                    char_offset += current_len

                # Start new buffer — keep last paragraph for overlap context
                if current_parts and len(current_parts[-1]) < target_chars // 4:
                    current_parts = [current_parts[-1]]
                    current_len = len(current_parts[0])
                else:
                    current_parts = []
                    current_len = 0

            current_parts.append(para)
            current_len += para_len

        # Flush remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            if len(chunk_text) >= min_chars:
                if prepend_title:
                    chunk_text = f"{prepend_title}\n\n{chunk_text}"
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_text=chunk_text,
                        char_offset=char_offset,
                        token_count_est=len(chunk_text) // chars_per_token,
                    )
                )

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving structure."""
        # Split on double newlines
        raw_paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        return paragraphs

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Use regex to split at sentence boundaries
        parts = _SENTENCE_END.split(text)
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
        return sentences

    # ------------------------------------------------------------------
    # Fixed-size chunking (original approach)
    # ------------------------------------------------------------------

    def _chunk_fixed(
        self,
        document_id: str,
        text: str,
        *,
        prepend_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Split text into fixed-size overlapping chunks."""
        # For short texts that fit in one chunk
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

            new_start = end - self.overlap
            if new_start <= start:
                break
            start = new_start
            if start >= len(text) - self.min_chunk_size:
                break

        return chunks

    def _find_break_point(self, text: str, start: int, target: int) -> int:
        """Find the best break point near the target position."""
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
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
