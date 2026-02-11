"""Content hashing and text normalisation utilities."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip leading/trailing whitespace, and collapse runs of whitespace to a single space."""
    return _WHITESPACE_RUN.sub(" ", text.lower().strip())


def content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text* after normalisation."""
    normalised = normalize_text(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

    Reads the file in 64 KiB chunks to handle large files without
    loading everything into memory at once.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65_536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
