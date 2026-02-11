"""Deduplication of documents using title similarity, Bates range overlap, and content hashing."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel
from rapidfuzz import fuzz

from epstein_pipeline.models.document import Document


class DuplicatePair(BaseModel):
    """A pair of documents identified as probable duplicates."""

    doc_id_1: str
    doc_id_2: str
    score: float  # 0.0 - 1.0, where 1.0 is identical
    reason: str  # Human-readable explanation (e.g. "title similarity: 0.95")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BATES_PATTERN = re.compile(r"([A-Z]+)(\d+)")


def _parse_bates_range(bates: str) -> tuple[str, int, int] | None:
    """Parse a Bates range like 'EFTA00039025-EFTA00039030'.

    Returns (prefix, start_num, end_num) or None if unparseable.
    """
    parts = bates.split("-")
    if len(parts) < 2:
        # Single Bates number -- treat as a one-page range.
        m = _BATES_PATTERN.match(parts[0].strip())
        if m:
            prefix, num_str = m.group(1), m.group(2)
            num = int(num_str)
            return (prefix, num, num)
        return None

    m1 = _BATES_PATTERN.match(parts[0].strip())
    m2 = _BATES_PATTERN.match(parts[-1].strip())
    if not m1 or not m2:
        return None

    prefix1, num1 = m1.group(1), int(m1.group(2))
    prefix2, num2 = m2.group(1), int(m2.group(2))

    # Bates ranges should share a prefix.
    if prefix1 != prefix2:
        return None

    return (prefix1, min(num1, num2), max(num1, num2))


def _bates_overlap(a: str, b: str) -> bool:
    """Return True if two Bates ranges overlap."""
    ra = _parse_bates_range(a)
    rb = _parse_bates_range(b)
    if ra is None or rb is None:
        return False
    if ra[0] != rb[0]:
        return False
    return ra[1] <= rb[2] and rb[1] <= ra[2]


def _content_hash(text: str) -> str:
    """SHA-256 of normalised text (lowered, whitespace-collapsed)."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Deduplicator:
    """Find duplicate document pairs using multiple signals.

    Signals checked (in order):
    1. **Content hash** -- if both documents have ``ocrText`` and their
       normalised hashes match, they are exact duplicates (score 1.0).
    2. **Bates range overlap** -- if both documents have ``batesRange``
       values whose numeric ranges overlap, they are likely duplicates.
    3. **Title similarity** -- fuzzy string match on titles using
       ``rapidfuzz.fuzz.ratio``.  Only reported when the score meets the
       configured *threshold*.
    """

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold

    def find_duplicates(self, documents: list[Document]) -> list[DuplicatePair]:
        """Scan all document pairs and return those that look like duplicates.

        The returned list is sorted by descending score so the most confident
        matches appear first.
        """
        pairs: list[DuplicatePair] = []
        seen_pairs: set[tuple[str, str]] = set()
        n = len(documents)

        # Pre-compute content hashes for documents that have OCR text.
        hashes: dict[str, str] = {}
        for doc in documents:
            if doc.ocrText and doc.ocrText.strip():
                hashes[doc.id] = _content_hash(doc.ocrText)

        # Build a hash -> list[doc_id] map to find exact content dupes in O(n).
        hash_groups: dict[str, list[str]] = {}
        for doc_id, h in hashes.items():
            hash_groups.setdefault(h, []).append(doc_id)

        for group in hash_groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair_key = (min(group[i], group[j]), max(group[i], group[j]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pairs.append(
                            DuplicatePair(
                                doc_id_1=pair_key[0],
                                doc_id_2=pair_key[1],
                                score=1.0,
                                reason="exact content hash match",
                            )
                        )

        # Pairwise checks for Bates overlap and title similarity.
        for i in range(n):
            for j in range(i + 1, n):
                a, b = documents[i], documents[j]
                pair_key = (min(a.id, b.id), max(a.id, b.id))
                if pair_key in seen_pairs:
                    continue

                # Bates range overlap
                if a.batesRange and b.batesRange:
                    if _bates_overlap(a.batesRange, b.batesRange):
                        seen_pairs.add(pair_key)
                        pairs.append(
                            DuplicatePair(
                                doc_id_1=pair_key[0],
                                doc_id_2=pair_key[1],
                                score=0.95,
                                reason=f"Bates range overlap: {a.batesRange} / {b.batesRange}",
                            )
                        )
                        continue

                # Title similarity
                if a.title and b.title:
                    ratio = fuzz.ratio(a.title.lower(), b.title.lower()) / 100.0
                    if ratio >= self.threshold:
                        seen_pairs.add(pair_key)
                        pairs.append(
                            DuplicatePair(
                                doc_id_1=pair_key[0],
                                doc_id_2=pair_key[1],
                                score=round(ratio, 4),
                                reason=f"title similarity: {ratio:.2%}",
                            )
                        )

        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs
