"""Deduplication of documents using multiple strategies.

Three-pass dedup pipeline:
1. **Exact** — content hash + title fuzzy (rapidfuzz) + Bates range overlap
2. **MinHash/LSH** — near-duplicate detection via shingling + MinHash (O(n))
3. **Semantic** — embedding cosine similarity for OCR-variant detection

Replaces the old O(n^2) pairwise approach with scalable algorithms.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel
from rapidfuzz import fuzz

from epstein_pipeline.config import DedupMode, Settings
from epstein_pipeline.models.document import Document

logger = logging.getLogger(__name__)


class DuplicatePair(BaseModel):
    """A pair of documents identified as probable duplicates."""

    doc_id_1: str
    doc_id_2: str
    score: float  # 0.0 - 1.0, where 1.0 is identical
    reason: str  # Human-readable explanation
    method: str = "exact"  # 'exact', 'minhash', 'semantic'


@dataclass
class DuplicateCluster:
    """A cluster of duplicate documents with a representative."""

    cluster_id: str
    document_ids: list[str] = field(default_factory=list)
    representative_id: str | None = None
    method: str = "exact"
    avg_similarity: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BATES_PATTERN = re.compile(r"([A-Z]+)(\d+)")


def _parse_bates_range(bates: str) -> tuple[str, int, int] | None:
    """Parse a Bates range like 'EFTA00039025-EFTA00039030'."""
    parts = bates.split("-")
    if len(parts) < 2:
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


def _text_shingles(text: str, k: int = 5) -> set[str]:
    """Generate character k-shingles from normalised text."""
    normalised = " ".join(text.lower().split())
    if len(normalised) < k:
        return {normalised}
    return {normalised[i : i + k] for i in range(len(normalised) - k + 1)}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Deduplicator:
    """Find duplicate documents using multiple strategies.

    Strategies (controlled via ``Settings.dedup_mode``):

    - **exact** — Content hash, Bates range overlap, title fuzzy match.
      The original approach, now optimized with hash-group detection.
    - **minhash** — MinHash/LSH for near-duplicate detection.
      O(n) candidate generation via ``datasketch``.
    - **semantic** — Embedding cosine similarity.
      Catches same document with different scan quality / OCR artifacts.
    - **all** — Run all three passes sequentially, merging results.
    """

    def __init__(self, settings: Settings | None = None, threshold: float = 0.90) -> None:
        self.settings = settings
        self.threshold = threshold if settings is None else settings.dedup_threshold
        self.mode = DedupMode.ALL if settings is None else settings.dedup_mode

        # MinHash settings
        self.jaccard_threshold = 0.80 if settings is None else settings.dedup_jaccard_threshold
        self.shingle_size = 5 if settings is None else settings.dedup_shingle_size
        self.num_perm = 128 if settings is None else settings.dedup_num_perm

        # Semantic settings
        self.semantic_threshold = 0.95 if settings is None else settings.dedup_semantic_threshold

    def find_duplicates(self, documents: list[Document]) -> list[DuplicatePair]:
        """Run configured dedup passes and return sorted duplicate pairs."""
        pairs: list[DuplicatePair] = []
        seen_pairs: set[tuple[str, str]] = set()

        if self.mode in (DedupMode.EXACT, DedupMode.ALL):
            exact_pairs = self._exact_dedup(documents)
            for p in exact_pairs:
                key = (min(p.doc_id_1, p.doc_id_2), max(p.doc_id_1, p.doc_id_2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append(p)
            logger.info("Exact dedup found %d pairs", len(exact_pairs))

        if self.mode in (DedupMode.MINHASH, DedupMode.ALL):
            minhash_pairs = self._minhash_dedup(documents)
            for p in minhash_pairs:
                key = (min(p.doc_id_1, p.doc_id_2), max(p.doc_id_1, p.doc_id_2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append(p)
            logger.info("MinHash dedup found %d new pairs", len(minhash_pairs))

        if self.mode in (DedupMode.SEMANTIC, DedupMode.ALL):
            semantic_pairs = self._semantic_dedup(documents)
            for p in semantic_pairs:
                key = (min(p.doc_id_1, p.doc_id_2), max(p.doc_id_1, p.doc_id_2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append(p)
            logger.info("Semantic dedup found %d new pairs", len(semantic_pairs))

        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs

    def find_clusters(self, documents: list[Document]) -> list[DuplicateCluster]:
        """Find duplicate pairs and group them into clusters.

        Uses Union-Find to merge overlapping pairs into connected components.
        Selects the document with the most OCR text as representative.
        """
        pairs = self.find_duplicates(documents)
        if not pairs:
            return []

        # Union-Find
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for p in pairs:
            parent.setdefault(p.doc_id_1, p.doc_id_1)
            parent.setdefault(p.doc_id_2, p.doc_id_2)
            union(p.doc_id_1, p.doc_id_2)

        # Group by root
        groups: dict[str, list[str]] = {}
        for doc_id in parent:
            root = find(doc_id)
            groups.setdefault(root, []).append(doc_id)

        # Build clusters
        doc_map = {d.id: d for d in documents}
        clusters: list[DuplicateCluster] = []

        for group_ids in groups.values():
            if len(group_ids) < 2:
                continue

            # Pick representative: longest OCR text
            best_id = max(
                group_ids,
                key=lambda did: len(doc_map[did].ocrText or "") if did in doc_map else 0,
            )

            # Compute average similarity from pairs within this group
            group_set = set(group_ids)
            sims = [p.score for p in pairs if p.doc_id_1 in group_set and p.doc_id_2 in group_set]

            clusters.append(
                DuplicateCluster(
                    cluster_id=str(uuid.uuid4())[:8],
                    document_ids=sorted(group_ids),
                    representative_id=best_id,
                    avg_similarity=sum(sims) / len(sims) if sims else 1.0,
                )
            )

        logger.info("Found %d duplicate clusters from %d pairs", len(clusters), len(pairs))
        return clusters

    # ------------------------------------------------------------------
    # Pass 1: Exact dedup (content hash + Bates + title fuzzy)
    # ------------------------------------------------------------------

    def _exact_dedup(self, documents: list[Document]) -> list[DuplicatePair]:
        """Exact dedup via content hash, Bates overlap, and title similarity."""
        pairs: list[DuplicatePair] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Content hash grouping (O(n))
        hashes: dict[str, str] = {}
        for doc in documents:
            if doc.ocrText and doc.ocrText.strip():
                hashes[doc.id] = _content_hash(doc.ocrText)

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
                                method="exact",
                            )
                        )

        # Bates range overlap (only check docs with batesRange)
        bates_docs = [(i, d) for i, d in enumerate(documents) if d.batesRange]
        for idx_i in range(len(bates_docs)):
            for idx_j in range(idx_i + 1, len(bates_docs)):
                _, a = bates_docs[idx_i]
                _, b = bates_docs[idx_j]
                pair_key = (min(a.id, b.id), max(a.id, b.id))
                if pair_key in seen_pairs:
                    continue
                if a.batesRange and b.batesRange and _bates_overlap(a.batesRange, b.batesRange):
                    seen_pairs.add(pair_key)
                    pairs.append(
                        DuplicatePair(
                            doc_id_1=pair_key[0],
                            doc_id_2=pair_key[1],
                            score=0.95,
                            reason=f"Bates range overlap: {a.batesRange} / {b.batesRange}",
                            method="exact",
                        )
                    )

        # Title similarity — group by first word to reduce O(n^2)
        title_groups: dict[str, list[int]] = {}
        for i, doc in enumerate(documents):
            if doc.title:
                first_word = doc.title.lower().split()[0] if doc.title.split() else ""
                title_groups.setdefault(first_word, []).append(i)

        for group_indices in title_groups.values():
            if len(group_indices) < 2:
                continue
            for idx_i in range(len(group_indices)):
                for idx_j in range(idx_i + 1, len(group_indices)):
                    a = documents[group_indices[idx_i]]
                    b = documents[group_indices[idx_j]]
                    pair_key = (min(a.id, b.id), max(a.id, b.id))
                    if pair_key in seen_pairs:
                        continue
                    ratio = fuzz.ratio(a.title.lower(), b.title.lower()) / 100.0
                    if ratio >= self.threshold:
                        seen_pairs.add(pair_key)
                        pairs.append(
                            DuplicatePair(
                                doc_id_1=pair_key[0],
                                doc_id_2=pair_key[1],
                                score=round(ratio, 4),
                                reason=f"title similarity: {ratio:.2%}",
                                method="exact",
                            )
                        )

        return pairs

    # ------------------------------------------------------------------
    # Pass 2: MinHash/LSH near-duplicate detection
    # ------------------------------------------------------------------

    def _minhash_dedup(self, documents: list[Document]) -> list[DuplicatePair]:
        """MinHash/LSH dedup using datasketch for O(n) candidate generation."""
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning(
                "datasketch not installed — skipping MinHash dedup. "
                "Install with: pip install datasketch"
            )
            return []

        pairs: list[DuplicatePair] = []

        # Only process docs with substantial text
        text_docs = [
            (doc.id, doc.ocrText or doc.summary or "")
            for doc in documents
            if (doc.ocrText and len(doc.ocrText) > 100) or (doc.summary and len(doc.summary) > 100)
        ]

        if len(text_docs) < 2:
            return pairs

        logger.info("Building MinHash signatures for %d documents...", len(text_docs))

        # Build MinHash signatures
        signatures: dict[str, MinHash] = {}
        for doc_id, text in text_docs:
            shingles = _text_shingles(text, self.shingle_size)
            mh = MinHash(num_perm=self.num_perm)
            for s in shingles:
                mh.update(s.encode("utf-8"))
            signatures[doc_id] = mh

        # Build LSH index
        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=self.num_perm)
        for doc_id, mh in signatures.items():
            try:
                lsh.insert(doc_id, mh)
            except ValueError:
                pass  # Duplicate key — skip

        # Query for candidates
        seen: set[tuple[str, str]] = set()
        for doc_id, mh in signatures.items():
            candidates = lsh.query(mh)
            for candidate_id in candidates:
                if candidate_id == doc_id:
                    continue
                pair_key = (min(doc_id, candidate_id), max(doc_id, candidate_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                jaccard = signatures[doc_id].jaccard(signatures[candidate_id])
                if jaccard >= self.jaccard_threshold:
                    pairs.append(
                        DuplicatePair(
                            doc_id_1=pair_key[0],
                            doc_id_2=pair_key[1],
                            score=round(jaccard, 4),
                            reason=f"MinHash Jaccard similarity: {jaccard:.2%}",
                            method="minhash",
                        )
                    )

        return pairs

    # ------------------------------------------------------------------
    # Pass 3: Semantic dedup (embedding cosine similarity)
    # ------------------------------------------------------------------

    def _semantic_dedup(self, documents: list[Document]) -> list[DuplicatePair]:
        """Semantic dedup using embedding cosine similarity.

        Uses a lightweight model (all-MiniLM-L6-v2) for fast comparison.
        Catches same document with different scan quality / OCR artifacts.
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not installed — skipping semantic dedup.")
            return []

        pairs: list[DuplicatePair] = []

        # Get document text (truncated for speed)
        text_docs = []
        for doc in documents:
            text = doc.ocrText or doc.summary or doc.title
            if text and len(text) > 50:
                text_docs.append((doc.id, text[:2000]))

        if len(text_docs) < 2:
            return pairs

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — skipping semantic dedup. "
                "Install with: pip install sentence-transformers"
            )
            return []

        logger.info("Computing semantic embeddings for %d documents...", len(text_docs))

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [t for _, t in text_docs]
        doc_ids = [d for d, _ in text_docs]

        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )

        # Cosine similarity matrix (embeddings are already normalized)
        sim_matrix = np.dot(embeddings, embeddings.T)

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                sim = float(sim_matrix[i, j])
                if sim >= self.semantic_threshold:
                    pairs.append(
                        DuplicatePair(
                            doc_id_1=min(doc_ids[i], doc_ids[j]),
                            doc_id_2=max(doc_ids[i], doc_ids[j]),
                            score=round(sim, 4),
                            reason=f"semantic similarity: {sim:.2%}",
                            method="semantic",
                        )
                    )

        return pairs
