"""Tests for deduplication — exact, MinHash/LSH, and semantic passes."""

from epstein_pipeline.models import Document
from epstein_pipeline.processors.dedup import (
    Deduplicator,
    DuplicateCluster,
    DuplicatePair,
)


def test_exact_title_match():
    docs = [
        Document(
            id="doc-1",
            title="EFTA Filing About Financial Records",
            source="efta",
            category="legal",
        ),
        Document(
            id="doc-2",
            title="EFTA Filing About Financial Records",
            source="efta",
            category="legal",
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) >= 1
    assert pairs[0].doc_id_1 == "doc-1"
    assert pairs[0].doc_id_2 == "doc-2"


def test_fuzzy_title_match():
    docs = [
        Document(
            id="doc-1",
            title="Jeffrey Epstein Financial Records 2005",
            source="financial",
            category="financial",
        ),
        Document(
            id="doc-2",
            title="Jeffrey Epstein Financial Records, 2005",
            source="financial",
            category="financial",
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) >= 1


def test_no_false_positive():
    docs = [
        Document(
            id="doc-1",
            title="Travel Records from 2003",
            source="travel",
            category="travel",
        ),
        Document(
            id="doc-2",
            title="Court Filing Regarding Estate Distribution",
            source="court-filing",
            category="legal",
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) == 0


def test_bates_overlap():
    docs = [
        Document(
            id="doc-1",
            title="Doc A",
            source="efta",
            category="other",
            batesRange="EFTA00039025-EFTA00039030",
        ),
        Document(
            id="doc-2",
            title="Doc B",
            source="efta",
            category="other",
            batesRange="EFTA00039028-EFTA00039035",
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert any("Bates range overlap" in p.reason for p in pairs)


def test_duplicate_pair_model():
    """DuplicatePair should include method field."""
    pair = DuplicatePair(
        doc_id_1="d1",
        doc_id_2="d2",
        score=0.95,
        reason="Identical titles",
        method="exact",
    )
    assert pair.method == "exact"
    assert pair.score == 0.95


def test_duplicate_cluster_dataclass():
    """DuplicateCluster should hold cluster info."""
    cluster = DuplicateCluster(
        cluster_id="c-001",
        document_ids=["d1", "d2", "d3"],
        representative_id="d1",
        method="minhash",
        avg_similarity=0.92,
    )
    assert len(cluster.document_ids) == 3
    assert cluster.representative_id == "d1"


def test_content_hash_dedup():
    """Documents with identical content hash should be detected."""
    text = "Identical OCR content for both documents."
    docs = [
        Document(
            id="doc-1",
            title="First Title",
            source="efta",
            category="other",
            ocrText=text,
        ),
        Document(
            id="doc-2",
            title="Second Title",
            source="efta",
            category="other",
            ocrText=text,
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    hash_pairs = [p for p in pairs if "content hash" in p.reason.lower()]
    assert len(hash_pairs) >= 1


def test_find_clusters():
    """find_clusters should group duplicate pairs into clusters."""
    docs = [
        Document(
            id="doc-1",
            title="Same Filing",
            source="efta",
            category="other",
            ocrText="Test content " * 20,
        ),
        Document(
            id="doc-2",
            title="Same Filing",
            source="efta",
            category="other",
            ocrText="Test content " * 20,
        ),
        Document(
            id="doc-3",
            title="Different Document",
            source="fbi",
            category="investigation",
            ocrText="Completely unrelated content " * 20,
        ),
    ]
    dedup = Deduplicator(threshold=0.90)
    clusters = dedup.find_clusters(docs)
    # doc-1 and doc-2 should be in same cluster, doc-3 separate
    assert len(clusters) >= 1
    cluster_ids = {c.cluster_id for c in clusters if "doc-1" in c.document_ids}
    for c in clusters:
        if c.cluster_id in cluster_ids:
            assert "doc-2" in c.document_ids
