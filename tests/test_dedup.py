"""Tests for deduplication."""

from epstein_pipeline.models import Document
from epstein_pipeline.processors.dedup import Deduplicator


def test_exact_title_match():
    docs = [
        Document(id="doc-1", title="EFTA Filing About Financial Records"),
        Document(id="doc-2", title="EFTA Filing About Financial Records"),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) >= 1
    assert pairs[0].doc_id_1 == "doc-1"
    assert pairs[0].doc_id_2 == "doc-2"


def test_fuzzy_title_match():
    docs = [
        Document(id="doc-1", title="Jeffrey Epstein Financial Records 2005"),
        Document(id="doc-2", title="Jeffrey Epstein Financial Records, 2005"),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) >= 1


def test_no_false_positive():
    docs = [
        Document(id="doc-1", title="Travel Records from 2003"),
        Document(id="doc-2", title="Court Filing Regarding Estate Distribution"),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert len(pairs) == 0


def test_bates_overlap():
    docs = [
        Document(id="doc-1", title="Doc A", batesRange="EFTA00039025-EFTA00039030"),
        Document(id="doc-2", title="Doc B", batesRange="EFTA00039028-EFTA00039035"),
    ]
    dedup = Deduplicator(threshold=0.90)
    pairs = dedup.find_duplicates(docs)
    assert any(p.reason == "bates_overlap" for p in pairs)
