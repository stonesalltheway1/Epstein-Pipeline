"""Tests for confidence scoring."""

from epstein_pipeline.processors.confidence import ConfidenceScorer


def test_exact_match_confidence(person_registry):
    scorer = ConfidenceScorer(person_registry)
    match = scorer.score_entity_match("Jeffrey Epstein")
    assert match is not None
    assert match.confidence == 1.0
    assert match.match_type == "exact"
    assert match.person_id == "p-0001"


def test_alias_match_confidence(person_registry):
    scorer = ConfidenceScorer(person_registry)
    match = scorer.score_entity_match("Jeff Epstein")
    assert match is not None
    assert match.confidence == 0.95
    assert match.match_type == "alias"


def test_fuzzy_match_confidence(person_registry):
    scorer = ConfidenceScorer(person_registry)
    match = scorer.score_entity_match("Jeffery Epstein")  # Misspelling
    assert match is not None
    assert match.confidence >= 0.75
    assert match.match_type == "fuzzy"


def test_no_match_confidence(person_registry):
    scorer = ConfidenceScorer(person_registry)
    match = scorer.score_entity_match("John Smith")
    assert match is None


def test_short_name_rejected(person_registry):
    scorer = ConfidenceScorer(person_registry)
    match = scorer.score_entity_match("JE")
    assert match is None


def test_document_link_confidence(person_registry):
    scorer = ConfidenceScorer(person_registry)
    score = scorer.score_document_link(
        "p-0001",
        {
            "ner_match": True,
            "direct_scan": True,
            "title_mention": False,
        },
    )
    assert 0.0 < score < 1.0

    # All signals present
    max_score = scorer.score_document_link(
        "p-0001",
        {
            "ner_match": True,
            "direct_scan": True,
            "title_mention": True,
            "bates_match": True,
            "metadata_match": True,
        },
    )
    assert max_score == 1.0
