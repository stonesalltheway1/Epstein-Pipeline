"""Tests for incremental processing state tracker."""

from pathlib import Path

from epstein_pipeline.state import ProcessingState


def test_state_basic(tmp_path: Path):
    db_path = tmp_path / "state.db"
    state = ProcessingState(db_path)

    assert not state.is_processed("abc123", "ocr")

    state.mark_processed("abc123", "ocr", "/output/abc123.json")
    assert state.is_processed("abc123", "ocr")
    assert not state.is_processed("abc123", "entities")

    result_path = state.get_result_path("abc123", "ocr")
    assert result_path == "/output/abc123.json"

    state.close()


def test_state_get_unprocessed(tmp_path: Path):
    db_path = tmp_path / "state.db"
    state = ProcessingState(db_path)

    state.mark_processed("hash1", "ocr")
    state.mark_processed("hash2", "ocr")
    state.mark_processed("hash3", "entities")

    unprocessed = state.get_unprocessed(["hash1", "hash2", "hash3", "hash4"], "ocr")
    assert set(unprocessed) == {"hash3", "hash4"}

    state.close()


def test_state_get_stats(tmp_path: Path):
    db_path = tmp_path / "state.db"
    state = ProcessingState(db_path)

    state.mark_processed("h1", "ocr")
    state.mark_processed("h2", "ocr")
    state.mark_processed("h3", "entities")

    stats = state.get_stats()
    assert stats["ocr"] == 2
    assert stats["entities"] == 1

    state.close()


def test_state_clear_stage(tmp_path: Path):
    db_path = tmp_path / "state.db"
    state = ProcessingState(db_path)

    state.mark_processed("h1", "ocr")
    state.mark_processed("h2", "ocr")
    state.mark_processed("h3", "entities")

    deleted = state.clear_stage("ocr")
    assert deleted == 2
    assert not state.is_processed("h1", "ocr")
    assert state.is_processed("h3", "entities")

    state.close()


def test_state_persistence(tmp_path: Path):
    db_path = tmp_path / "state.db"

    # Write with one instance
    state1 = ProcessingState(db_path)
    state1.mark_processed("persistent", "ocr")
    state1.close()

    # Read with another instance
    state2 = ProcessingState(db_path)
    assert state2.is_processed("persistent", "ocr")
    state2.close()
