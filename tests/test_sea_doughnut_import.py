"""Tests for Sea_Doughnut importer."""

import sqlite3
from pathlib import Path

import pytest

from epstein_pipeline.importers.sea_doughnut import SeaDoughnutImporter


@pytest.fixture
def mock_corpus_db(tmp_path: Path) -> Path:
    """Create a mock full_text_corpus.db with sample data."""
    db_path = tmp_path / "full_text_corpus.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            dataset TEXT,
            bates_number TEXT,
            page_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE transcripts (
            doc_id TEXT PRIMARY KEY,
            text TEXT,
            language TEXT DEFAULT 'en',
            duration REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            entity_type TEXT,
            text TEXT,
            confidence REAL DEFAULT 0
        )
    """)

    # Insert sample data
    for i in range(50):
        conn.execute(
            "INSERT INTO documents"
            " (doc_id, title, text, dataset, bates_number, page_count)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"sd-{i:04d}",
                f"Document {i}",
                f"Text content for document {i}",
                "ds1",
                f"EFTA{i:08d}",
                5,
            ),
        )

    conn.execute(
        "INSERT INTO transcripts (doc_id, text, language, duration) VALUES (?, ?, ?, ?)",
        ("tx-001", "Hello world transcript", "en", 120.0),
    )

    for i in range(10):
        conn.execute(
            "INSERT INTO entities (doc_id, entity_type, text, confidence) VALUES (?, ?, ?, ?)",
            (f"sd-{i:04d}", "PERSON", f"Person {i}", 0.9),
        )

    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def mock_redaction_db(tmp_path: Path) -> Path:
    """Create a mock redaction_analysis_v2.db."""
    db_path = tmp_path / "redaction_analysis_v2.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE redaction_scores (
            doc_id TEXT PRIMARY KEY,
            total_redactions INTEGER,
            proper_redactions INTEGER,
            improper_redactions INTEGER,
            redaction_density REAL,
            page_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE recovered_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            page_number INTEGER,
            text TEXT,
            confidence REAL
        )
    """)

    for i in range(20):
        conn.execute(
            "INSERT INTO redaction_scores"
            " (doc_id, total_redactions, proper_redactions,"
            " improper_redactions, redaction_density, page_count)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (f"sd-{i:04d}", 10, 8, 2, 0.3, 5),
        )

    conn.execute(
        "INSERT INTO recovered_text (doc_id, page_number, text, confidence) VALUES (?, ?, ?, ?)",
        ("sd-0001", 2, "Recovered text from page 2", 0.85),
    )

    conn.commit()
    conn.close()
    return tmp_path


def test_import_documents(mock_corpus_db: Path):
    importer = SeaDoughnutImporter(mock_corpus_db)
    count = importer.import_documents(limit=10)
    assert count == 10


def test_import_documents_full(mock_corpus_db: Path, tmp_path: Path):
    importer = SeaDoughnutImporter(mock_corpus_db)
    out_dir = tmp_path / "output"
    count = importer.import_documents(output_dir=out_dir)
    assert count == 50
    # Check that output files were created
    json_files = list(out_dir.rglob("*.json"))
    assert len(json_files) > 0


def test_import_redaction_scores(mock_redaction_db: Path):
    importer = SeaDoughnutImporter(mock_redaction_db)
    scores = importer.import_redaction_scores()
    assert len(scores) == 20
    assert scores[0].total_redactions == 10


def test_import_recovered_text(mock_redaction_db: Path):
    importer = SeaDoughnutImporter(mock_redaction_db)
    texts = importer.import_recovered_text()
    assert len(texts) == 1
    assert texts[0].page_number == 2


def test_import_transcripts(mock_corpus_db: Path):
    importer = SeaDoughnutImporter(mock_corpus_db)
    transcripts = importer.import_transcripts()
    assert len(transcripts) == 1
    assert transcripts[0].text == "Hello world transcript"


def test_import_entities(mock_corpus_db: Path):
    importer = SeaDoughnutImporter(mock_corpus_db)
    entities = importer.import_entities()
    assert len(entities) == 10
    assert entities[0].entity_type == "PERSON"


def test_import_missing_dir():
    with pytest.raises(FileNotFoundError):
        SeaDoughnutImporter(Path("/nonexistent/path"))
