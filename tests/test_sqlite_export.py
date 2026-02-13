"""Tests for SQLite export."""

import sqlite3
from pathlib import Path

from epstein_pipeline.exporters.sqlite_export import SqliteExporter
from epstein_pipeline.models.forensics import RecoveredText, RedactionScore


def test_basic_export(tmp_path: Path, sample_documents, sample_persons):
    db_path = tmp_path / "test.db"
    exporter = SqliteExporter()
    result_path = exporter.export(sample_documents, sample_persons, db_path)

    assert result_path.exists()
    conn = sqlite3.connect(str(result_path))

    # Check documents
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 3

    # Check persons
    count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    assert count == 3

    # Check document_persons
    count = conn.execute("SELECT COUNT(*) FROM document_persons").fetchone()[0]
    assert count > 0

    # Check FTS
    results = conn.execute(
        "SELECT id FROM documents WHERE rowid IN"
        " (SELECT rowid FROM documents_fts"
        " WHERE documents_fts MATCH 'Epstein')"
    ).fetchall()
    assert len(results) > 0

    conn.close()


def test_export_with_forensic_data(tmp_path: Path, sample_documents, sample_persons):
    db_path = tmp_path / "test_forensic.db"
    exporter = SqliteExporter()

    scores = [
        RedactionScore(
            document_id="doc-001",
            total_redactions=5,
            proper_redactions=3,
            improper_redactions=2,
        ),
    ]
    recovered = [
        RecoveredText(
            document_id="doc-001",
            page_number=1,
            text="Hidden text",
            confidence=0.9,
        ),
    ]

    exporter.export(
        sample_documents,
        sample_persons,
        db_path,
        redaction_scores=scores,
        recovered_texts=recovered,
    )

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM redaction_scores").fetchone()[0]
    assert count == 1

    count = conn.execute("SELECT COUNT(*) FROM recovered_text").fetchone()[0]
    assert count == 1
    conn.close()


def test_export_empty(tmp_path: Path):
    db_path = tmp_path / "empty.db"
    exporter = SqliteExporter()
    exporter.export([], [], db_path)
    assert db_path.exists()
