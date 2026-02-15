"""Tests for Sea_Doughnut importer.

Tests mock the exact SQLite schemas produced by the Sea_Doughnut pipeline:
- full_text_corpus.db: documents(efta_number, dataset, total_pages, file_size)
                       pages(efta_number, page_number, text_content, char_count)
- redaction_analysis_v2.db: document_summary(efta_number, total_redactions, ...)
                            redactions(efta_number, page_number, hidden_text, ...)
- transcripts.db: transcripts(efta_number, file_path, transcript, ...)
- concordance_metadata.db: provenance_map, sdny_efta_bridge, productions, opt_documents
- persons_registry.json: [{name, slug, aliases, category, ...}, ...]
"""

import json
import sqlite3
from pathlib import Path

import pytest

from epstein_pipeline.importers.sea_doughnut import (
    SeaDoughnutImporter,
    efta_to_dataset,
    efta_to_doj_url,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a mock Sea_Doughnut data directory with all databases."""
    _create_corpus_db(tmp_path)
    _create_redaction_db(tmp_path)
    _create_transcripts_db(tmp_path)
    _create_concordance_db(tmp_path)
    _create_persons_json(tmp_path)
    return tmp_path


def _create_corpus_db(tmp_path: Path) -> None:
    """Create mock full_text_corpus.db with actual schema."""
    db_path = tmp_path / "full_text_corpus.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            efta_number TEXT PRIMARY KEY,
            dataset INTEGER,
            total_pages INTEGER,
            file_size INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE pages (
            efta_number TEXT,
            page_number INTEGER,
            text_content TEXT,
            char_count INTEGER,
            PRIMARY KEY (efta_number, page_number)
        )
    """)

    # Insert 50 mock documents across datasets
    for i in range(50):
        efta_num = 1000 + i
        efta = f"EFTA{efta_num:08d}"
        ds = 1
        pages = 3
        conn.execute(
            "INSERT INTO documents (efta_number, dataset, total_pages, file_size) "
            "VALUES (?, ?, ?, ?)",
            (efta, ds, pages, 50000),
        )
        for p in range(1, pages + 1):
            conn.execute(
                "INSERT INTO pages (efta_number, page_number, text_content, char_count) "
                "VALUES (?, ?, ?, ?)",
                (efta, p, f"Page {p} text for {efta}. Contains case information.", 48),
            )

    conn.commit()
    conn.close()


def _create_redaction_db(tmp_path: Path) -> None:
    """Create mock redaction_analysis_v2.db with actual schema."""
    db_path = tmp_path / "redaction_analysis_v2.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE document_summary (
            efta_number TEXT PRIMARY KEY,
            total_redactions INTEGER,
            bad_redactions INTEGER,
            proper_redactions INTEGER,
            has_recoverable_text BOOLEAN
        )
    """)
    conn.execute("""
        CREATE TABLE redactions (
            efta_number TEXT,
            page_number INTEGER,
            hidden_text TEXT,
            confidence REAL,
            redaction_type TEXT
        )
    """)

    for i in range(20):
        efta = f"EFTA{1000 + i:08d}"
        conn.execute(
            "INSERT INTO document_summary "
            "(efta_number, total_redactions, bad_redactions,"
            " proper_redactions, has_recoverable_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (efta, 10, 2, 8, i < 5),
        )

    # A few recovered text entries
    _redact_sql = (
        "INSERT INTO redactions "
        "(efta_number, page_number, hidden_text, confidence, redaction_type) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    conn.execute(
        _redact_sql,
        ("EFTA00001001", 2, "Recovered text from page 2 of document", 0.85, "bad_overlay"),
    )
    conn.execute(
        _redact_sql,
        ("EFTA00001003", 1, "ab", 0.2, "bad_overlay"),  # too short, should be filtered
    )
    conn.execute(
        _redact_sql,
        ("EFTA00001005", 3, None, 0.0, "proper"),  # null text, should be filtered
    )

    conn.commit()
    conn.close()


def _create_transcripts_db(tmp_path: Path) -> None:
    """Create mock transcripts.db with actual schema."""
    db_path = tmp_path / "transcripts.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE transcripts (
            efta_number TEXT PRIMARY KEY,
            file_path TEXT,
            file_type TEXT,
            duration_secs REAL,
            language TEXT,
            transcript TEXT,
            word_count INTEGER,
            dataset_source TEXT
        )
    """)

    conn.execute(
        "INSERT INTO transcripts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "EFTA00001500",
            "/path/to/audio.m4a",
            "m4a",
            120.5,
            "en",
            "Hello world this is a test transcript",
            7,
            "ds1",
        ),
    )
    conn.execute(
        "INSERT INTO transcripts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "EFTA00001501",
            "/path/to/silent.mp4",
            "mp4",
            3625.0,
            "en",
            "",
            0,
            "ds9",
        ),  # empty transcript, should be skipped
    )

    conn.commit()
    conn.close()


def _create_concordance_db(tmp_path: Path) -> None:
    """Create mock concordance_metadata.db with actual schema."""
    db_path = tmp_path / "concordance_metadata.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE provenance_map (
            dataset INTEGER,
            efta_start TEXT,
            efta_end TEXT,
            efta_start_num INTEGER,
            efta_end_num INTEGER,
            sdny_gm_start TEXT,
            sdny_gm_end TEXT,
            source_description TEXT,
            source_category TEXT,
            doc_count INTEGER,
            page_count INTEGER,
            confidence TEXT
        )
    """)
    conn.execute(
        "INSERT INTO provenance_map VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "EFTA00000001",
            "EFTA00003158",
            1,
            3158,
            "SDNY_GM_000001",
            "SDNY_GM_003158",
            "Initial prosecution files",
            "prosecution",
            3158,
            15000,
            "high",
        ),
    )

    conn.execute("""
        CREATE TABLE sdny_efta_bridge (
            efta_number TEXT,
            sdny_gm_number TEXT
        )
    """)
    for i in range(100):
        conn.execute(
            "INSERT INTO sdny_efta_bridge VALUES (?, ?)",
            (f"EFTA{i + 1:08d}", f"SDNY_GM_{i + 1:08d}"),
        )

    conn.execute("""
        CREATE TABLE productions (
            id INTEGER PRIMARY KEY,
            description TEXT
        )
    """)
    conn.execute("INSERT INTO productions VALUES (1, 'Test production')")

    conn.execute("""
        CREATE TABLE opt_documents (
            efta_number TEXT PRIMARY KEY,
            page_count INTEGER
        )
    """)
    for i in range(50):
        conn.execute(
            "INSERT INTO opt_documents VALUES (?, ?)",
            (f"EFTA{1000 + i:08d}", 3),
        )

    conn.commit()
    conn.close()


def _create_persons_json(tmp_path: Path) -> None:
    """Create mock persons_registry.json."""
    persons = [
        {
            "name": "Jeffrey Epstein",
            "slug": "jeffrey-epstein",
            "aliases": ["JE"],
            "category": "key-figure",
            "description": "Convicted sex trafficker",
        },
        {
            "name": "Ghislaine Maxwell",
            "slug": "ghislaine-maxwell",
            "aliases": ["G-Max"],
            "category": "associate",
        },
        {
            "name": "",  # empty name, should be skipped
            "slug": "",
            "aliases": [],
            "category": "unknown",
        },
    ]
    (tmp_path / "persons_registry.json").write_text(json.dumps(persons, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestEftaMapping:
    def test_efta_to_dataset_ds1(self):
        assert efta_to_dataset(1000) == 1

    def test_efta_to_dataset_ds9(self):
        assert efta_to_dataset(100000) == 9

    def test_efta_to_dataset_ds12(self):
        assert efta_to_dataset(2731000) == 12

    def test_efta_to_dataset_gap_falls_back(self):
        # 5587-5704 is a gap between DS3 and DS4
        result = efta_to_dataset(5600)
        assert result == 3  # falls back to nearest lower

    def test_efta_to_doj_url(self):
        url = efta_to_doj_url("EFTA00001000")
        assert url == "https://www.justice.gov/epstein/files/DataSet%201/EFTA00001000.pdf"

    def test_efta_to_doj_url_ds10(self):
        url = efta_to_doj_url("EFTA01500000")
        assert url == "https://www.justice.gov/epstein/files/DataSet%2010/EFTA01500000.pdf"

    def test_efta_to_doj_url_invalid(self):
        assert efta_to_doj_url("not-an-efta") is None


# ---------------------------------------------------------------------------
# Integration tests: importer
# ---------------------------------------------------------------------------


class TestImporter:
    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            SeaDoughnutImporter(Path("/nonexistent/path"))

    def test_import_documents(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        count = importer.import_documents(limit=10)
        assert count == 10

    def test_import_documents_full(self, data_dir: Path, tmp_path: Path):
        importer = SeaDoughnutImporter(data_dir)
        out_dir = tmp_path / "output"
        count = importer.import_documents(output_dir=out_dir)
        assert count == 50
        # Check NDJSON output files were created
        ndjson_files = list(out_dir.rglob("*.ndjson"))
        assert len(ndjson_files) > 0
        # Verify NDJSON format (one JSON object per line)
        with open(ndjson_files[0]) as f:
            first_line = f.readline()
            doc = json.loads(first_line)
            assert "id" in doc
            assert doc["id"].startswith("EFTA")
            assert "ocrText" in doc
            assert "pdfUrl" in doc
            assert doc["pdfUrl"].startswith("https://www.justice.gov/")

    def test_import_redaction_scores(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        scores = importer.import_redaction_scores()
        assert len(scores) == 20
        assert scores[0].total_redactions == 10
        assert scores[0].proper_redactions == 8
        assert scores[0].improper_redactions == 2

    def test_import_recovered_text(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        texts = importer.import_recovered_text()
        # Only 1 entry should pass the LENGTH(hidden_text) > 3 filter
        assert len(texts) == 1
        assert texts[0].page_number == 2
        assert "Recovered text" in texts[0].text

    def test_import_transcripts(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        transcripts = importer.import_transcripts()
        # Only 1 transcript has content (the other is empty)
        assert len(transcripts) == 1
        assert transcripts[0].text == "Hello world this is a test transcript"
        assert transcripts[0].duration_seconds == 120.5

    def test_import_persons(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        persons = importer.import_persons()
        # 3 entries but 1 has empty name, so 2
        assert len(persons) == 2
        assert persons[0]["name"] == "Jeffrey Epstein"
        assert persons[0]["shortBio"] == "Convicted sex trafficker"
        assert persons[1]["name"] == "Ghislaine Maxwell"
        assert "shortBio" not in persons[1]  # no description

    def test_import_concordance(self, data_dir: Path):
        importer = SeaDoughnutImporter(data_dir)
        summary = importer.import_concordance_summary()
        assert summary is not None
        assert len(summary.provenance_ranges) == 1
        assert summary.provenance_ranges[0].dataset == 1
        assert summary.sdny_bridge_count == 100
        assert summary.production_count == 1
        assert summary.opt_document_count == 50

    def test_import_all(self, data_dir: Path, tmp_path: Path):
        importer = SeaDoughnutImporter(data_dir)
        out_dir = tmp_path / "output"
        corpus = importer.import_all(output_dir=out_dir)

        assert corpus.document_count == 50
        assert len(corpus.redaction_scores) == 20
        assert len(corpus.recovered_texts) == 1
        assert len(corpus.transcripts) == 1
        assert corpus.concordance is not None
        assert corpus.concordance.sdny_bridge_count == 100

        # Check output files
        assert (out_dir / "persons-registry.json").exists()
        assert (out_dir / "concordance-summary.json").exists()

    def test_entities_returns_empty(self, data_dir: Path):
        """Entities are not pre-extracted; importer returns empty list."""
        importer = SeaDoughnutImporter(data_dir)
        entities = importer.import_entities()
        assert entities == []

    def test_images_returns_empty(self, data_dir: Path):
        """Images are on-disk, not in DB; importer returns empty list."""
        importer = SeaDoughnutImporter(data_dir)
        images = importer.import_images()
        assert images == []
