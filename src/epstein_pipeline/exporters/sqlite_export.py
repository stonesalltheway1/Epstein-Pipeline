"""SQLite exporter for the Epstein Pipeline.

Creates a self-contained SQLite database with documents, persons, and a
many-to-many join table.  Includes an FTS5 full-text search index over
document titles, summaries, and OCR text for fast keyword queries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.document import Document, Person

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Core tables
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    date          TEXT,
    source        TEXT NOT NULL,
    category      TEXT NOT NULL,
    summary       TEXT,
    pdf_url       TEXT,
    page_count    INTEGER,
    bates_range   TEXT,
    ocr_text      TEXT,
    tags          TEXT   -- semicolon-separated
);

CREATE TABLE IF NOT EXISTS persons (
    id         TEXT PRIMARY KEY,
    slug       TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    aliases    TEXT,    -- semicolon-separated
    category   TEXT NOT NULL,
    short_bio  TEXT
);

CREATE TABLE IF NOT EXISTS document_persons (
    document_id  TEXT NOT NULL REFERENCES documents(id),
    person_id    TEXT NOT NULL REFERENCES persons(id),
    PRIMARY KEY (document_id, person_id)
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_documents_date     ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_source   ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_persons_slug       ON persons(slug);
CREATE INDEX IF NOT EXISTS idx_persons_category   ON persons(category);
CREATE INDEX IF NOT EXISTS idx_dp_document        ON document_persons(document_id);
CREATE INDEX IF NOT EXISTS idx_dp_person          ON document_persons(person_id);

-- FTS5 full-text search index on documents
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    summary,
    ocr_text,
    content='documents',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync with the documents table
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, summary, ocr_text)
    VALUES (new.rowid, new.title, new.summary, new.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, summary, ocr_text)
    VALUES ('delete', old.rowid, old.title, old.summary, old.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, summary, ocr_text)
    VALUES ('delete', old.rowid, old.title, old.summary, old.ocr_text);
    INSERT INTO documents_fts(rowid, title, summary, ocr_text)
    VALUES (new.rowid, new.title, new.summary, new.ocr_text);
END;
"""


class SqliteExporter:
    """Export documents and persons to a SQLite database with FTS5 search."""

    def __init__(self) -> None:
        self._console = Console()

    def export(
        self,
        documents: list[Document],
        persons: list[Person],
        db_path: Path,
    ) -> Path:
        """Create a SQLite database with documents, persons, and FTS5 index.

        Parameters
        ----------
        documents:
            List of Document models to export.
        persons:
            List of Person models to export.
        db_path:
            Path for the output SQLite database file.

        Returns
        -------
        Path
            The path to the created database file.
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing database to start fresh
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(_SCHEMA_SQL)

            self._insert_persons(conn, persons)
            self._insert_documents(conn, documents)
            self._insert_document_persons(conn, documents)

            # Optimize the FTS index
            conn.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
            conn.execute("ANALYZE")
            conn.commit()

        finally:
            conn.close()

        size_mb = db_path.stat().st_size / (1024 * 1024)
        self._console.print(
            f"[green]Created SQLite database at {db_path.resolve()}[/green]"
        )
        self._console.print(
            f"  [green]Documents:[/green] {len(documents):,}"
        )
        self._console.print(
            f"  [green]Persons:[/green]   {len(persons):,}"
        )
        self._console.print(
            f"  [green]Size:[/green]      {size_mb:.1f} MB"
        )
        self._console.print(
            f"  [green]FTS5 index:[/green] title, summary, ocr_text"
        )
        return db_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_persons(self, conn: sqlite3.Connection, persons: list[Person]) -> None:
        """Insert all persons into the persons table."""
        rows = [
            (
                p.id,
                p.slug,
                p.name,
                "; ".join(p.aliases) if p.aliases else None,
                p.category,
                p.shortBio,
            )
            for p in persons
        ]

        conn.executemany(
            """
            INSERT OR REPLACE INTO persons (id, slug, name, aliases, category, short_bio)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} persons[/dim]")

    def _insert_documents(
        self, conn: sqlite3.Connection, documents: list[Document]
    ) -> None:
        """Insert all documents into the documents table."""
        rows = [
            (
                doc.id,
                doc.title,
                doc.date,
                doc.source,
                doc.category,
                doc.summary,
                doc.pdfUrl,
                doc.pageCount,
                doc.batesRange,
                doc.ocrText,
                "; ".join(doc.tags) if doc.tags else None,
            )
            for doc in documents
        ]

        conn.executemany(
            """
            INSERT OR REPLACE INTO documents
                (id, title, date, source, category, summary, pdf_url,
                 page_count, bates_range, ocr_text, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} documents[/dim]")

    def _insert_document_persons(
        self, conn: sqlite3.Connection, documents: list[Document]
    ) -> None:
        """Insert document-person relationships into the join table."""
        rows: list[tuple[str, str]] = []
        for doc in documents:
            for person_id in doc.personIds:
                rows.append((doc.id, person_id))

        if rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO document_persons (document_id, person_id)
                VALUES (?, ?)
                """,
                rows,
            )
            conn.commit()

        self._console.print(
            f"  [dim]Inserted {len(rows):,} document-person links[/dim]"
        )
