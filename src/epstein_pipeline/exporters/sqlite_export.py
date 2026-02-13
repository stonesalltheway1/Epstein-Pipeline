"""SQLite exporter for the Epstein Pipeline.

Creates a self-contained SQLite database with documents, persons, and a
many-to-many join table.  Includes FTS5 full-text search and additional
tables for redaction scores, recovered text, transcripts, extracted
entities, and extracted images -- matching the site's seed-sqlite.mjs schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console

from epstein_pipeline.models.document import Document, Person
from epstein_pipeline.models.forensics import (
    ExtractedEntity,
    ExtractedImage,
    RecoveredText,
    RedactionScore,
    Transcript,
)

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
    source_url    TEXT,
    archive_url   TEXT,
    page_count    INTEGER,
    bates_range   TEXT,
    ocr_text      TEXT,
    tags          TEXT,
    location_ids  TEXT,
    verification_status TEXT
);

CREATE TABLE IF NOT EXISTS persons (
    id         TEXT PRIMARY KEY,
    slug       TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    aliases    TEXT,
    category   TEXT NOT NULL,
    short_bio  TEXT
);

CREATE TABLE IF NOT EXISTS document_persons (
    document_id  TEXT NOT NULL REFERENCES documents(id),
    person_id    TEXT NOT NULL REFERENCES persons(id),
    PRIMARY KEY (document_id, person_id)
);

-- Forensic analysis tables
CREATE TABLE IF NOT EXISTS redaction_scores (
    document_id        TEXT PRIMARY KEY,
    total_redactions   INTEGER DEFAULT 0,
    proper_redactions  INTEGER DEFAULT 0,
    improper_redactions INTEGER DEFAULT 0,
    redaction_density  REAL DEFAULT 0,
    page_count         INTEGER
);

CREATE TABLE IF NOT EXISTS recovered_text (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    confidence   REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transcripts (
    document_id      TEXT PRIMARY KEY,
    source_path      TEXT,
    text             TEXT NOT NULL,
    language         TEXT DEFAULT 'en',
    duration_seconds REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    text         TEXT NOT NULL,
    confidence   REAL DEFAULT 0,
    person_id    TEXT
);

CREATE TABLE IF NOT EXISTS extracted_images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_number  INTEGER,
    image_index  INTEGER,
    width        INTEGER,
    height       INTEGER,
    format       TEXT,
    file_path    TEXT,
    description  TEXT,
    size_bytes   INTEGER DEFAULT 0
);

-- Vector embedding chunks (BLOB for portable SQLite; F32_BLOB for Turso)
CREATE TABLE IF NOT EXISTS document_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   BLOB,
    UNIQUE(document_id, chunk_index)
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_documents_date       ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_source     ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_category   ON documents(category);
CREATE INDEX IF NOT EXISTS idx_persons_slug         ON persons(slug);
CREATE INDEX IF NOT EXISTS idx_persons_category     ON persons(category);
CREATE INDEX IF NOT EXISTS idx_dp_document          ON document_persons(document_id);
CREATE INDEX IF NOT EXISTS idx_dp_person            ON document_persons(person_id);
CREATE INDEX IF NOT EXISTS idx_recovered_doc        ON recovered_text(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_doc         ON extracted_entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type        ON extracted_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_images_doc           ON extracted_images(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc           ON document_chunks(document_id);

-- FTS5 full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    summary,
    ocr_text,
    content='documents',
    content_rowid='rowid'
);

-- Triggers for FTS sync
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
    """Export documents, persons, and forensic data to a SQLite database."""

    def __init__(self) -> None:
        self._console = Console()

    def export(
        self,
        documents: list[Document],
        persons: list[Person],
        db_path: Path,
        *,
        redaction_scores: list[RedactionScore] | None = None,
        recovered_texts: list[RecoveredText] | None = None,
        transcripts: list[Transcript] | None = None,
        entities: list[ExtractedEntity] | None = None,
        images: list[ExtractedImage] | None = None,
    ) -> Path:
        """Create a SQLite database with all pipeline data.

        Parameters
        ----------
        documents:
            List of Document models.
        persons:
            List of Person models.
        db_path:
            Output database file path.
        redaction_scores:
            Optional redaction analysis scores.
        recovered_texts:
            Optional text recovered from redactions.
        transcripts:
            Optional audio/video transcripts.
        entities:
            Optional extracted entities.
        images:
            Optional extracted image metadata.

        Returns
        -------
        Path
            The path to the created database file.
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(_SCHEMA_SQL)

            self._insert_persons(conn, persons)
            self._insert_documents(conn, documents)
            self._insert_document_persons(conn, documents)

            if redaction_scores:
                self._insert_redaction_scores(conn, redaction_scores)
            if recovered_texts:
                self._insert_recovered_text(conn, recovered_texts)
            if transcripts:
                self._insert_transcripts(conn, transcripts)
            if entities:
                self._insert_entities(conn, entities)
            if images:
                self._insert_images(conn, images)

            # Optimize
            conn.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
            conn.execute("ANALYZE")
            conn.commit()

        finally:
            conn.close()

        size_mb = db_path.stat().st_size / (1024 * 1024)
        self._console.print(f"\n[green]Created SQLite database at {db_path.resolve()}[/green]")
        self._console.print(f"  Documents:        {len(documents):,}")
        self._console.print(f"  Persons:          {len(persons):,}")
        if redaction_scores:
            self._console.print(f"  Redaction scores: {len(redaction_scores):,}")
        if recovered_texts:
            self._console.print(f"  Recovered texts:  {len(recovered_texts):,}")
        if transcripts:
            self._console.print(f"  Transcripts:      {len(transcripts):,}")
        if entities:
            self._console.print(f"  Entities:         {len(entities):,}")
        if images:
            self._console.print(f"  Images:           {len(images):,}")
        self._console.print(f"  Size:             {size_mb:.1f} MB")
        self._console.print("  FTS5 index:       title, summary, ocr_text")

        return db_path

    # ------------------------------------------------------------------
    # Core table inserts
    # ------------------------------------------------------------------

    def _insert_persons(self, conn: sqlite3.Connection, persons: list[Person]) -> None:
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
            "INSERT OR REPLACE INTO persons"
            " (id, slug, name, aliases, category, short_bio)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} persons[/dim]")

    def _insert_documents(self, conn: sqlite3.Connection, documents: list[Document]) -> None:
        rows = [
            (
                doc.id,
                doc.title,
                doc.date,
                doc.source,
                doc.category,
                doc.summary,
                doc.pdfUrl,
                doc.sourceUrl,
                doc.archiveUrl,
                doc.pageCount,
                doc.batesRange,
                doc.ocrText,
                "; ".join(doc.tags) if doc.tags else None,
                "; ".join(doc.locationIds) if doc.locationIds else None,
                doc.verificationStatus,
            )
            for doc in documents
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO documents
               (id, title, date, source, category, summary, pdf_url, source_url, archive_url,
                page_count, bates_range, ocr_text, tags, location_ids, verification_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} documents[/dim]")

    def _insert_document_persons(self, conn: sqlite3.Connection, documents: list[Document]) -> None:
        rows = [(doc.id, pid) for doc in documents for pid in doc.personIds]
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO document_persons (document_id, person_id) VALUES (?, ?)",
                rows,
            )
            conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} document-person links[/dim]")

    # ------------------------------------------------------------------
    # Forensic table inserts
    # ------------------------------------------------------------------

    def _insert_redaction_scores(
        self, conn: sqlite3.Connection, scores: list[RedactionScore]
    ) -> None:
        rows = [
            (
                s.document_id,
                s.total_redactions,
                s.proper_redactions,
                s.improper_redactions,
                s.redaction_density,
                s.page_count,
            )
            for s in scores
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO redaction_scores
               (document_id, total_redactions, proper_redactions, improper_redactions,
                redaction_density, page_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} redaction scores[/dim]")

    def _insert_recovered_text(self, conn: sqlite3.Connection, texts: list[RecoveredText]) -> None:
        rows = [(t.document_id, t.page_number, t.text, t.confidence) for t in texts]
        conn.executemany(
            "INSERT INTO recovered_text"
            " (document_id, page_number, text, confidence)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} recovered text entries[/dim]")

    def _insert_transcripts(self, conn: sqlite3.Connection, transcripts: list[Transcript]) -> None:
        rows = [
            (t.document_id, t.source_path, t.text, t.language, t.duration_seconds)
            for t in transcripts
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO transcripts
               (document_id, source_path, text, language, duration_seconds)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} transcripts[/dim]")

    def _insert_entities(self, conn: sqlite3.Connection, entities: list[ExtractedEntity]) -> None:
        rows = [(e.document_id, e.entity_type, e.text, e.confidence, e.person_id) for e in entities]
        conn.executemany(
            "INSERT INTO extracted_entities"
            " (document_id, entity_type, text, confidence, person_id)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} entities[/dim]")

    def _insert_images(self, conn: sqlite3.Connection, images: list[ExtractedImage]) -> None:
        rows = [
            (
                i.document_id,
                i.page_number,
                i.image_index,
                i.width,
                i.height,
                i.format,
                i.file_path,
                i.description,
                i.size_bytes,
            )
            for i in images
        ]
        conn.executemany(
            """INSERT INTO extracted_images
               (document_id, page_number, image_index, width, height, format,
                file_path, description, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        self._console.print(f"  [dim]Inserted {len(rows):,} images[/dim]")
