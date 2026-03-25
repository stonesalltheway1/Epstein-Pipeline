"""Idempotent Neon Postgres schema migration.

Creates all tables, indexes, and extensions needed by the pipeline.
Safe to run repeatedly — uses IF NOT EXISTS throughout.

Usage:
    epstein-pipeline migrate --database-url=$NEON_DATABASE_URL
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Schema version tracking ──────────────────────────────────────────────────

SCHEMA_VERSION = 3

MIGRATION_SQL = """
-- ============================================================================
-- Epstein Pipeline v1.0 — Neon Postgres Schema
-- ============================================================================

-- Enable pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for fast text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Schema version ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Documents ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    date                DATE,
    source              TEXT NOT NULL,
    category            TEXT NOT NULL,
    summary             TEXT,
    tags                TEXT[] NOT NULL DEFAULT '{}',
    pdf_url             TEXT,
    source_url          TEXT,
    archive_url         TEXT,
    page_count          INTEGER,
    bates_range         TEXT,
    ocr_text            TEXT,
    location_ids        TEXT[] NOT NULL DEFAULT '{}',
    verification_status TEXT,
    ocr_confidence      REAL,
    classified_category TEXT,
    content_hash        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents (category);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents (date);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm ON documents USING gin (title gin_trgm_ops);

-- ── Persons ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS persons (
    id          TEXT PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    category    TEXT NOT NULL,
    short_bio   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_persons_slug ON persons (slug);
CREATE INDEX IF NOT EXISTS idx_persons_name_trgm ON persons USING gin (name gin_trgm_ops);

-- ── Document ↔ Person join table ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS document_persons (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    person_id   TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, person_id)
);

CREATE INDEX IF NOT EXISTS idx_document_persons_person ON document_persons (person_id);

-- ── Entities (NER results) ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS entities (
    id              SERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL,  -- PERSON, ORG, PHONE, EMAIL_ADDR, CASE_NUMBER, etc.
    entity_value    TEXT NOT NULL,
    confidence      REAL,
    entity_source   TEXT,  -- 'spacy', 'gliner', 'regex'
    source_span     TEXT,  -- original text context
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entities_document ON entities (document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_value_trgm
    ON entities USING gin (entity_value gin_trgm_ops);

-- ── Document embeddings (pgvector) ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS document_embeddings (
    id              SERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       vector(768),
    model_name      TEXT NOT NULL DEFAULT 'nomic-ai/nomic-embed-text-v2-moe',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index, model_name)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_document ON document_embeddings (document_id);

-- HNSW index for fast approximate nearest neighbor search.
-- HNSW outperforms IVFFlat at all dataset sizes (no minimum row requirement,
-- better recall, incremental inserts). Tuned for 2M+ vectors on Neon Scale.
-- m=24 for better recall at scale; ef_construction=100 for index quality.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_embeddings_vector_cosine'
    ) THEN
        CREATE INDEX idx_embeddings_vector_cosine
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 100);
    END IF;
END $$;

-- ── Duplicate clusters ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS duplicate_clusters (
    id                  SERIAL PRIMARY KEY,
    cluster_id          TEXT NOT NULL,
    document_id         TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    is_representative   BOOLEAN NOT NULL DEFAULT false,
    similarity          REAL NOT NULL,
    dedup_method        TEXT NOT NULL,  -- 'exact', 'minhash', 'semantic'
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dup_clusters_cluster ON duplicate_clusters (cluster_id);
CREATE INDEX IF NOT EXISTS idx_dup_clusters_document ON duplicate_clusters (document_id);

-- ── Relationships (knowledge graph edges) ───────────────────────────────────

CREATE TABLE IF NOT EXISTS relationships (
    id                  SERIAL PRIMARY KEY,
    person1_id          TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    person2_id          TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL,  -- FLEW_WITH, EMPLOYED_BY, ASSOCIATED_WITH, etc.
    weight              REAL NOT NULL DEFAULT 1.0,
    evidence_doc_id     TEXT REFERENCES documents(id) ON DELETE SET NULL,
    context_snippet     TEXT,
    extraction_method   TEXT,  -- 'co-occurrence', 'llm', 'manual'
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (person1_id, person2_id, relationship_type, evidence_doc_id)
);

CREATE INDEX IF NOT EXISTS idx_relationships_p1 ON relationships (person1_id);
CREATE INDEX IF NOT EXISTS idx_relationships_p2 ON relationships (person2_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (relationship_type);

-- ── Emails ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS emails (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    from_name   TEXT,
    from_email  TEXT,
    from_person_slug TEXT,
    date        DATE,
    body        TEXT NOT NULL,
    folder      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS email_recipients (
    id          SERIAL PRIMARY KEY,
    email_id    TEXT NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    recipient_type TEXT NOT NULL,  -- 'to', 'cc'
    name        TEXT,
    email       TEXT,
    person_slug TEXT
);

CREATE INDEX IF NOT EXISTS idx_email_recipients_email ON email_recipients (email_id);

CREATE TABLE IF NOT EXISTS email_persons (
    email_id    TEXT NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    person_id   TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    PRIMARY KEY (email_id, person_id)
);

-- ── Flights ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS flights (
    id              TEXT PRIMARY KEY,
    date            DATE,
    aircraft        TEXT,
    tail_number     TEXT,
    origin          TEXT,
    destination     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS flight_passengers (
    flight_id   TEXT NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
    person_id   TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'passenger',  -- 'passenger' or 'pilot'
    PRIMARY KEY (flight_id, person_id, role)
);

CREATE INDEX IF NOT EXISTS idx_flight_passengers_person ON flight_passengers (person_id);

-- ── Locations ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS locations (
    id          TEXT PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    lat         REAL,
    lon         REAL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Document integrity hashes ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS document_hashes (
    doc_id          TEXT PRIMARY KEY,
    sha256          TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    file_path       TEXT,
    file_size       BIGINT,
    verify_status   TEXT DEFAULT 'pending',
    verified_at     TIMESTAMPTZ,
    forensics       JSONB
);

CREATE INDEX IF NOT EXISTS idx_dh_status ON document_hashes (verify_status);
CREATE INDEX IF NOT EXISTS idx_dh_dataset ON document_hashes (dataset);

-- ── Document change log (integrity events) ──────────────────────────────────

CREATE TABLE IF NOT EXISTS document_changes (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    doc_id          TEXT NOT NULL,
    change_type     TEXT NOT NULL,
    dataset         TEXT,
    detected_at     TIMESTAMPTZ DEFAULT now(),
    detected_by     TEXT NOT NULL,
    old_sha256      TEXT,
    new_sha256      TEXT,
    http_status     INTEGER,
    details         JSONB DEFAULT '{}'::jsonb,
    lead_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_dc_doc ON document_changes (doc_id);
CREATE INDEX IF NOT EXISTS idx_dc_type ON document_changes (change_type);
CREATE INDEX IF NOT EXISTS idx_dc_date ON document_changes (detected_at DESC);

-- ── Updated-at trigger ──────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'set_updated_at_documents'
    ) THEN
        CREATE TRIGGER set_updated_at_documents
            BEFORE UPDATE ON documents
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END $$;

-- ── Semantic search function ────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    document_id TEXT,
    chunk_text TEXT,
    chunk_index INTEGER,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        de.document_id,
        de.chunk_text,
        de.chunk_index,
        1 - (de.embedding <=> query_embedding) AS similarity
    FROM document_embeddings de
    WHERE 1 - (de.embedding <=> query_embedding) > match_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ══════════════════════════════════════════════════════════════════════════════
-- v3: Video Depositions & International Court Documents
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS video_depositions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    deponent TEXT,
    deponent_person_id TEXT,
    case_name TEXT,
    deposition_date DATE,
    duration_seconds FLOAT,
    source_type TEXT,
    source_url TEXT,
    video_url TEXT,
    thumbnail_url TEXT,
    language TEXT DEFAULT 'en',
    speaker_count INTEGER DEFAULT 0,
    segment_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_depositions_deponent ON video_depositions (deponent_person_id);
CREATE INDEX IF NOT EXISTS idx_depositions_date ON video_depositions (deposition_date);
CREATE INDEX IF NOT EXISTS idx_depositions_case ON video_depositions (case_name);

CREATE TABLE IF NOT EXISTS deposition_segments (
    id SERIAL PRIMARY KEY,
    deposition_id TEXT NOT NULL REFERENCES video_depositions(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    speaker TEXT,
    speaker_person_id TEXT,
    text TEXT NOT NULL,
    embedding vector(768),
    UNIQUE(deposition_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_dep_segments_deposition ON deposition_segments (deposition_id);
CREATE INDEX IF NOT EXISTS idx_dep_segments_speaker ON deposition_segments (speaker_person_id);
CREATE INDEX IF NOT EXISTS idx_dep_segments_embedding
    ON deposition_segments USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 100);

-- Full-text search on segment text
ALTER TABLE deposition_segments
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS idx_dep_segments_fts ON deposition_segments USING GIN (tsv);

-- International court document columns on existing documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS jurisdiction TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS language_code TEXT DEFAULT 'en';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS court_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS case_number_local TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_jurisdiction ON documents (jurisdiction);

-- ── Record migration version ────────────────────────────────────────────────

INSERT INTO schema_migrations (version)
VALUES (3)
ON CONFLICT (version) DO NOTHING;
"""


async def run_migration(database_url: str) -> None:
    """Run the schema migration against a Neon Postgres database.

    Uses psycopg (v3) async connection for non-blocking execution.
    """
    try:
        import psycopg
    except ImportError:
        raise ImportError(
            "psycopg is required for Neon migrations. "
            "Install with: pip install 'epstein-pipeline[neon]'"
        )

    logger.info("Connecting to Neon Postgres...")
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        async with conn.cursor() as cur:
            # Check current schema version
            try:
                await cur.execute("SELECT max(version) FROM schema_migrations")
                row = await cur.fetchone()
                current_version = row[0] if row and row[0] else 0
            except Exception:
                current_version = 0
                await conn.rollback()

            if current_version >= SCHEMA_VERSION:
                logger.info(
                    f"Schema already at version {current_version} "
                    f"(target: {SCHEMA_VERSION}). Nothing to do."
                )
                return

            logger.info(f"Migrating schema from v{current_version} to v{SCHEMA_VERSION}...")
            await cur.execute(MIGRATION_SQL)
            await conn.commit()

    logger.info(f"Schema migration to v{SCHEMA_VERSION} complete.")


def run_migration_sync(database_url: str) -> None:
    """Synchronous wrapper for run_migration (for CLI use)."""
    import asyncio

    asyncio.run(run_migration(database_url))
