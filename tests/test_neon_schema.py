"""Tests for the Neon Postgres schema migration module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from epstein_pipeline.exporters.neon_schema import (
    MIGRATION_SQL,
    SCHEMA_VERSION,
    run_migration,
    run_migration_sync,
)

# ── MIGRATION_SQL content checks ──────────────────────────────


class TestMigrationSQL:
    """Verify MIGRATION_SQL contains expected DDL."""

    def test_is_nonempty_string(self):
        assert isinstance(MIGRATION_SQL, str)
        assert len(MIGRATION_SQL) > 100

    _TABLES = [
        "documents",
        "persons",
        "document_persons",
        "entities",
        "document_embeddings",
        "duplicate_clusters",
        "relationships",
        "emails",
        "email_recipients",
        "email_persons",
        "flights",
        "flight_passengers",
        "locations",
        "schema_migrations",
    ]

    @pytest.mark.parametrize("table", _TABLES)
    def test_contains_table(self, table: str):
        stmt = f"CREATE TABLE IF NOT EXISTS {table}"
        assert stmt in MIGRATION_SQL

    def test_contains_pgvector_extension(self):
        assert "CREATE EXTENSION IF NOT EXISTS vector" in MIGRATION_SQL

    def test_contains_trgm_extension(self):
        assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in MIGRATION_SQL

    _INDEXES = [
        "idx_documents_source",
        "idx_documents_category",
        "idx_persons_slug",
        "idx_entities_document",
        "idx_embeddings_document",
        "idx_relationships_p1",
        "idx_relationships_p2",
    ]

    @pytest.mark.parametrize("index", _INDEXES)
    def test_contains_index(self, index: str):
        assert index in MIGRATION_SQL

    def test_contains_semantic_search_fn(self):
        assert "CREATE OR REPLACE FUNCTION semantic_search" in MIGRATION_SQL
        assert "query_embedding vector(768)" in MIGRATION_SQL

    def test_schema_version_positive(self):
        assert SCHEMA_VERSION >= 1


# ── run_migration ─────────────────────────────────────────────


def _make_mock_cur(**kwargs):
    """Build async cursor mock supporting async-with."""
    cur = AsyncMock(**kwargs)
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    return cur


def _make_mock_conn(cur):
    """Build async connection mock supporting async-with."""
    conn = AsyncMock()
    # cursor() is a sync call returning an async context manager
    conn.cursor = MagicMock(return_value=cur)
    conn.rollback = AsyncMock()
    conn.commit = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    return conn


@pytest.mark.asyncio
async def test_run_migration_executes_sql():
    """Mock psycopg and verify migration SQL is executed."""
    mock_cur = _make_mock_cur(
        **{
            "execute": AsyncMock(
                side_effect=[Exception("no table"), None],
            ),
        },
    )
    mock_conn = _make_mock_conn(mock_cur)

    mock_psycopg = MagicMock()
    mock_psycopg.AsyncConnection.connect = AsyncMock(
        return_value=mock_conn,
    )

    with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
        await run_migration("postgres://fake:5432/test")

    # Rolled back after version-check failure, then executed SQL.
    mock_conn.rollback.assert_awaited_once()
    mock_cur.execute.assert_any_await(MIGRATION_SQL)
    mock_conn.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_migration_skips_if_current():
    """If schema is already at target version, skip migration."""
    mock_cur = _make_mock_cur()
    mock_cur.fetchone = AsyncMock(return_value=(SCHEMA_VERSION,))
    mock_conn = _make_mock_conn(mock_cur)

    mock_psycopg = MagicMock()
    mock_psycopg.AsyncConnection.connect = AsyncMock(
        return_value=mock_conn,
    )

    with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
        await run_migration("postgres://fake:5432/test")

    # MIGRATION_SQL should NOT have been executed
    calls = [c.args[0] for c in mock_cur.execute.await_args_list]
    assert MIGRATION_SQL not in calls


# ── run_migration_sync ────────────────────────────────────────


def test_run_migration_sync_calls_asyncio_run():
    """Verify the sync wrapper delegates to asyncio.run()."""
    with patch("asyncio.run") as mock_run:
        run_migration_sync("postgres://fake:5432/test")
        mock_run.assert_called_once()
        coro = mock_run.call_args[0][0]
        # The coroutine should be from run_migration
        assert coro is not None
