"""Tests for Neon Postgres exporter (all DB calls mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from epstein_pipeline.config import Settings
from epstein_pipeline.exporters.neon_export import (
    NeonExporter,
    SemanticSearchResult,
    _batches,
    _mask_url,
    _require_psycopg,
)
from epstein_pipeline.models.document import EntityResult, Flight

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def neon_settings(tmp_path):
    """Settings with a fake Neon URL."""
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / ".cache",
        neon_database_url="postgresql://user:pass@host.neon.tech/db",
        neon_batch_size=2,
    )


@pytest.fixture
def mock_pool():
    """Create a fully-mocked AsyncConnectionPool."""
    cursor = AsyncMock()
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=(0,))

    conn = AsyncMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.commit = AsyncMock()

    # Support `async with pool.connection() as conn`
    pool = AsyncMock()
    conn_ctx = AsyncMock()
    conn_ctx.__aenter__ = AsyncMock(return_value=conn)
    conn_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.connection = MagicMock(return_value=conn_ctx)
    pool.open = AsyncMock()
    pool.close = AsyncMock()

    # Support `async with conn.cursor() as cur`
    cur_ctx = AsyncMock()
    cur_ctx.__aenter__ = AsyncMock(return_value=cursor)
    cur_ctx.__aexit__ = AsyncMock(return_value=False)
    conn.cursor = MagicMock(return_value=cur_ctx)

    return pool, conn, cursor


# ── Unit tests ───────────────────────────────────────────────────────


class TestBatchesHelper:
    def test_even_split(self):
        result = list(_batches([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_remainder(self):
        result = list(_batches([1, 2, 3], 2))
        assert result == [[1, 2], [3]]

    def test_empty(self):
        assert list(_batches([], 5)) == []


class TestMaskUrl:
    def test_masks_password(self):
        url = "postgresql://user:secret@host.neon.tech/db"
        assert _mask_url(url) == "postgresql://user:***@host.neon.tech/db"

    def test_no_password(self):
        url = "postgresql://host.neon.tech/db"
        assert _mask_url(url) == url


class TestSemanticSearchResult:
    def test_fields(self):
        r = SemanticSearchResult(
            document_id="doc-1",
            chunk_text="some text",
            chunk_index=0,
            similarity=0.95,
            title="A Document",
        )
        assert r.document_id == "doc-1"
        assert r.similarity == 0.95
        assert r.title == "A Document"

    def test_default_title_none(self):
        r = SemanticSearchResult(document_id="d", chunk_text="t", chunk_index=0, similarity=0.5)
        assert r.title is None


class TestRequirePsycopg:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", False)
    def test_raises_when_missing(self):
        with pytest.raises(ImportError, match="psycopg"):
            _require_psycopg()

    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    def test_no_error_when_present(self):
        _require_psycopg()  # should not raise


class TestNeonExporterInit:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    def test_requires_url(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
        )
        with pytest.raises(ValueError, match="EPSTEIN_NEON_DATABASE_URL"):
            NeonExporter(s)

    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    def test_init_stores_settings(self, neon_settings):
        exp = NeonExporter(neon_settings)
        assert exp.batch_size == 2
        assert "neon.tech" in exp.database_url


# ── Async upsert tests ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestUpsertDocuments:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    async def test_upsert_documents(self, neon_settings, mock_pool, sample_documents):
        pool, conn, cursor = mock_pool
        exp = NeonExporter(neon_settings)
        exp._pool = pool

        count = await exp.upsert_documents(sample_documents)
        assert count == len(sample_documents)
        assert cursor.execute.await_count >= len(sample_documents)

    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    async def test_upsert_empty(self, neon_settings):
        exp = NeonExporter(neon_settings)
        assert await exp.upsert_documents([]) == 0


@pytest.mark.asyncio
class TestUpsertPersons:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    async def test_upsert_persons(self, neon_settings, mock_pool, sample_persons):
        pool, conn, cursor = mock_pool
        exp = NeonExporter(neon_settings)
        exp._pool = pool

        count = await exp.upsert_persons(sample_persons)
        assert count == len(sample_persons)


@pytest.mark.asyncio
class TestUpsertFlights:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    async def test_upsert_flights(self, neon_settings, mock_pool):
        pool, conn, cursor = mock_pool
        exp = NeonExporter(neon_settings)
        exp._pool = pool

        flights = [
            Flight(
                id="f-001",
                date="1999-06-01",
                aircraft="Gulfstream",
                passengerIds=["p-0001"],
                pilotIds=["p-0099"],
            ),
        ]
        count = await exp.upsert_flights(flights)
        assert count == 1
        # flight + delete passengers + 1 passenger + 1 pilot = 4
        assert cursor.execute.await_count >= 4


@pytest.mark.asyncio
class TestUpsertEntities:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    async def test_upsert_entities(self, neon_settings, mock_pool):
        pool, conn, cursor = mock_pool
        exp = NeonExporter(neon_settings)
        exp._pool = pool

        entities = {
            "doc-001": [
                EntityResult(
                    entity_type="PERSON",
                    value="Jeffrey Epstein",
                    confidence=0.99,
                ),
            ],
        }
        count = await exp.upsert_entities(entities)
        assert count == 1


# ── Sync wrapper test ────────────────────────────────────────────────


class TestSyncWrapper:
    @patch("epstein_pipeline.exporters.neon_export.HAS_PSYCOPG", True)
    @patch("epstein_pipeline.exporters.neon_export.NeonExporter")
    def test_export_to_neon_sync(self, mock_exporter_cls, neon_settings):
        from epstein_pipeline.exporters.neon_export import (
            export_to_neon_sync,
        )

        mock_instance = AsyncMock()
        mock_instance.export_all = AsyncMock(return_value={"documents": 3})
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_exporter_cls.return_value = mock_instance

        result = export_to_neon_sync(neon_settings, documents=[])
        assert result == {"documents": 3}
