"""Tests for Neo4j knowledge graph exporter (all driver calls mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from epstein_pipeline.config import Settings
from epstein_pipeline.exporters.neo4j_export import (
    Neo4jExporter,
    _batches,
    _neo4j_label,
    _neo4j_rel_type,
    _require_neo4j,
)
from epstein_pipeline.processors.knowledge_graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)

# ── Helper function tests ───────────────────────────────────────────────


class TestNeo4jLabel:
    def test_person(self):
        assert _neo4j_label("person") == "Person"

    def test_org(self):
        assert _neo4j_label("org") == "Organization"

    def test_location(self):
        assert _neo4j_label("location") == "Location"

    def test_document(self):
        assert _neo4j_label("document") == "Document"

    def test_unknown(self):
        assert _neo4j_label("widget") == "Entity"

    def test_case_insensitive(self):
        assert _neo4j_label("PERSON") == "Person"
        assert _neo4j_label("Person") == "Person"


class TestNeo4jRelType:
    def test_co_occurrence(self):
        assert _neo4j_rel_type("co-occurrence") == "CO_OCCURRENCE"

    def test_flew_with(self):
        assert _neo4j_rel_type("FLEW_WITH") == "FLEW_WITH"

    def test_co_passenger(self):
        assert _neo4j_rel_type("co-passenger") == "CO_PASSENGER"

    def test_correspondence(self):
        assert _neo4j_rel_type("correspondence") == "CORRESPONDENCE"

    def test_spaces_replaced(self):
        assert _neo4j_rel_type("financial link") == "FINANCIAL_LINK"


class TestBatches:
    def test_exact_split(self):
        items = [1, 2, 3, 4]
        result = list(_batches(items, 2))
        assert result == [[1, 2], [3, 4]]

    def test_remainder(self):
        items = [1, 2, 3, 4, 5]
        result = list(_batches(items, 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        assert list(_batches([], 10)) == []

    def test_single_batch(self):
        items = [1, 2]
        result = list(_batches(items, 10))
        assert result == [[1, 2]]


class TestRequireNeo4j:
    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", False)
    def test_raises_when_missing(self):
        with pytest.raises(ImportError, match="neo4j"):
            _require_neo4j()

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    def test_no_error_when_present(self):
        _require_neo4j()  # should not raise


# ── Exporter init tests ─────────────────────────────────────────────────


class TestNeo4jExporterInit:
    @pytest.fixture()
    def _settings(self, tmp_path):
        return Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
        )

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    def test_requires_uri(self, _settings):
        with pytest.raises(ValueError, match="EPSTEIN_NEO4J_URI"):
            Neo4jExporter(_settings)

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    def test_requires_password(self, _settings):
        _settings.neo4j_uri = "bolt://localhost:7687"
        with pytest.raises(ValueError, match="EPSTEIN_NEO4J_PASSWORD"):
            Neo4jExporter(_settings)

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    def test_init_stores_settings(self, _settings):
        _settings.neo4j_uri = "bolt://localhost:7687"
        _settings.neo4j_password = "test"
        exp = Neo4jExporter(_settings)
        assert exp.batch_size == 500
        assert exp.retry_max == 3


# ── Merge tests (mocked driver) ─────────────────────────────────────────


def _make_mock_driver():
    """Create a mock Neo4j async driver chain."""
    result = AsyncMock()
    result.consume = AsyncMock()
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    driver = AsyncMock()
    driver.session = MagicMock(return_value=session)
    return driver, session, result


@pytest.mark.asyncio
class TestMergeNodes:
    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_merge_nodes(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        driver, session, _ = _make_mock_driver()
        exp._driver = driver

        nodes = [
            GraphNode(id="p-0001", label="Jeffrey Epstein", type="person"),
            GraphNode(id="p-0002", label="Ghislaine Maxwell", type="person"),
        ]
        count = await exp.merge_nodes(nodes)
        assert count == 2
        session.run.assert_awaited()

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_merge_empty(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        assert await exp.merge_nodes([]) == 0

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_merge_mixed_types(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        driver, session, _ = _make_mock_driver()
        exp._driver = driver

        nodes = [
            GraphNode(id="p-0001", label="Jeffrey Epstein", type="person"),
            GraphNode(id="org-001", label="JP Morgan", type="org"),
            GraphNode(id="loc-001", label="Palm Beach", type="location"),
        ]
        count = await exp.merge_nodes(nodes)
        assert count == 3


@pytest.mark.asyncio
class TestMergeEdges:
    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_merge_edges(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        driver, session, _ = _make_mock_driver()
        exp._driver = driver

        edges = [
            GraphEdge(source="p-0001", target="p-0002", type="co-occurrence", weight=3.0),
            GraphEdge(source="p-0001", target="p-0003", type="FLEW_WITH", weight=2.0),
        ]
        count = await exp.merge_edges(edges)
        assert count == 2

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_merge_empty_edges(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        assert await exp.merge_edges([]) == 0


@pytest.mark.asyncio
class TestExportGraph:
    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_export_graph_calls_schema_and_merge(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        exp.ensure_schema = AsyncMock()
        exp.merge_nodes = AsyncMock(return_value=3)
        exp.merge_edges = AsyncMock(return_value=2)

        graph = KnowledgeGraph(
            nodes=[GraphNode(id="a", label="A", type="person")],
            edges=[GraphEdge(source="a", target="b", type="co-occurrence")],
        )
        counts = await exp.export_graph(graph)
        assert counts == {"nodes": 3, "edges": 2}
        exp.ensure_schema.assert_awaited_once()
        exp.merge_nodes.assert_awaited_once()
        exp.merge_edges.assert_awaited_once()

    @patch("epstein_pipeline.exporters.neo4j_export.HAS_NEO4J", True)
    async def test_export_empty_graph(self, tmp_path):
        s = Settings(
            data_dir=tmp_path / "d",
            output_dir=tmp_path / "o",
            cache_dir=tmp_path / "c",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="test",
        )
        exp = Neo4jExporter(s)
        exp.ensure_schema = AsyncMock()
        exp.merge_nodes = AsyncMock(return_value=0)
        exp.merge_edges = AsyncMock(return_value=0)

        graph = KnowledgeGraph()
        counts = await exp.export_graph(graph)
        assert counts == {"nodes": 0, "edges": 0}
