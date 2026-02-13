"""Tests for knowledge graph builder."""

import json
from pathlib import Path

from epstein_pipeline.models.document import Document
from epstein_pipeline.processors.knowledge_graph import KnowledgeGraphBuilder


def test_build_graph_from_documents(sample_documents):
    builder = KnowledgeGraphBuilder()
    builder.add_documents(sample_documents)
    graph = builder.build()

    assert graph.node_count > 0
    assert graph.edge_count >= 0


def test_co_occurrence_edges():
    docs = [
        Document(
            id="doc-1",
            title="Test",
            source="other",
            category="other",
            personIds=["p-0001", "p-0002", "p-0003"],
        ),
    ]
    builder = KnowledgeGraphBuilder()
    builder.add_documents(docs)
    graph = builder.build()

    # 3 persons should create 3 co-occurrence edges (3 choose 2)
    assert graph.node_count == 3
    assert graph.edge_count == 3


def test_export_json(tmp_path: Path, sample_documents):
    builder = KnowledgeGraphBuilder()
    builder.add_documents(sample_documents)
    graph = builder.build()

    out_path = tmp_path / "graph.json"
    KnowledgeGraphBuilder.export_json(graph, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "nodes" in data
    assert "links" in data


def test_export_gexf(tmp_path: Path, sample_documents):
    builder = KnowledgeGraphBuilder()
    builder.add_documents(sample_documents)
    graph = builder.build()

    out_path = tmp_path / "graph.gexf"
    KnowledgeGraphBuilder.export_gexf(graph, out_path)

    assert out_path.exists()
    content = out_path.read_text()
    assert "<gexf" in content
    assert "node" in content


def test_empty_graph():
    builder = KnowledgeGraphBuilder()
    graph = builder.build()
    assert graph.node_count == 0
    assert graph.edge_count == 0
