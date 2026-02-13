"""Knowledge graph builder.

Builds a weighted entity-relationship graph from pipeline output.
Nodes: persons, organizations, locations, documents.
Edges: co-occurrence, co-passengers, correspondence, financial.
Export: GEXF (Gephi), JSON (D3.js-compatible).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from epstein_pipeline.models.document import Document, Email, Flight

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    label: str
    type: str  # "person", "org", "location", "document"
    attributes: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source: str
    target: str
    type: str  # "co-occurrence", "co-passenger", "correspondence", "financial"
    weight: float = 1.0
    attributes: dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """A weighted entity-relationship graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class KnowledgeGraphBuilder:
    """Build a knowledge graph from pipeline-processed data."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edge_counter: dict[tuple[str, str, str], float] = Counter()
        self._edge_attrs: dict[tuple[str, str, str], dict] = defaultdict(dict)

    def _add_node(self, node_id: str, label: str, node_type: str, **attrs) -> None:
        """Add or update a node."""
        if node_id not in self._nodes:
            self._nodes[node_id] = GraphNode(
                id=node_id, label=label, type=node_type, attributes=attrs
            )
        else:
            self._nodes[node_id].attributes.update(attrs)

    def _add_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0, **attrs
    ) -> None:
        """Add or increment an edge."""
        # Normalize edge direction (alphabetical order)
        if source > target:
            source, target = target, source
        key = (source, target, edge_type)
        self._edge_counter[key] += weight
        self._edge_attrs[key].update(attrs)

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add document-based co-occurrence edges."""
        for doc in documents:
            if not doc.personIds:
                continue

            # Add person nodes
            for pid in doc.personIds:
                self._add_node(pid, pid, "person")

            # Add co-occurrence edges between all person pairs in each document
            pids = sorted(set(doc.personIds))
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    self._add_edge(
                        pids[i],
                        pids[j],
                        "co-occurrence",
                        weight=1.0,
                        doc_id=doc.id,
                    )

    def add_flights(self, flights: list[Flight]) -> None:
        """Add co-passenger edges from flight logs."""
        for flight in flights:
            all_pax = sorted(set(flight.passengerIds + flight.pilotIds))
            for pid in all_pax:
                self._add_node(pid, pid, "person")

            for i in range(len(all_pax)):
                for j in range(i + 1, len(all_pax)):
                    self._add_edge(
                        all_pax[i],
                        all_pax[j],
                        "co-passenger",
                        weight=2.0,  # Flights are stronger signals
                        flight_id=flight.id,
                        date=flight.date,
                    )

    def add_emails(self, emails: list[Email]) -> None:
        """Add correspondence edges from emails."""
        for email in emails:
            if not email.personIds:
                continue

            pids = sorted(set(email.personIds))
            for pid in pids:
                self._add_node(pid, pid, "person")

            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    self._add_edge(
                        pids[i],
                        pids[j],
                        "correspondence",
                        weight=1.5,
                        email_id=email.id,
                    )

    def add_person_labels(self, persons: dict[str, str]) -> None:
        """Update node labels with actual person names.

        Parameters
        ----------
        persons:
            Mapping of person_id to display name.
        """
        for pid, name in persons.items():
            if pid in self._nodes:
                self._nodes[pid].label = name

    def build(self) -> KnowledgeGraph:
        """Build the final knowledge graph."""
        edges = []
        for (source, target, edge_type), weight in self._edge_counter.items():
            attrs = self._edge_attrs.get((source, target, edge_type), {})
            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    type=edge_type,
                    weight=weight,
                    attributes=attrs,
                )
            )

        # Sort edges by weight descending
        edges.sort(key=lambda e: e.weight, reverse=True)

        return KnowledgeGraph(
            nodes=list(self._nodes.values()),
            edges=edges,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_json(graph: KnowledgeGraph, path: Path) -> None:
        """Export as D3.js-compatible JSON (nodes + links)."""
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "type": n.type,
                    **n.attributes,
                }
                for n in graph.nodes
            ],
            "links": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "weight": e.weight,
                }
                for e in graph.edges
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def export_gexf(graph: KnowledgeGraph, path: Path) -> None:
        """Export as GEXF (Gephi graph format)."""
        gexf = Element(
            "gexf",
            {
                "xmlns": "http://www.gexf.net/1.3",
                "version": "1.3",
            },
        )
        graph_el = SubElement(
            gexf,
            "graph",
            {
                "defaultedgetype": "undirected",
                "mode": "static",
            },
        )

        # Attributes
        node_attrs = SubElement(graph_el, "attributes", {"class": "node"})
        SubElement(node_attrs, "attribute", {"id": "0", "title": "type", "type": "string"})

        edge_attrs = SubElement(graph_el, "attributes", {"class": "edge"})
        SubElement(edge_attrs, "attribute", {"id": "0", "title": "type", "type": "string"})

        # Nodes
        nodes_el = SubElement(graph_el, "nodes")
        for node in graph.nodes:
            node_el = SubElement(nodes_el, "node", {"id": node.id, "label": node.label})
            attvalues = SubElement(node_el, "attvalues")
            SubElement(attvalues, "attvalue", {"for": "0", "value": node.type})

        # Edges
        edges_el = SubElement(graph_el, "edges")
        for i, edge in enumerate(graph.edges):
            edge_el = SubElement(
                edges_el,
                "edge",
                {
                    "id": str(i),
                    "source": edge.source,
                    "target": edge.target,
                    "weight": str(edge.weight),
                },
            )
            attvalues = SubElement(edge_el, "attvalues")
            SubElement(attvalues, "attvalue", {"for": "0", "value": edge.type})

        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ElementTree(gexf)
        tree.write(str(path), encoding="unicode", xml_declaration=True)
