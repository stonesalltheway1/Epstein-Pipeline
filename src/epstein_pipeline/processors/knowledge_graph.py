"""Knowledge graph builder with co-occurrence and LLM-based relationship extraction.

Builds a weighted entity-relationship graph from pipeline output.
Nodes: persons, organizations, locations, documents.
Edges: co-occurrence, co-passengers, correspondence, financial, LLM-extracted typed relationships.
Export: GEXF (Gephi), JSON (D3.js-compatible), Neon Postgres.

Relationship types for LLM extraction:
FLEW_WITH, EMPLOYED_BY, ASSOCIATED_WITH, MENTIONED_IN,
PARTY_TO, WITNESS_IN, DEFENDANT_IN
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Document, Email, Flight

logger = logging.getLogger(__name__)

# Typed relationship labels for LLM extraction
RELATIONSHIP_TYPES = [
    "FLEW_WITH",
    "EMPLOYED_BY",
    "ASSOCIATED_WITH",
    "MENTIONED_IN",
    "PARTY_TO",
    "WITNESS_IN",
    "DEFENDANT_IN",
    "FINANCIAL_LINK",
    "FAMILY_MEMBER",
    "LEGAL_COUNSEL",
]


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
    type: str
    weight: float = 1.0
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    """A relationship extracted by LLM from document text."""

    person1: str
    person2: str
    relationship_type: str
    confidence: float
    evidence_snippet: str
    document_id: str


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
    """Build a knowledge graph from pipeline-processed data.

    Supports two modes:
    1. **Co-occurrence** (default) — edges from shared documents, flights, emails
    2. **LLM extraction** (opt-in) — typed relationships via OpenAI/Anthropic API
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self._nodes: dict[str, GraphNode] = {}
        self._edge_counter: dict[tuple[str, str, str], float] = Counter()
        self._edge_attrs: dict[tuple[str, str, str], dict] = defaultdict(dict)
        self._extracted_relationships: list[ExtractedRelationship] = []

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
        if source > target:
            source, target = target, source
        key = (source, target, edge_type)
        self._edge_counter[key] += weight
        self._edge_attrs[key].update(attrs)

    # ------------------------------------------------------------------
    # Co-occurrence build methods
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add document-based co-occurrence edges."""
        for doc in documents:
            if not doc.personIds:
                continue

            for pid in doc.personIds:
                self._add_node(pid, pid, "person")

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
                        weight=2.0,
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
        """Update node labels with actual person names."""
        for pid, name in persons.items():
            if pid in self._nodes:
                self._nodes[pid].label = name

    # ------------------------------------------------------------------
    # LLM-based relationship extraction (opt-in)
    # ------------------------------------------------------------------

    def extract_relationships_llm(
        self,
        documents: list[Document],
        person_names: dict[str, str] | None = None,
        max_documents: int = 100,
    ) -> list[ExtractedRelationship]:
        """Extract typed relationships from document text using an LLM.

        Only processes documents with at least 2 person IDs.
        Results are cached to avoid re-processing.

        Parameters
        ----------
        documents : list[Document]
            Documents to extract relationships from.
        person_names : dict[str, str] | None
            Map of person_id to name for context.
        max_documents : int
            Maximum documents to process (LLM calls are expensive).
        """
        if not self.settings or not self.settings.kg_extract_relationships:
            logger.info(
                "LLM relationship extraction disabled (set EPSTEIN_KG_EXTRACT_RELATIONSHIPS=true)"
            )
            return []

        # Filter to documents with multiple persons
        candidates = [d for d in documents if len(d.personIds) >= 2 and (d.ocrText or d.summary)]
        candidates = candidates[:max_documents]

        if not candidates:
            return []

        logger.info("Extracting relationships from %d documents via LLM...", len(candidates))

        for doc in candidates:
            try:
                rels = self._extract_from_document(doc, person_names or {})
                self._extracted_relationships.extend(rels)

                # Add extracted relationships as typed edges
                for rel in rels:
                    self._add_edge(
                        rel.person1,
                        rel.person2,
                        rel.relationship_type,
                        weight=rel.confidence * 3.0,  # LLM relationships weighted higher
                        evidence_doc_id=rel.document_id,
                        context_snippet=rel.evidence_snippet[:200],
                        extraction_method="llm",
                    )
            except Exception as exc:
                logger.warning("LLM extraction failed for %s: %s", doc.id, exc)

        logger.info("Extracted %d relationships", len(self._extracted_relationships))
        return self._extracted_relationships

    def _extract_from_document(
        self,
        doc: Document,
        person_names: dict[str, str],
    ) -> list[ExtractedRelationship]:
        """Extract relationships from a single document using the configured LLM."""
        text = doc.ocrText or doc.summary or ""
        if not text:
            return []

        # Truncate for LLM context window
        text = text[:4000]

        # Build person context
        persons_in_doc = [f"- {pid}: {person_names.get(pid, pid)}" for pid in doc.personIds]
        persons_context = "\n".join(persons_in_doc)

        rel_types = ", ".join(RELATIONSHIP_TYPES)
        prompt = (
            "Analyze this document excerpt and identify relationships "
            "between the people listed.\n\n"
            f"People mentioned:\n{persons_context}\n\n"
            f"Document text:\n{text}\n\n"
            "For each relationship found, provide:\n"
            "1. person1_id (from the list above)\n"
            "2. person2_id (from the list above)\n"
            f"3. relationship_type (one of: {rel_types})\n"
            "4. confidence (0.0-1.0)\n"
            "5. evidence (brief quote supporting this relationship)\n\n"
            "Return as JSON array. If no relationships found, return [].\n"
            'Example: [{"person1": "p-0001", "person2": "p-0002", '
            '"type": "FLEW_WITH", "confidence": 0.9, '
            '"evidence": "flew together on..."}]'
        )

        provider = self.settings.kg_llm_provider if self.settings else "openai"
        model = self.settings.kg_llm_model if self.settings else "gpt-4o-mini"

        try:
            response_text = self._call_llm(prompt, provider, model)
            return self._parse_llm_response(response_text, doc.id)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return []

    def _call_llm(self, prompt: str, provider: str, model: str) -> str:
        """Call the configured LLM provider."""
        if provider == "openai":
            try:
                from openai import OpenAI

                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                return response.choices[0].message.content or ""
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif provider == "anthropic":
            try:
                import anthropic

                client = anthropic.Anthropic()
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text if response.content else ""
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

        raise ValueError(f"Unknown LLM provider: {provider}")

    def _parse_llm_response(
        self, response_text: str, document_id: str
    ) -> list[ExtractedRelationship]:
        """Parse the LLM's JSON response into ExtractedRelationship objects."""
        try:
            # Find JSON array in response
            text = response_text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                return []

            data = json.loads(text[start:end])
            relationships = []

            for item in data:
                rel_type = item.get("type", "ASSOCIATED_WITH")
                if rel_type not in RELATIONSHIP_TYPES:
                    rel_type = "ASSOCIATED_WITH"

                relationships.append(
                    ExtractedRelationship(
                        person1=item.get("person1", ""),
                        person2=item.get("person2", ""),
                        relationship_type=rel_type,
                        confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                        evidence_snippet=item.get("evidence", ""),
                        document_id=document_id,
                    )
                )

            return relationships

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

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
