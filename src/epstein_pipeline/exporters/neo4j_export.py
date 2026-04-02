"""Neo4j knowledge graph exporter.

Exports the KnowledgeGraph (nodes + edges) from KnowledgeGraphBuilder
to a Neo4j database using batch MERGE operations for idempotency.

Uses the official neo4j Python driver (async API).

Usage:
    exporter = Neo4jExporter(settings)
    await exporter.export_graph(graph)
    await exporter.close()
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from rich.console import Console

from epstein_pipeline.config import Settings
from epstein_pipeline.processors.knowledge_graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)

logger = logging.getLogger(__name__)

# ── Optional dependency ──────────────────────────────────────────────────

try:
    from neo4j import AsyncGraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


def _require_neo4j() -> None:
    """Raise ImportError with install instructions if neo4j is missing."""
    if not HAS_NEO4J:
        raise ImportError(
            "neo4j is required for Neo4j export. "
            "Install with: pip install 'epstein-pipeline[neo4j]'"
        )


# ── Node type to Neo4j label mapping ────────────────────────────────────

_NODE_LABEL_MAP: dict[str, str] = {
    "person": "Person",
    "org": "Organization",
    "location": "Location",
    "document": "Document",
}


def _neo4j_label(node_type: str) -> str:
    """Map a GraphNode.type to a Neo4j node label."""
    return _NODE_LABEL_MAP.get(node_type.lower(), "Entity")


def _neo4j_rel_type(edge_type: str) -> str:
    """Map a GraphEdge.type to a Neo4j relationship type."""
    return edge_type.upper().replace("-", "_").replace(" ", "_")


# ── Batch helper ─────────────────────────────────────────────────────────


def _batches(items: list, size: int):  # noqa: ANN201
    """Yield successive batches of *size* from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Retry helper (mirrors neon_export pattern) ──────────────────────────

_RETRIABLE_KEYWORDS = (
    "connection",
    "timed out",
    "timeout",
    "service unavailable",
    "session expired",
)


async def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):  # noqa: ANN001, ANN201
    """Execute *fn* with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            msg = str(e).lower()
            retriable = any(kw in msg for kw in _RETRIABLE_KEYWORDS)
            if retriable and attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
                logger.warning(
                    "Retriable Neo4j error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            raise


# ── Exporter class ───────────────────────────────────────────────────────


class Neo4jExporter:
    """Export a KnowledgeGraph to Neo4j using batch MERGE operations.

    Parameters
    ----------
    settings : Settings
        Pipeline settings (must include neo4j_uri and neo4j_password).
    """

    def __init__(self, settings: Settings) -> None:
        _require_neo4j()

        if not settings.neo4j_uri:
            raise ValueError(
                "EPSTEIN_NEO4J_URI must be set for Neo4j export. "
                "Example: bolt://localhost:7687 or neo4j+s://xxx.databases.neo4j.io"
            )
        if not settings.neo4j_password:
            raise ValueError("EPSTEIN_NEO4J_PASSWORD must be set for Neo4j export.")

        self.settings = settings
        self.batch_size = settings.neo4j_batch_size
        self.retry_max = settings.neo4j_retry_max
        self.retry_base_delay = settings.neo4j_retry_base_delay
        self._driver = None
        self._console = Console()

    async def _get_driver(self):  # noqa: ANN201
        """Lazy-init the Neo4j async driver."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
            )
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self.settings.neo4j_uri)
        return self._driver

    async def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j driver")

    async def __aenter__(self) -> Neo4jExporter:
        await self._get_driver()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Schema setup ─────────────────────────────────────────────────────

    async def ensure_schema(self) -> None:
        """Create uniqueness constraints and indexes (idempotent)."""
        driver = await self._get_driver()
        constraints = [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT organization_id IF NOT EXISTS "
            "FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS "
            "FOR (l:Location) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]
        async with driver.session(database=self.settings.neo4j_database) as session:
            for stmt in constraints:
                await session.run(stmt)
        logger.info("Neo4j schema constraints ensured")

    # ── MERGE nodes ──────────────────────────────────────────────────────

    async def merge_nodes(self, nodes: list[GraphNode]) -> int:
        """Batch MERGE nodes into Neo4j, grouped by label.

        Returns the number of nodes merged.
        """
        if not nodes:
            return 0

        driver = await self._get_driver()
        total = 0

        # Group nodes by Neo4j label for efficient UNWIND
        by_label: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            label = _neo4j_label(node.type)
            props: dict[str, Any] = {
                "id": node.id,
                "label": node.label,
                "type": node.type,
            }
            # Only include serializable scalar attributes
            for k, v in node.attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    props[k] = v
            by_label.setdefault(label, []).append(props)

        for label, node_props_list in by_label.items():
            for batch in _batches(node_props_list, self.batch_size):
                query = (
                    f"UNWIND $nodes AS props "
                    f"MERGE (n:{label} {{id: props.id}}) "
                    f"SET n += props"
                )

                async def _merge(query=query, batch=batch):  # noqa: ANN001, ANN202
                    async with driver.session(
                        database=self.settings.neo4j_database
                    ) as session:
                        result = await session.run(query, nodes=batch)
                        await result.consume()

                await _with_retry(_merge, self.retry_max, self.retry_base_delay)
                total += len(batch)

        self._console.print(f"  [dim]Merged {total:,} nodes[/dim]")
        return total

    # ── MERGE edges ──────────────────────────────────────────────────────

    async def merge_edges(self, edges: list[GraphEdge]) -> int:
        """Batch MERGE edges into Neo4j.

        Returns the number of edges merged.
        """
        if not edges:
            return 0

        driver = await self._get_driver()
        total = 0

        # Group edges by relationship type for efficient UNWIND
        by_type: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            rel_type = _neo4j_rel_type(edge.type)
            props: dict[str, Any] = {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "type": edge.type,
            }
            for k, v in edge.attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    props[k] = v
            by_type.setdefault(rel_type, []).append(props)

        for rel_type, edge_props_list in by_type.items():
            for batch in _batches(edge_props_list, self.batch_size):
                query = (
                    f"UNWIND $edges AS props "
                    f"MATCH (a {{id: props.source}}) "
                    f"MATCH (b {{id: props.target}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"SET r.weight = props.weight, r.type = props.type"
                )

                async def _merge(query=query, batch=batch):  # noqa: ANN001, ANN202
                    async with driver.session(
                        database=self.settings.neo4j_database
                    ) as session:
                        result = await session.run(query, edges=batch)
                        await result.consume()

                await _with_retry(_merge, self.retry_max, self.retry_base_delay)
                total += len(batch)

        self._console.print(f"  [dim]Merged {total:,} edges[/dim]")
        return total

    # ── Full export ──────────────────────────────────────────────────────

    async def export_graph(self, graph: KnowledgeGraph) -> dict[str, int]:
        """Export a complete KnowledgeGraph to Neo4j.

        Creates schema, merges nodes, then merges edges.

        Returns
        -------
        dict[str, int]
            Counts: {"nodes": N, "edges": M}
        """
        self._console.print("\n[bold]Exporting knowledge graph to Neo4j[/bold]")
        self._console.print(f"  [dim]URI: {self.settings.neo4j_uri}[/dim]")
        self._console.print(
            f"  [dim]Graph: {graph.node_count} nodes, {graph.edge_count} edges[/dim]\n"
        )

        await self.ensure_schema()

        counts = {
            "nodes": await self.merge_nodes(graph.nodes),
            "edges": await self.merge_edges(graph.edges),
        }

        self._console.print("\n[bold green]Neo4j export complete[/bold green]")
        for key, count in counts.items():
            self._console.print(f"  {key}: {count:,}")

        return counts

    # ── Query helpers ────────────────────────────────────────────────────

    async def get_node_count(self) -> int:
        """Return the total number of nodes in the database."""
        driver = await self._get_driver()
        async with driver.session(database=self.settings.neo4j_database) as session:
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            record = await result.single()
            return record["cnt"] if record else 0

    async def get_edge_count(self) -> int:
        """Return the total number of relationships in the database."""
        driver = await self._get_driver()
        async with driver.session(database=self.settings.neo4j_database) as session:
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = await result.single()
            return record["cnt"] if record else 0

    async def clear_all(self) -> None:
        """Delete all nodes and relationships. USE WITH CAUTION."""
        driver = await self._get_driver()
        async with driver.session(database=self.settings.neo4j_database) as session:
            await session.run("MATCH (n) DETACH DELETE n")
        self._console.print("[yellow]All Neo4j data deleted[/yellow]")


# ── Synchronous wrapper for CLI use ──────────────────────────────────────


def export_graph_to_neo4j_sync(
    settings: Settings,
    graph: KnowledgeGraph,
) -> dict[str, int]:
    """Synchronous wrapper for Neo4jExporter.export_graph (for CLI use)."""

    async def _run() -> dict[str, int]:
        async with Neo4jExporter(settings) as exporter:
            return await exporter.export_graph(graph)

    return asyncio.run(_run())
