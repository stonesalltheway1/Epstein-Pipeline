"""Neon Postgres exporter for the Epstein Pipeline.

Exports documents, persons, emails, flights, entities, embeddings,
duplicate clusters, and relationships to Neon Postgres with pgvector
for semantic search.

Uses psycopg v3 async API with connection pooling and batched upserts
(ON CONFLICT DO UPDATE) for idempotent re-runs.

Usage:
    exporter = NeonExporter(settings)
    await exporter.export_all(documents, persons, emails, flights, ...)
    results = await exporter.semantic_search(query_embedding, top_k=10)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import (
    Document,
    Email,
    EntityResult,
    Flight,
    Person,
)

logger = logging.getLogger(__name__)

# ── Optional dependency imports ───────────────────────────────────────────────

try:
    import psycopg  # noqa: F401 — used for availability check
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

try:
    from pgvector.psycopg import register_vector_async

    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


def _require_psycopg() -> None:
    """Raise a helpful error if psycopg is not installed."""
    if not HAS_PSYCOPG:
        raise ImportError(
            "psycopg and psycopg_pool are required for Neon export. "
            "Install with: pip install 'epstein-pipeline[neon]'"
        )


def _require_pgvector() -> None:
    """Raise a helpful error if pgvector is not installed."""
    if not HAS_PGVECTOR:
        raise ImportError(
            "pgvector is required for embedding export. "
            "Install with: pip install 'epstein-pipeline[neon]'"
        )


# ── Progress bar factory ─────────────────────────────────────────────────────


def _make_progress() -> Progress:
    """Create a Rich progress bar for batch operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


# ── Semantic search result ────────────────────────────────────────────────────


@dataclass
class SemanticSearchResult:
    """A single result from a pgvector semantic search."""

    document_id: str
    chunk_text: str
    chunk_index: int
    similarity: float
    title: str | None = None


# ── Batch helper ──────────────────────────────────────────────────────────────


def _batches(items: Sequence[Any], size: int):
    """Yield successive batches from a sequence."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Main exporter class ──────────────────────────────────────────────────────


class NeonExporter:
    """Export pipeline data to Neon Postgres with pgvector support.

    Parameters
    ----------
    settings : Settings
        Pipeline settings (must include neon_database_url).
    """

    def __init__(self, settings: Settings) -> None:
        _require_psycopg()

        if not settings.neon_database_url:
            raise ValueError(
                "EPSTEIN_NEON_DATABASE_URL must be set for Neon export. "
                "Get a connection string from https://console.neon.tech"
            )

        self.settings = settings
        self.database_url = settings.neon_database_url
        self.pool_size = settings.neon_pool_size
        self.batch_size = settings.neon_batch_size
        self._pool: AsyncConnectionPool | None = None
        self._console = Console()

    # ── Connection pool management ────────────────────────────────────────

    async def _get_pool(self) -> AsyncConnectionPool:
        """Get or create the async connection pool."""
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                conninfo=self.database_url,
                min_size=1,
                max_size=self.pool_size,
                open=False,
            )
            await self._pool.open()
            logger.info("Opened Neon connection pool (max_size=%d)", self.pool_size)
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed Neon connection pool")

    async def __aenter__(self) -> NeonExporter:
        await self._get_pool()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Upsert: Documents ─────────────────────────────────────────────────

    async def upsert_documents(self, documents: list[Document]) -> int:
        """Upsert documents into the documents table.

        Returns the number of rows upserted.
        """
        if not documents:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting documents", total=len(documents))

            for batch in _batches(documents, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for doc in batch:
                            await cur.execute(
                                """
                                INSERT INTO documents (
                                    id, title, date, source, category, summary,
                                    tags, pdf_url, source_url, archive_url,
                                    page_count, bates_range, ocr_text,
                                    location_ids, verification_status
                                ) VALUES (
                                    %(id)s, %(title)s, %(date)s, %(source)s,
                                    %(category)s, %(summary)s, %(tags)s,
                                    %(pdf_url)s, %(source_url)s, %(archive_url)s,
                                    %(page_count)s, %(bates_range)s, %(ocr_text)s,
                                    %(location_ids)s, %(verification_status)s
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    title = EXCLUDED.title,
                                    date = EXCLUDED.date,
                                    source = EXCLUDED.source,
                                    category = EXCLUDED.category,
                                    summary = EXCLUDED.summary,
                                    tags = EXCLUDED.tags,
                                    pdf_url = EXCLUDED.pdf_url,
                                    source_url = EXCLUDED.source_url,
                                    archive_url = EXCLUDED.archive_url,
                                    page_count = EXCLUDED.page_count,
                                    bates_range = EXCLUDED.bates_range,
                                    ocr_text = EXCLUDED.ocr_text,
                                    location_ids = EXCLUDED.location_ids,
                                    verification_status = EXCLUDED.verification_status
                                """,
                                {
                                    "id": doc.id,
                                    "title": doc.title,
                                    "date": doc.date,
                                    "source": doc.source,
                                    "category": doc.category,
                                    "summary": doc.summary,
                                    "tags": doc.tags or [],
                                    "pdf_url": doc.pdfUrl,
                                    "source_url": doc.sourceUrl,
                                    "archive_url": doc.archiveUrl,
                                    "page_count": doc.pageCount,
                                    "bates_range": doc.batesRange,
                                    "ocr_text": doc.ocrText,
                                    "location_ids": doc.locationIds or [],
                                    "verification_status": doc.verificationStatus,
                                },
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} documents[/dim]")
        return total

    # ── Upsert: Document hashes (integrity) ───────────────────────────────

    async def upsert_document_hashes(
        self,
        hashes: list[dict[str, Any]],
    ) -> int:
        """Upsert document hashes into the document_hashes table.

        Each hash dict should have: doc_id, sha256, dataset, file_path (optional),
        file_size (optional).

        Returns the number of rows upserted.
        """
        if not hashes:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting document hashes", total=len(hashes))

            for batch in _batches(hashes, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for h in batch:
                            await cur.execute(
                                """
                                INSERT INTO document_hashes (
                                    doc_id, sha256, dataset, file_path, file_size
                                ) VALUES (
                                    %(doc_id)s, %(sha256)s, %(dataset)s,
                                    %(file_path)s, %(file_size)s
                                )
                                ON CONFLICT (doc_id) DO UPDATE SET
                                    sha256 = EXCLUDED.sha256,
                                    dataset = EXCLUDED.dataset,
                                    file_path = EXCLUDED.file_path,
                                    file_size = EXCLUDED.file_size
                                """,
                                {
                                    "doc_id": h["doc_id"],
                                    "sha256": h["sha256"],
                                    "dataset": h.get("dataset", "unknown"),
                                    "file_path": h.get("file_path"),
                                    "file_size": h.get("file_size"),
                                },
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} document hashes[/dim]")
        return total

    async def insert_document_change(
        self,
        doc_id: str,
        change_type: str,
        detected_by: str,
        *,
        dataset: str | None = None,
        old_sha256: str | None = None,
        new_sha256: str | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Insert a single document change event."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO document_changes (
                        doc_id, change_type, dataset, detected_by,
                        old_sha256, new_sha256, http_status, details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        doc_id,
                        change_type,
                        dataset,
                        detected_by,
                        old_sha256,
                        new_sha256,
                        http_status,
                        __import__("json").dumps(details or {}),
                    ),
                )
            await conn.commit()

    # ── Upsert: Document-Person links ─────────────────────────────────────

    async def _upsert_document_persons(self, documents: list[Document]) -> int:
        """Upsert document-person join table rows.

        Returns the number of links upserted.
        """
        links = [(doc.id, pid) for doc in documents for pid in doc.personIds]
        if not links:
            return 0

        pool = await self._get_pool()
        total = 0

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for batch in _batches(links, self.batch_size):
                    for doc_id, person_id in batch:
                        await cur.execute(
                            """
                            INSERT INTO document_persons (document_id, person_id)
                            VALUES (%s, %s)
                            ON CONFLICT (document_id, person_id) DO NOTHING
                            """,
                            (doc_id, person_id),
                        )
                    total += len(batch)
            await conn.commit()

        self._console.print(f"  [dim]Upserted {total:,} document-person links[/dim]")
        return total

    # ── Upsert: Persons ───────────────────────────────────────────────────

    async def upsert_persons(self, persons: list[Person]) -> int:
        """Upsert persons into the persons table.

        Returns the number of rows upserted.
        """
        if not persons:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting persons", total=len(persons))

            for batch in _batches(persons, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for person in batch:
                            await cur.execute(
                                """
                                INSERT INTO persons (
                                    id, slug, name, aliases, category, short_bio
                                ) VALUES (
                                    %(id)s, %(slug)s, %(name)s, %(aliases)s,
                                    %(category)s, %(short_bio)s
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    slug = EXCLUDED.slug,
                                    name = EXCLUDED.name,
                                    aliases = EXCLUDED.aliases,
                                    category = EXCLUDED.category,
                                    short_bio = EXCLUDED.short_bio
                                """,
                                {
                                    "id": person.id,
                                    "slug": person.slug,
                                    "name": person.name,
                                    "aliases": person.aliases or [],
                                    "category": person.category,
                                    "short_bio": person.shortBio,
                                },
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} persons[/dim]")
        return total

    # ── Upsert: Emails ────────────────────────────────────────────────────

    async def upsert_emails(self, emails: list[Email]) -> int:
        """Upsert emails, recipients, and email-person links.

        Returns the number of emails upserted.
        """
        if not emails:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting emails", total=len(emails))

            for batch in _batches(emails, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for email in batch:
                            # Upsert the email itself
                            await cur.execute(
                                """
                                INSERT INTO emails (
                                    id, subject, from_name, from_email,
                                    from_person_slug, date, body, folder
                                ) VALUES (
                                    %(id)s, %(subject)s, %(from_name)s,
                                    %(from_email)s, %(from_person_slug)s,
                                    %(date)s, %(body)s, %(folder)s
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    subject = EXCLUDED.subject,
                                    from_name = EXCLUDED.from_name,
                                    from_email = EXCLUDED.from_email,
                                    from_person_slug = EXCLUDED.from_person_slug,
                                    date = EXCLUDED.date,
                                    body = EXCLUDED.body,
                                    folder = EXCLUDED.folder
                                """,
                                {
                                    "id": email.id,
                                    "subject": email.subject,
                                    "from_name": email.from_.name,
                                    "from_email": email.from_.email,
                                    "from_person_slug": email.from_.personSlug,
                                    "date": email.date,
                                    "body": email.body,
                                    "folder": email.folder,
                                },
                            )

                            # Delete old recipients and re-insert
                            await cur.execute(
                                "DELETE FROM email_recipients WHERE email_id = %s",
                                (email.id,),
                            )

                            # Insert "to" recipients
                            for recip in email.to:
                                await cur.execute(
                                    """
                                    INSERT INTO email_recipients (
                                        email_id, recipient_type, name,
                                        email, person_slug
                                    ) VALUES (%s, 'to', %s, %s, %s)
                                    """,
                                    (
                                        email.id,
                                        recip.name,
                                        recip.email,
                                        recip.personSlug,
                                    ),
                                )

                            # Insert "cc" recipients
                            for recip in email.cc:
                                await cur.execute(
                                    """
                                    INSERT INTO email_recipients (
                                        email_id, recipient_type, name,
                                        email, person_slug
                                    ) VALUES (%s, 'cc', %s, %s, %s)
                                    """,
                                    (
                                        email.id,
                                        recip.name,
                                        recip.email,
                                        recip.personSlug,
                                    ),
                                )

                            # Upsert email-person links
                            for pid in email.personIds:
                                await cur.execute(
                                    """
                                    INSERT INTO email_persons (email_id, person_id)
                                    VALUES (%s, %s)
                                    ON CONFLICT (email_id, person_id) DO NOTHING
                                    """,
                                    (email.id, pid),
                                )

                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} emails[/dim]")
        return total

    # ── Upsert: Flights ───────────────────────────────────────────────────

    async def upsert_flights(self, flights: list[Flight]) -> int:
        """Upsert flights and passenger/pilot links.

        Returns the number of flights upserted.
        """
        if not flights:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting flights", total=len(flights))

            for batch in _batches(flights, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for flight in batch:
                            await cur.execute(
                                """
                                INSERT INTO flights (
                                    id, date, aircraft, tail_number,
                                    origin, destination
                                ) VALUES (
                                    %(id)s, %(date)s, %(aircraft)s,
                                    %(tail_number)s, %(origin)s, %(destination)s
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    date = EXCLUDED.date,
                                    aircraft = EXCLUDED.aircraft,
                                    tail_number = EXCLUDED.tail_number,
                                    origin = EXCLUDED.origin,
                                    destination = EXCLUDED.destination
                                """,
                                {
                                    "id": flight.id,
                                    "date": flight.date,
                                    "aircraft": flight.aircraft,
                                    "tail_number": flight.tailNumber,
                                    "origin": flight.origin,
                                    "destination": flight.destination,
                                },
                            )

                            # Delete old passenger/pilot links and re-insert
                            await cur.execute(
                                "DELETE FROM flight_passengers WHERE flight_id = %s",
                                (flight.id,),
                            )

                            for pid in flight.passengerIds:
                                await cur.execute(
                                    """
                                    INSERT INTO flight_passengers
                                        (flight_id, person_id, role)
                                    VALUES (%s, %s, 'passenger')
                                    ON CONFLICT (flight_id, person_id, role)
                                        DO NOTHING
                                    """,
                                    (flight.id, pid),
                                )

                            for pid in flight.pilotIds:
                                await cur.execute(
                                    """
                                    INSERT INTO flight_passengers
                                        (flight_id, person_id, role)
                                    VALUES (%s, %s, 'pilot')
                                    ON CONFLICT (flight_id, person_id, role)
                                        DO NOTHING
                                    """,
                                    (flight.id, pid),
                                )

                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} flights[/dim]")
        return total

    # ── Upsert: Entities ──────────────────────────────────────────────────

    async def upsert_entities(
        self,
        entities: dict[str, list[EntityResult]],
    ) -> int:
        """Upsert NER entities keyed by document_id.

        Parameters
        ----------
        entities : dict[str, list[EntityResult]]
            Mapping of document_id -> list of extracted entities.

        Returns the total number of entity rows upserted.
        """
        if not entities:
            return 0

        # Flatten to (doc_id, entity) tuples
        flat: list[tuple[str, EntityResult]] = [
            (doc_id, ent) for doc_id, ents in entities.items() for ent in ents
        ]

        if not flat:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting entities", total=len(flat))

            for batch in _batches(flat, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for doc_id, ent in batch:
                            # Delete existing entities for this doc+type+value
                            # to avoid duplicates, then insert fresh
                            await cur.execute(
                                """
                                INSERT INTO entities (
                                    document_id, entity_type, entity_value,
                                    confidence, entity_source, source_span
                                ) VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    doc_id,
                                    ent.entity_type,
                                    ent.value,
                                    ent.confidence,
                                    ent.source,
                                    ent.span,
                                ),
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} entities[/dim]")
        return total

    # ── Upsert: Embeddings (pgvector) ─────────────────────────────────────

    async def upsert_embeddings(
        self,
        embedding_results: list[Any],
    ) -> int:
        """Upsert document chunk embeddings into document_embeddings.

        Parameters
        ----------
        embedding_results : list[EmbeddingResult]
            Results from the EmbeddingProcessor. Each contains chunks
            and their corresponding vector embeddings.

        Returns the total number of embedding rows upserted.
        """
        if not embedding_results:
            return 0

        _require_pgvector()

        # Flatten to individual (chunk, embedding, model_name) tuples
        flat: list[tuple[str, int, str, list[float], str]] = []
        for result in embedding_results:
            for chunk, embedding in zip(result.chunks, result.embeddings):
                flat.append(
                    (
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        embedding,
                        result.model_name,
                    )
                )

        if not flat:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting embeddings", total=len(flat))

            for batch in _batches(flat, self.batch_size):
                async with pool.connection() as conn:
                    # Register pgvector type for this connection
                    await register_vector_async(conn)

                    async with conn.cursor() as cur:
                        for doc_id, chunk_idx, chunk_text, embedding, model in batch:
                            await cur.execute(
                                """
                                INSERT INTO document_embeddings (
                                    document_id, chunk_index, chunk_text,
                                    embedding, model_name
                                ) VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (document_id, chunk_index, model_name)
                                DO UPDATE SET
                                    chunk_text = EXCLUDED.chunk_text,
                                    embedding = EXCLUDED.embedding
                                """,
                                (doc_id, chunk_idx, chunk_text, embedding, model),
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} embeddings[/dim]")
        return total

    # ── Upsert: Duplicate clusters ────────────────────────────────────────

    async def upsert_duplicate_clusters(
        self,
        clusters: list[dict[str, Any]],
    ) -> int:
        """Upsert duplicate cluster records.

        Parameters
        ----------
        clusters : list[dict]
            Each dict should have keys: cluster_id, document_id,
            is_representative, similarity, dedup_method.

        Returns the total number of rows upserted.
        """
        if not clusters:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting duplicate clusters", total=len(clusters))

            for batch in _batches(clusters, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for cluster in batch:
                            await cur.execute(
                                """
                                INSERT INTO duplicate_clusters (
                                    cluster_id, document_id, is_representative,
                                    similarity, dedup_method
                                ) VALUES (
                                    %(cluster_id)s, %(document_id)s,
                                    %(is_representative)s, %(similarity)s,
                                    %(dedup_method)s
                                )
                                """,
                                {
                                    "cluster_id": cluster["cluster_id"],
                                    "document_id": cluster["document_id"],
                                    "is_representative": cluster.get("is_representative", False),
                                    "similarity": cluster.get("similarity", 1.0),
                                    "dedup_method": cluster.get("dedup_method", "exact"),
                                },
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} duplicate cluster entries[/dim]")
        return total

    # ── Upsert: Relationships ─────────────────────────────────────────────

    async def upsert_relationships(
        self,
        relationships: list[dict[str, Any]],
    ) -> int:
        """Upsert knowledge graph relationships.

        Parameters
        ----------
        relationships : list[dict]
            Each dict should have keys: person1_id, person2_id,
            relationship_type, weight, evidence_doc_id (optional),
            context_snippet (optional), extraction_method (optional).

        Returns the total number of rows upserted.
        """
        if not relationships:
            return 0

        pool = await self._get_pool()
        total = 0

        with _make_progress() as progress:
            task = progress.add_task("Upserting relationships", total=len(relationships))

            for batch in _batches(relationships, self.batch_size):
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for rel in batch:
                            await cur.execute(
                                """
                                INSERT INTO relationships (
                                    person1_id, person2_id, relationship_type,
                                    weight, evidence_doc_id, context_snippet,
                                    extraction_method
                                ) VALUES (
                                    %(person1_id)s, %(person2_id)s,
                                    %(relationship_type)s, %(weight)s,
                                    %(evidence_doc_id)s, %(context_snippet)s,
                                    %(extraction_method)s
                                )
                                ON CONFLICT (
                                    person1_id, person2_id,
                                    relationship_type, evidence_doc_id
                                ) DO UPDATE SET
                                    weight = EXCLUDED.weight,
                                    context_snippet = EXCLUDED.context_snippet,
                                    extraction_method = EXCLUDED.extraction_method
                                """,
                                {
                                    "person1_id": rel["person1_id"],
                                    "person2_id": rel["person2_id"],
                                    "relationship_type": rel["relationship_type"],
                                    "weight": rel.get("weight", 1.0),
                                    "evidence_doc_id": rel.get("evidence_doc_id"),
                                    "context_snippet": rel.get("context_snippet"),
                                    "extraction_method": rel.get("extraction_method"),
                                },
                            )
                    await conn.commit()

                total += len(batch)
                progress.advance(task, len(batch))

        self._console.print(f"  [dim]Upserted {total:,} relationships[/dim]")
        return total

    # ── Semantic search ───────────────────────────────────────────────────

    async def semantic_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> list[SemanticSearchResult]:
        """Perform semantic search using pgvector cosine similarity.

        Calls the `semantic_search()` SQL function defined in neon_schema.py,
        which uses the IVFFlat/HNSW index on document_embeddings.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector (must be 768 dimensions to match the index).
        top_k : int
            Maximum number of results to return.
        threshold : float
            Minimum cosine similarity threshold (0.0-1.0).

        Returns
        -------
        list[SemanticSearchResult]
            Ranked results with document_id, chunk_text, similarity, etc.
        """
        _require_pgvector()

        pool = await self._get_pool()

        async with pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        ss.document_id,
                        ss.chunk_text,
                        ss.chunk_index,
                        ss.similarity,
                        d.title
                    FROM semantic_search(
                        %s::vector(768), %s, %s
                    ) ss
                    LEFT JOIN documents d ON d.id = ss.document_id
                    ORDER BY ss.similarity DESC
                    """,
                    (query_embedding, threshold, top_k),
                )
                rows = await cur.fetchall()

        results = [
            SemanticSearchResult(
                document_id=row["document_id"],
                chunk_text=row["chunk_text"],
                chunk_index=row["chunk_index"],
                similarity=row["similarity"],
                title=row.get("title"),
            )
            for row in rows
        ]

        logger.info(
            "Semantic search returned %d results (threshold=%.2f)",
            len(results),
            threshold,
        )
        return results

    # ── Full export ───────────────────────────────────────────────────────

    async def export_all(
        self,
        *,
        documents: list[Document] | None = None,
        persons: list[Person] | None = None,
        emails: list[Email] | None = None,
        flights: list[Flight] | None = None,
        entities: dict[str, list[EntityResult]] | None = None,
        embedding_results: list[Any] | None = None,
        duplicate_clusters: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        document_hashes: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Export all pipeline data to Neon Postgres.

        Upserts data in the correct order to satisfy foreign key constraints:
        persons first, then documents, then join tables and dependent data.

        Parameters
        ----------
        documents : list[Document] | None
            Documents to upsert.
        persons : list[Person] | None
            Persons to upsert (must be done before documents for FK).
        emails : list[Email] | None
            Emails with recipients and person links.
        flights : list[Flight] | None
            Flights with passenger/pilot links.
        entities : dict[str, list[EntityResult]] | None
            NER entities keyed by document_id.
        embedding_results : list[EmbeddingResult] | None
            Chunk embeddings from the EmbeddingProcessor.
        duplicate_clusters : list[dict] | None
            Dedup cluster records.
        relationships : list[dict] | None
            Knowledge graph relationship records.

        Returns
        -------
        dict[str, int]
            Counts of rows upserted per table type.
        """
        self._console.print("\n[bold]Exporting to Neon Postgres[/bold]")
        self._console.print(f"  [dim]Database: {_mask_url(self.database_url)}[/dim]")
        self._console.print(f"  [dim]Batch size: {self.batch_size}[/dim]\n")

        counts: dict[str, int] = {}

        try:
            # 1. Persons first (other tables reference them)
            if persons:
                counts["persons"] = await self.upsert_persons(persons)

            # 2. Documents (referenced by entities, embeddings, etc.)
            if documents:
                counts["documents"] = await self.upsert_documents(documents)
                counts["document_persons"] = await self._upsert_document_persons(documents)

            # 3. Dependent data (requires documents and persons to exist)
            if emails:
                counts["emails"] = await self.upsert_emails(emails)

            if flights:
                counts["flights"] = await self.upsert_flights(flights)

            if entities:
                counts["entities"] = await self.upsert_entities(entities)

            if embedding_results:
                counts["embeddings"] = await self.upsert_embeddings(embedding_results)

            if duplicate_clusters:
                counts["duplicate_clusters"] = await self.upsert_duplicate_clusters(
                    duplicate_clusters
                )

            if relationships:
                counts["relationships"] = await self.upsert_relationships(relationships)

            # 4. Document integrity hashes
            if document_hashes:
                counts["document_hashes"] = await self.upsert_document_hashes(
                    document_hashes
                )

        except Exception:
            logger.exception("Error during Neon export")
            raise

        # Print summary
        self._console.print("\n[bold green]Neon export complete[/bold green]")
        for table, count in counts.items():
            self._console.print(f"  {table}: {count:,}")

        return counts

    # ── Utility: Get row counts ───────────────────────────────────────────

    async def get_table_counts(self) -> dict[str, int]:
        """Get row counts for all pipeline tables.

        Useful for verifying export results or monitoring data growth.
        """
        pool = await self._get_pool()
        tables = [
            "documents",
            "persons",
            "document_persons",
            "entities",
            "document_embeddings",
            "emails",
            "email_recipients",
            "email_persons",
            "flights",
            "flight_passengers",
            "duplicate_clusters",
            "relationships",
        ]

        counts: dict[str, int] = {}
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for table in tables:
                    try:
                        await cur.execute(f"SELECT count(*) FROM {table}")
                        row = await cur.fetchone()
                        counts[table] = row[0] if row else 0
                    except Exception:
                        counts[table] = -1
                        await conn.rollback()

        return counts

    # ── Utility: Clear all data ───────────────────────────────────────────

    async def truncate_all(self) -> None:
        """Truncate all pipeline tables. USE WITH CAUTION.

        This removes all data but preserves the schema. Useful for
        clean re-imports during development.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    TRUNCATE TABLE
                        duplicate_clusters,
                        relationships,
                        document_embeddings,
                        entities,
                        flight_passengers,
                        flights,
                        email_persons,
                        email_recipients,
                        emails,
                        document_persons,
                        documents,
                        persons
                    CASCADE
                    """
                )
            await conn.commit()

        self._console.print("[yellow]All pipeline tables truncated[/yellow]")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mask_url(url: str) -> str:
    """Mask the password in a database URL for safe logging."""
    # postgresql://user:password@host/db -> postgresql://user:***@host/db
    import re

    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


# ── Synchronous wrapper for CLI use ──────────────────────────────────────────


def export_to_neon_sync(
    settings: Settings,
    *,
    documents: list[Document] | None = None,
    persons: list[Person] | None = None,
    emails: list[Email] | None = None,
    flights: list[Flight] | None = None,
    entities: dict[str, list[EntityResult]] | None = None,
    embedding_results: list[Any] | None = None,
    duplicate_clusters: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Synchronous wrapper for NeonExporter.export_all (for CLI use)."""

    async def _run() -> dict[str, int]:
        async with NeonExporter(settings) as exporter:
            return await exporter.export_all(
                documents=documents,
                persons=persons,
                emails=emails,
                flights=flights,
                entities=entities,
                embedding_results=embedding_results,
                duplicate_clusters=duplicate_clusters,
                relationships=relationships,
            )

    return asyncio.run(_run())
