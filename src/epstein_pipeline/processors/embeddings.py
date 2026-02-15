"""Embedding processor using sentence-transformers.

Chunks documents and generates vector embeddings for semantic search.
Supports GPU and CPU inference with batch processing.
Output formats: NDJSON, SQLite, or Neon Postgres (pgvector).
"""

from __future__ import annotations

import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

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
from epstein_pipeline.models.document import Document
from epstein_pipeline.processors.chunker import Chunker, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a single document."""

    document_id: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    processing_time_ms: int = 0
    model_name: str = ""


class EmbeddingProcessor:
    """Generate embeddings for document chunks.

    Parameters
    ----------
    settings : Settings
        Pipeline settings.
    model_name : str | None
        HuggingFace model ID. Default: nomic-ai/nomic-embed-text-v2-moe.
    dimensions : int
        Output embedding dimensions. 768 (full) or 256 (Matryoshka).
    batch_size : int | None
        Chunks per batch. None for auto-detect (64 GPU, 16 CPU).
    device : str | None
        Force device ("cuda", "cpu", "mps"). None for auto-detect.
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        dimensions: int = 768,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings
        self.model_name = model_name or settings.embedding_model
        self.dimensions = dimensions
        self.device = device
        self._model = None  # lazy load

        # Use semantic chunker by default
        self._chunker = Chunker(
            chunk_size=settings.embedding_chunk_size,
            overlap=settings.embedding_chunk_overlap,
            mode=settings.chunker_mode,
            target_tokens=settings.chunker_target_tokens,
            min_tokens=settings.chunker_min_tokens,
            max_tokens=settings.chunker_max_tokens,
        )

        if batch_size is None:
            self.batch_size = 64 if self._has_gpu() else 16
        else:
            self.batch_size = batch_size

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install epstein-pipeline[embeddings]"
            )

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )

        # Apply Matryoshka truncation if needed
        model_dim = self._model.get_sentence_embedding_dimension()
        if self.dimensions < model_dim:
            self._model.truncate_dim = self.dimensions
            logger.info(
                "Truncating embeddings from %d to %d dims",
                model_dim,
                self.dimensions,
            )

        return self._model

    def _has_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def embed_document(self, doc: Document) -> EmbeddingResult:
        """Chunk and embed a single document."""
        start_ms = time.monotonic_ns() // 1_000_000

        parts = []
        if doc.ocrText:
            parts.append(doc.ocrText)
        elif doc.summary:
            parts.append(doc.summary)

        text = "\n\n".join(parts)
        if not text.strip():
            return EmbeddingResult(
                document_id=doc.id,
                model_name=self.model_name,
            )

        chunks = self._chunker.chunk_document(
            doc.id,
            text,
            prepend_title=doc.title,
        )

        if not chunks:
            return EmbeddingResult(
                document_id=doc.id,
                model_name=self.model_name,
            )

        texts = [c.chunk_text for c in chunks]
        embeddings = self.embed_texts(texts)

        elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
        return EmbeddingResult(
            document_id=doc.id,
            chunks=chunks,
            embeddings=embeddings,
            processing_time_ms=elapsed,
            model_name=self.model_name,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of raw text strings."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if embeddings.shape[1] > self.dimensions:
            embeddings = embeddings[:, : self.dimensions]
        return embeddings.tolist()

    def process_batch(
        self,
        documents: list[Document],
        output_dir: Path,
        *,
        fmt: str = "ndjson",
    ) -> list[EmbeddingResult]:
        """Process documents: chunk, embed, and write output.

        Parameters
        ----------
        documents : list[Document]
            Documents to process.
        output_dir : Path
            Output directory for NDJSON/SQLite files.
        fmt : str
            Output format: "ndjson", "sqlite", or "neon".
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        embeddable = [d for d in documents if (d.ocrText and len(d.ocrText) > 50) or d.summary]

        if not embeddable:
            logger.warning("No documents with text to embed")
            return []

        logger.info(
            "Embedding %d documents (of %d total)",
            len(embeddable),
            len(documents),
        )

        # Chunk all documents first
        all_chunks: list[tuple[Document, list[DocumentChunk]]] = []
        for doc in embeddable:
            parts = []
            if doc.ocrText:
                parts.append(doc.ocrText)
            elif doc.summary:
                parts.append(doc.summary)
            text = "\n\n".join(parts)
            chunks = self._chunker.chunk_document(doc.id, text, prepend_title=doc.title)
            if chunks:
                all_chunks.append((doc, chunks))

        # Flatten chunks for batch embedding
        flat_texts: list[str] = []
        chunk_map: list[tuple[int, int]] = []
        for doc_idx, (_, chunks) in enumerate(all_chunks):
            for chunk_idx, chunk in enumerate(chunks):
                flat_texts.append(chunk.chunk_text)
                chunk_map.append((doc_idx, chunk_idx))

        logger.info(
            "Generated %d chunks from %d documents",
            len(flat_texts),
            len(all_chunks),
        )

        # Batch embed with progress bar
        all_embeddings: list[list[float]] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task("Embedding chunks", total=len(flat_texts))
            model = self._load_model()

            for i in range(0, len(flat_texts), self.batch_size):
                batch = flat_texts[i : i + self.batch_size]
                embs = model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                if embs.shape[1] > self.dimensions:
                    embs = embs[:, : self.dimensions]
                all_embeddings.extend(embs.tolist())
                progress.advance(task, len(batch))

        # Reassemble into results (optimized — no linear scan per chunk)
        # Build offset map: doc_idx -> starting flat index
        doc_offsets: list[int] = []
        offset = 0
        for _, chunks in all_chunks:
            doc_offsets.append(offset)
            offset += len(chunks)

        results: list[EmbeddingResult] = []
        for doc_idx, (doc, chunks) in enumerate(all_chunks):
            start = doc_offsets[doc_idx]
            doc_embeddings = all_embeddings[start : start + len(chunks)]
            results.append(
                EmbeddingResult(
                    document_id=doc.id,
                    chunks=chunks,
                    embeddings=doc_embeddings,
                    model_name=self.model_name,
                )
            )

        # Write output
        if fmt == "ndjson":
            out_path = output_dir / "embeddings.ndjson"
            self.write_ndjson(results, out_path)
        elif fmt == "sqlite":
            out_path = output_dir / "embeddings.db"
            self.write_sqlite(results, out_path)
        elif fmt == "neon":
            self.write_neon(results)

        return results

    def write_ndjson(self, results: list[EmbeddingResult], output_path: Path) -> None:
        """Write embedding results as NDJSON (one JSON per line)."""
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                for chunk, embedding in zip(result.chunks, result.embeddings):
                    line = json.dumps(
                        {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "chunk_text": chunk.chunk_text,
                            "embedding": embedding,
                        },
                        ensure_ascii=False,
                    )
                    f.write(line + "\n")
                    count += 1

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Wrote %d chunks to %s (%.1f MB)", count, output_path, size_mb)

    def write_sqlite(self, results: list[EmbeddingResult], db_path: Path) -> None:
        """Write embedding results to SQLite with BLOB columns."""
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT NOT NULL,
                embedding   BLOB,
                UNIQUE(document_id, chunk_index)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id)")

        count = 0
        for result in results:
            for chunk, embedding in zip(result.chunks, result.embeddings):
                blob = float_list_to_f32_blob(embedding)
                conn.execute(
                    "INSERT OR REPLACE INTO document_chunks"
                    " (document_id, chunk_index, chunk_text, embedding)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        blob,
                    ),
                )
                count += 1

        conn.commit()
        conn.close()
        logger.info("Wrote %d chunks to %s", count, db_path)

    def write_neon(self, results: list[EmbeddingResult]) -> None:
        """Write embedding results to Neon Postgres with pgvector.

        Requires EPSTEIN_NEON_DATABASE_URL to be set.
        """
        import asyncio

        database_url = self.settings.neon_database_url
        if not database_url:
            logger.error("NEON_DATABASE_URL not set — cannot write to Neon")
            return

        asyncio.run(self._write_neon_async(results, database_url))

    async def _write_neon_async(self, results: list[EmbeddingResult], database_url: str) -> None:
        """Async implementation of Neon embedding write."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector_async
        except ImportError:
            logger.error(
                "psycopg and pgvector required. Install with: pip install 'epstein-pipeline[neon]'"
            )
            return

        count = 0
        batch_size = self.settings.neon_batch_size

        async with await psycopg.AsyncConnection.connect(database_url) as conn:
            await register_vector_async(conn)

            for result in results:
                for i in range(0, len(result.chunks), batch_size):
                    batch_chunks = result.chunks[i : i + batch_size]
                    batch_embeds = result.embeddings[i : i + batch_size]

                    async with conn.cursor() as cur:
                        for chunk, embedding in zip(batch_chunks, batch_embeds):
                            await cur.execute(
                                """
                                INSERT INTO document_embeddings
                                    (document_id, chunk_index, chunk_text, embedding, model_name)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (document_id, chunk_index, model_name)
                                DO UPDATE SET
                                    chunk_text = EXCLUDED.chunk_text,
                                    embedding = EXCLUDED.embedding
                                """,
                                (
                                    chunk.document_id,
                                    chunk.chunk_index,
                                    chunk.chunk_text,
                                    embedding,
                                    result.model_name,
                                ),
                            )
                            count += 1

                    await conn.commit()

        logger.info("Wrote %d embedding chunks to Neon Postgres", count)


def float_list_to_f32_blob(values: list[float]) -> bytes:
    """Convert a list of floats to a packed F32 binary blob.

    This is the format expected by libSQL/Turso's F32_BLOB column type.
    """
    return struct.pack(f"{len(values)}f", *values)
