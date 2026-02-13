"""Incremental processing state tracker backed by SQLite.

Tracks which files have been processed through which pipeline stages,
enabling resumable batch processing without re-doing completed work.

State database is stored at ``.cache/pipeline_state.db`` by default.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

STAGES = ("ocr", "entities", "redaction", "images", "transcription", "dedup", "graph")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processing_state (
    file_hash   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    result_path TEXT,
    completed_at REAL NOT NULL,
    PRIMARY KEY (file_hash, stage)
);

CREATE INDEX IF NOT EXISTS idx_state_stage ON processing_state(stage);
CREATE INDEX IF NOT EXISTS idx_state_hash  ON processing_state(file_hash);
"""


class ProcessingState:
    """Track which files have been processed through which stages.

    Uses a lightweight SQLite database for persistence across runs.

    Usage::

        state = ProcessingState(Path(".cache/pipeline_state.db"))
        if not state.is_processed(file_hash, "ocr"):
            result = process(file)
            state.mark_processed(file_hash, "ocr", str(result_path))
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or Path(".cache/pipeline_state.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def is_processed(self, file_hash: str, stage: str) -> bool:
        """Check if a file has been processed through a given stage."""
        row = self._conn.execute(
            "SELECT 1 FROM processing_state WHERE file_hash = ? AND stage = ?",
            (file_hash, stage),
        ).fetchone()
        return row is not None

    def mark_processed(
        self,
        file_hash: str,
        stage: str,
        result_path: str | None = None,
    ) -> None:
        """Record that a file has been processed through a stage."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO processing_state (file_hash, stage, result_path, completed_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_hash, stage, result_path, time.time()),
        )
        self._conn.commit()

    def get_result_path(self, file_hash: str, stage: str) -> str | None:
        """Get the result path for a processed file."""
        row = self._conn.execute(
            "SELECT result_path FROM processing_state WHERE file_hash = ? AND stage = ?",
            (file_hash, stage),
        ).fetchone()
        return row[0] if row else None

    def get_unprocessed(
        self,
        file_hashes: list[str],
        stage: str,
    ) -> list[str]:
        """Return hashes from the input list that have NOT been processed.

        This is the primary API for batch processing -- compute hashes for
        all input files, call this method, and only process the unprocessed ones.
        """
        if not file_hashes:
            return []

        processed = set()
        # Query in batches of 500 to avoid SQLite variable limits
        for i in range(0, len(file_hashes), 500):
            batch = file_hashes[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                "SELECT file_hash FROM processing_state"
                f" WHERE stage = ? AND file_hash IN ({placeholders})",
                [stage] + batch,
            ).fetchall()
            processed.update(row[0] for row in rows)

        return [h for h in file_hashes if h not in processed]

    def get_stats(self) -> dict[str, int]:
        """Get counts of processed items per stage."""
        rows = self._conn.execute(
            "SELECT stage, COUNT(*) FROM processing_state GROUP BY stage"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def clear_stage(self, stage: str) -> int:
        """Clear all records for a specific stage. Returns count deleted."""
        cursor = self._conn.execute("DELETE FROM processing_state WHERE stage = ?", (stage,))
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
