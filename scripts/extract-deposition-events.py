#!/usr/bin/env python3
"""Extract temporal events from all deposition transcripts in Neon.

Pulls segment text from deposition_segments, concatenates per deposition,
runs temporal extraction via gpt-4.1-mini, and upserts results back to
Neon's temporal_events table.

Usage:
    python scripts/extract-deposition-events.py
    python scripts/extract-deposition-events.py --dry-run
    python scripts/extract-deposition-events.py --limit 5
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ── Load env ─────────────────────────────────────────────────────────────

_raw_url = os.environ.get("EPSTEIN_NEON_DATABASE_URL", "")
if not _raw_url:
    # Try .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("EPSTEIN_NEON_DATABASE_URL="):
                _raw_url = line.split("=", 1)[1].strip()
                break

if not _raw_url:
    print("ERROR: EPSTEIN_NEON_DATABASE_URL not set")
    sys.exit(1)

DB_URL = re.sub(r"[&?]sslnegotiation=[^&]*", "", _raw_url)


def main():
    import argparse

    import psycopg
    from rich.console import Console

    from epstein_pipeline.config import Settings
    from epstein_pipeline.processors.temporal_extractor import TemporalExtractor

    parser = argparse.ArgumentParser(description="Extract temporal events from depositions")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write to DB")
    parser.add_argument("--limit", type=int, default=None, help="Max depositions to process")
    parser.add_argument("--min-words", type=int, default=50, help="Skip depositions under N words")
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini", help="OpenAI model to use",
    )
    parser.add_argument("--output-dir", type=str, default="output/events", help="Save JSON events")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold]Temporal Event Extraction from Depositions[/bold]\n")

    # ── Step 1: Inspect live schema ──────────────────────────────────────
    console.print("[dim]Inspecting live Neon schema...[/dim]")
    conn = psycopg.connect(DB_URL)
    cur = conn.cursor()

    # Check temporal_events table exists
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'temporal_events' ORDER BY ordinal_position
    """)
    te_cols = [r[0] for r in cur.fetchall()]
    if not te_cols:
        console.print("[yellow]temporal_events table not found. Creating it...[/yellow]")
        if not args.dry_run:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS temporal_events (
                    id              SERIAL PRIMARY KEY,
                    document_id     TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    date_raw        TEXT,
                    event_type      TEXT NOT NULL,
                    description     TEXT NOT NULL,
                    participants    TEXT[] NOT NULL DEFAULT '{}',
                    locations       TEXT[] NOT NULL DEFAULT '{}',
                    confidence      REAL NOT NULL DEFAULT 0.5,
                    source_chunk    TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (document_id, date, event_type, description)
                );
                CREATE INDEX IF NOT EXISTS idx_temporal_events_document
                    ON temporal_events (document_id);
                CREATE INDEX IF NOT EXISTS idx_temporal_events_date
                    ON temporal_events (date);
                CREATE INDEX IF NOT EXISTS idx_temporal_events_type
                    ON temporal_events (event_type);
                CREATE INDEX IF NOT EXISTS idx_temporal_events_participants
                    ON temporal_events USING gin (participants);
            """)
            conn.commit()
            console.print("[green]temporal_events table created[/green]")
        else:
            console.print("[yellow]DRY RUN: would create temporal_events table[/yellow]")
    else:
        console.print(f"  temporal_events columns: {te_cols}")

    # ── Step 2: Pull depositions ─────────────────────────────────────────
    console.print("\n[dim]Loading depositions from Neon...[/dim]")

    query = """
        SELECT vd.id, vd.title, vd.deposition_date,
               STRING_AGG(ds.text, ' ' ORDER BY ds.segment_index) AS full_text,
               COUNT(ds.segment_index) AS seg_count,
               SUM(LENGTH(ds.text)) AS char_count
        FROM video_depositions vd
        JOIN deposition_segments ds ON ds.deposition_id = vd.id
        GROUP BY vd.id, vd.title, vd.deposition_date
        HAVING SUM(LENGTH(ds.text)) > %s
        ORDER BY SUM(LENGTH(ds.text)) DESC
    """
    params = [args.min_words * 5]  # rough chars-per-word estimate

    if args.limit:
        query += " LIMIT %s"
        params.append(args.limit)

    cur.execute(query, params)
    depositions = cur.fetchall()
    console.print(f"  Found {len(depositions)} depositions with >{args.min_words} words\n")

    if not depositions:
        console.print("[yellow]No depositions to process.[/yellow]")
        cur.close()
        conn.close()
        return

    # ── Step 3: Run temporal extraction ──────────────────────────────────
    settings = Settings()
    settings.temporal_llm_provider = "openai"
    settings.temporal_llm_model = args.model
    settings.temporal_confidence_threshold = 0.3

    extractor = TemporalExtractor(settings, backend="openai", model=args.model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events: dict[str, list[dict]] = {}
    total_events = 0
    total_tokens_est = 0

    for i, (dep_id, title, dep_date, full_text, seg_count, char_count) in enumerate(depositions):
        word_count = len(full_text.split())
        dep_date_str = str(dep_date) if dep_date else None

        console.print(
            f"  [{i + 1}/{len(depositions)}] {title[:60]} "
            f"({word_count:,} words, {seg_count} segments)"
        )

        if args.dry_run:
            console.print(f"    [dim]DRY RUN: would extract events[/dim]")
            continue

        start = time.time()
        result = extractor.extract(full_text, document_id=dep_id, document_date=dep_date_str)
        elapsed = time.time() - start

        # Convert to dicts for Neon upsert
        event_dicts = [
            {
                "date": e.date,
                "date_raw": e.date_raw,
                "event_type": e.event_type,
                "description": e.description,
                "participants": e.participants,
                "locations": e.locations,
                "confidence": e.confidence,
                "source_chunk": e.source_chunk,
            }
            for e in result.events
        ]

        all_events[dep_id] = event_dicts
        total_events += len(event_dicts)
        total_tokens_est += word_count * 1.3

        console.print(
            f"    -> {len(result.events)} events in {elapsed:.1f}s"
        )

        # Save per-deposition JSON
        out_path = out_dir / f"{dep_id}-events.json"
        out_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # ── Step 4: Upsert to Neon ───────────────────────────────────────────
    if all_events and not args.dry_run:
        console.print(f"\n[dim]Upserting {total_events} events to Neon...[/dim]")

        # Check temporal_events columns again (may have been created above)
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'temporal_events' ORDER BY ordinal_position
        """)
        te_cols = [r[0] for r in cur.fetchall()]

        if te_cols:
            inserted = 0
            for doc_id, events in all_events.items():
                for event in events:
                    try:
                        cur.execute(
                            """
                            INSERT INTO temporal_events (
                                document_id, date, date_raw, event_type,
                                description, participants, locations,
                                confidence, source_chunk
                            ) VALUES (
                                %(document_id)s, %(date)s, %(date_raw)s,
                                %(event_type)s, %(description)s,
                                %(participants)s, %(locations)s,
                                %(confidence)s, %(source_chunk)s
                            )
                            ON CONFLICT (document_id, date, event_type, description)
                            DO UPDATE SET
                                participants = EXCLUDED.participants,
                                locations = EXCLUDED.locations,
                                confidence = EXCLUDED.confidence
                            """,
                            {
                                "document_id": doc_id,
                                "date": event["date"],
                                "date_raw": event.get("date_raw"),
                                "event_type": event["event_type"],
                                "description": event["description"],
                                "participants": event.get("participants", []),
                                "locations": event.get("locations", []),
                                "confidence": event.get("confidence", 0.5),
                                "source_chunk": event.get("source_chunk"),
                            },
                        )
                        inserted += 1
                    except Exception as exc:
                        console.print(f"    [red]Error inserting event: {exc}[/red]")
                        conn.rollback()

            conn.commit()
            console.print(f"  [green]Inserted {inserted} events into temporal_events[/green]")
        else:
            console.print("[yellow]temporal_events table not available, skipping DB insert[/yellow]")

    # ── Summary ──────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Depositions processed: {len(depositions)}")
    console.print(f"  Total events extracted: {total_events}")
    console.print(f"  Estimated cost: ~${total_tokens_est / 1e6 * 0.40 + total_events * 100 / 1e6 * 1.60:.2f}")
    console.print(f"  Events saved to: {out_dir}/")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
