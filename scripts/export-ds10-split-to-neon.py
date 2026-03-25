"""Re-export DS10 transcripts as individual depositions (one per EFTA file).

Replaces the single monolithic vd-ds10-seized-media record with 174 individual
deposition records, each with their own segments. Much more browsable.
"""

import json
import glob
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["EPSTEIN_NEON_DATABASE_URL"].replace("&sslnegotiation=direct", "")
TRANSCRIPT_DIR = Path("E:/epstein-video-depositions/transcripts/ds10")
CATALOG_PATH = Path("E:/epstein-video-depositions/raw/ds10-media-catalog.json")


def load_catalog():
    """Load the DS10 media catalog for file metadata."""
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            items = json.load(f)
        return {item["efta"]: item for item in items}
    return {}


def main():
    conn = psycopg.connect(DB_URL)
    cur = conn.cursor()

    # Load catalog for file sizes/extensions
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog)} entries")

    # Delete the old monolithic record
    cur.execute("DELETE FROM deposition_segments WHERE deposition_id = 'vd-ds10-seized-media'")
    old_segs = cur.rowcount
    cur.execute("DELETE FROM video_depositions WHERE id = 'vd-ds10-seized-media'")
    conn.commit()
    if old_segs:
        print(f"Deleted old monolithic record ({old_segs:,} segments)")

    # Read all transcript files
    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    print(f"Found {len(transcript_files)} transcript files")

    inserted_deps = 0
    inserted_segs = 0
    skipped = 0

    for tf in transcript_files:
        try:
            with open(tf, encoding="utf-8") as f:
                t = json.load(f)
        except Exception:
            continue

        segments = t.get("segments", [])
        text = t.get("text", "")
        duration = t.get("duration_seconds", 0)
        words = len(text.split())

        # Skip empty/silent files
        if len(segments) == 0 or words < 5:
            skipped += 1
            continue

        efta_id = os.path.basename(tf).replace(".json", "")
        dep_id = f"vd-ds10-{efta_id}"
        cat_entry = catalog.get(efta_id, {})
        ext = cat_entry.get("extension", ".mp4")

        # Generate a useful title from the first few words of transcript
        preview = text[:100].strip()
        if len(text) > 100:
            preview = preview.rsplit(" ", 1)[0] + "..."

        title = f"DS10: {efta_id} ({ext})"

        # Insert deposition record
        cur.execute("""
            INSERT INTO video_depositions (
                id, title, deponent, case_name,
                deposition_date, duration_seconds, source_type, source_url,
                language, segment_count, word_count, description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                duration_seconds = EXCLUDED.duration_seconds,
                segment_count = EXCLUDED.segment_count,
                word_count = EXCLUDED.word_count,
                description = EXCLUDED.description
        """, (
            dep_id,
            title,
            None,
            "DOJ EFTA Dataset 10 (Seized Media)",
            "2025-01-01",
            duration,
            "ds10",
            f"https://epsteinexposed.com/documents/{efta_id}",
            t.get("language", "en"),
            len(segments),
            words,
            preview,
        ))

        # Delete any existing segments for this deposition
        cur.execute("DELETE FROM deposition_segments WHERE deposition_id = %s", (dep_id,))

        # Insert segments
        if segments:
            values = []
            params = []
            for i, seg in enumerate(segments):
                values.append("(%s, %s, %s, %s, %s, %s)")
                params.extend([
                    dep_id,
                    i,
                    seg["start"],
                    seg["end"],
                    seg.get("speaker"),
                    seg["text"],
                ])

            # Batch insert (up to 500 at a time)
            batch_size = 500
            for batch_start in range(0, len(values), batch_size):
                batch_values = values[batch_start:batch_start + batch_size]
                batch_params = params[batch_start * 6:(batch_start + batch_size) * 6]
                sql = f"""
                    INSERT INTO deposition_segments
                        (deposition_id, segment_index, start_time, end_time, speaker, text)
                    VALUES {', '.join(batch_values)}
                """
                cur.execute(sql, batch_params)

        conn.commit()
        inserted_deps += 1
        inserted_segs += len(segments)

        if inserted_deps % 50 == 0:
            print(f"  {inserted_deps} depositions, {inserted_segs:,} segments inserted...")

    # Final stats
    cur.execute("SELECT COUNT(*) FROM video_depositions")
    total_deps = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM deposition_segments")
    total_segs = cur.fetchone()[0]

    print(f"\nDone!")
    print(f"  DS10 depositions created: {inserted_deps}")
    print(f"  DS10 segments inserted: {inserted_segs:,}")
    print(f"  Skipped (silent/empty): {skipped}")
    print(f"  Total depositions in DB: {total_deps}")
    print(f"  Total segments in DB: {total_segs:,}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
