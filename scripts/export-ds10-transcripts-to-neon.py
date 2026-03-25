"""Export DS10 seized media transcripts to Neon Postgres.

Reads 719 transcript JSON files from E:/epstein-video-depositions/transcripts/ds10/
and inserts them into video_depositions + deposition_segments tables.

Groups files by EFTA ID range to create logical deposition records.
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


def main():
    conn = psycopg.connect(DB_URL)
    cur = conn.cursor()

    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    if not transcript_files:
        print("No transcript files found!")
        sys.exit(1)

    print(f"Found {len(transcript_files)} DS10 transcript files")

    # First pass: gather stats
    total_duration = 0
    total_segments = 0
    total_words = 0
    valid_files = []

    for tf in transcript_files:
        try:
            with open(tf, encoding="utf-8") as f:
                t = json.load(f)
            dur = t.get("duration_seconds", 0)
            segs = len(t.get("segments", []))
            words = len(t.get("text", "").split())

            # Skip empty/silent files
            if segs == 0 or words < 5:
                continue

            total_duration += dur
            total_segments += segs
            total_words += words
            valid_files.append((tf, t))
        except Exception as e:
            print(f"  Skipping {os.path.basename(tf)}: {e}")

    print(f"Valid files: {len(valid_files)} of {len(transcript_files)}")
    print(f"Total: {total_duration/3600:.1f} hours, {total_segments:,} segments, ~{total_words:,} words")

    # Create the DS10 deposition record
    deposition_id = "vd-ds10-seized-media"
    print(f"\nUpserting deposition: {deposition_id}")

    cur.execute("""
        INSERT INTO video_depositions (
            id, title, deponent, deponent_person_id, case_name,
            deposition_date, duration_seconds, source_type, source_url,
            language, speaker_count, segment_count, word_count, description
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            duration_seconds = EXCLUDED.duration_seconds,
            segment_count = EXCLUDED.segment_count,
            word_count = EXCLUDED.word_count,
            description = EXCLUDED.description
    """, (
        deposition_id,
        "DOJ Dataset 10 — Seized Device Media Transcripts",
        None,  # No single deponent — multiple sources
        None,
        "DOJ EFTA Release (Dataset 10)",
        "2025-01-01",  # Approximate release date
        total_duration,
        "ds10",
        "https://www.justice.gov/epstein",
        "en",
        0,
        total_segments,
        total_words,
        f"Transcripts of {len(valid_files)} audio/video files from devices seized during the Epstein investigation. "
        f"Part of DOJ Dataset 10 (Media & Digital Evidence). "
        f"{total_duration/3600:.1f} hours of audio transcribed using faster-whisper large-v3-turbo.",
    ))
    conn.commit()
    print(f"  Deposition record upserted")

    # Delete existing segments for clean re-import
    cur.execute("DELETE FROM deposition_segments WHERE deposition_id = %s", (deposition_id,))
    deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"  Cleared {deleted:,} existing segments")

    # Insert segments in batches
    print(f"\nInserting {total_segments:,} segments from {len(valid_files)} files...")

    batch_size = 500
    segment_offset = 0
    inserted = 0

    for file_idx, (tf, t) in enumerate(valid_files):
        efta_id = os.path.basename(tf).replace(".json", "")
        segments = t.get("segments", [])

        # Build batch
        values = []
        params = []
        for i, seg in enumerate(segments):
            values.append("(%s, %s, %s, %s, %s, %s)")
            params.extend([
                deposition_id,
                segment_offset + i,
                seg["start"],
                seg["end"],
                seg.get("speaker"),
                f"[{efta_id}] {seg['text']}",  # Prefix with EFTA ID for traceability
            ])

        if values:
            # Batch insert
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

            inserted += len(segments)
            segment_offset += len(segments)

        if (file_idx + 1) % 100 == 0 or file_idx == len(valid_files) - 1:
            print(f"  {file_idx + 1}/{len(valid_files)} files, {inserted:,} segments inserted")

    # Verify
    cur.execute("SELECT COUNT(*) FROM deposition_segments WHERE deposition_id = %s", (deposition_id,))
    count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM video_depositions")
    total_deps = cur.fetchone()[0]

    print(f"\nDone! DS10 export complete:")
    print(f"  Files processed: {len(valid_files)}")
    print(f"  Total duration: {total_duration/3600:.1f} hours")
    print(f"  Segments in DB: {count:,}")
    print(f"  Words: ~{total_words:,}")
    print(f"  Total depositions in DB: {total_deps}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
