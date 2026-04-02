"""Export Maxwell interview transcripts to Neon Postgres.

Reads transcript JSON files from E:/epstein-video-depositions/transcripts/
and inserts them into the video_depositions + deposition_segments tables.

Usage:
    python scripts/export-depositions-to-neon.py
"""

import json
import glob
import os
import sys
from pathlib import Path
from urllib.parse import unquote

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["EPSTEIN_NEON_DATABASE_URL"].replace("&sslnegotiation=direct", "")
TRANSCRIPT_DIR = Path("E:/epstein-video-depositions/transcripts/maxwell-interview")
METADATA_DIR = Path("E:/epstein-video-depositions/raw/vd-maxwell-interview-2025")


def get_deposition_metadata():
    """Read the deposition metadata JSON."""
    meta_path = METADATA_DIR / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "id": "vd-maxwell-interview-2025",
        "title": "Ghislaine Maxwell — DOJ Prison Interview (Todd Blanche)",
        "deponent": "Ghislaine Maxwell",
        "case": "DOJ Investigation",
        "date": "2025-07-24",
        "source_type": "justice-gov",
        "person_id": "p-0002",
        "description": "2-day interview by DAG Todd Blanche at FCI Tallahassee, July 24-25, 2025.",
    }


def parse_part_info(filename: str) -> dict:
    """Extract day/part number from filename."""
    decoded = unquote(filename)
    # e.g., "Day 1 - Part 3 - 7_24_25_Tallahassee.005 (R).json"
    info = {"day": 0, "part": 0, "redacted": "(R)" in decoded}

    if "Day 1" in decoded:
        info["day"] = 1
    elif "Day 2" in decoded:
        info["day"] = 2

    for token in decoded.split("-"):
        token = token.strip()
        if token.startswith("Part"):
            try:
                info["part"] = int(token.split()[1])
            except (IndexError, ValueError):
                pass

    return info


def main():
    conn = psycopg.connect(DB_URL)
    cur = conn.cursor()

    meta = get_deposition_metadata()
    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))

    if not transcript_files:
        print("No transcript files found!")
        sys.exit(1)

    print(f"Found {len(transcript_files)} transcript files")

    # Calculate totals across all parts
    total_duration = 0
    total_segments = 0
    total_words = 0
    all_segments = []

    for tf in transcript_files:
        with open(tf, encoding="utf-8") as f:
            t = json.load(f)

        part_info = parse_part_info(os.path.basename(tf))
        total_duration += t.get("duration_seconds", 0)
        total_words += len(t.get("text", "").split())

        # Build segment records with part offset
        for i, seg in enumerate(t.get("segments", [])):
            all_segments.append({
                "segment_index": total_segments + i,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker"),
                "speaker_person_id": None,  # Will be linked later via NER
                "source_file": os.path.basename(tf),
                "day": part_info["day"],
                "part": part_info["part"],
            })

        total_segments += len(t.get("segments", []))

    print(f"Total: {total_duration/3600:.1f} hours, {total_segments:,} segments, ~{total_words:,} words")

    # Upsert the deposition record
    deposition_id = meta["id"]
    print(f"\nInserting deposition: {deposition_id}")

    cur.execute("""
        INSERT INTO video_depositions (
            id, title, deponent, deponent_person_id, case_name,
            deposition_date, duration_seconds, source_type, source_url,
            language, speaker_count, segment_count, word_count, description
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            duration_seconds = EXCLUDED.duration_seconds,
            segment_count = EXCLUDED.segment_count,
            word_count = EXCLUDED.word_count
    """, (
        deposition_id,
        meta["title"],
        meta["deponent"],
        meta.get("person_id"),
        meta["case"],
        meta["date"],
        total_duration,
        meta["source_type"],
        "https://www.justice.gov/maxwell-interview",
        "en",
        2,  # Maxwell + interviewer (Todd Blanche)
        total_segments,
        total_words,
        meta.get("description", ""),
    ))
    conn.commit()
    print(f"  Deposition record upserted: {total_duration/60:.0f} min, {total_segments} segments")

    # Delete existing segments for clean re-import
    cur.execute("DELETE FROM deposition_segments WHERE deposition_id = %s", (deposition_id,))
    conn.commit()

    # Batch insert segments
    print(f"\nInserting {len(all_segments):,} segments...")

    batch_size = 200
    inserted = 0
    for i in range(0, len(all_segments), batch_size):
        batch = all_segments[i:i + batch_size]
        values = []
        params = []
        for seg in batch:
            values.append("(%s, %s, %s, %s, %s, %s)")
            params.extend([
                deposition_id,
                seg["segment_index"],
                seg["start_time"],
                seg["end_time"],
                seg.get("speaker"),
                seg["text"],
            ])

        sql = f"""
            INSERT INTO deposition_segments
                (deposition_id, segment_index, start_time, end_time, speaker, text)
            VALUES {', '.join(values)}
        """
        cur.execute(sql, params)
        conn.commit()
        inserted += len(batch)

        if inserted % 1000 == 0 or inserted == len(all_segments):
            print(f"  {inserted:,}/{len(all_segments):,} segments inserted")

    print(f"\nDone! Exported to Neon:")
    print(f"  Deposition: {meta['title']}")
    print(f"  Duration: {total_duration/3600:.1f} hours")
    print(f"  Segments: {total_segments:,}")
    print(f"  Words: ~{total_words:,}")

    # Verify
    cur.execute("SELECT COUNT(*) FROM deposition_segments WHERE deposition_id = %s", (deposition_id,))
    count = cur.fetchone()[0]
    print(f"  Verified: {count:,} segments in database")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
