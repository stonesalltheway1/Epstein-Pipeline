#!/usr/bin/env python3
"""
Phase 3: Match detected face embeddings against the gallery.
Uses pgvector cosine similarity to find person matches.
Applies tiered confidence thresholds for auto-confirm vs human review.
"""

import os
import time
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "epstein-index"

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env.local")

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

# Match thresholds
AUTO_CONFIRM_THRESHOLD = 0.75    # Auto-confirm, no review needed
PENDING_THRESHOLD = 0.55         # Queue for human review
LOG_THRESHOLD = 0.45             # Log but don't display
BATCH_SIZE = 500


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Check existing matches to enable resume
    cur.execute("SELECT DISTINCT detection_id FROM face_matches")
    already_matched = {r[0] for r in cur.fetchall()}
    print(f"Already matched detections: {len(already_matched)}")

    # Get all embeddings that haven't been matched yet
    cur.execute("""
        SELECT fe.id, fe.detection_id, fe.embedding
        FROM face_embeddings fe
        WHERE fe.detection_id NOT IN (
            SELECT DISTINCT detection_id FROM face_matches
        )
        ORDER BY fe.id
    """)
    embeddings = cur.fetchall()
    print(f"Embeddings to match: {len(embeddings)}")

    if not embeddings:
        print("All embeddings already matched!")
        return

    auto_confirmed = 0
    pending = 0
    low_confidence = 0
    no_match = 0
    start_time = time.time()

    for i, (emb_id, detection_id, embedding_data) in enumerate(embeddings):
        # Find top matches from gallery using cosine similarity
        # pgvector <=> operator returns cosine distance (1 - similarity)
        cur.execute("""
            SELECT
                g.person_id,
                g.id as gallery_id,
                1 - (fe.embedding <=> g.embedding) as similarity,
                p.name
            FROM face_embeddings fe
            CROSS JOIN face_gallery g
            JOIN persons p ON p.id = g.person_id
            WHERE fe.id = %s
              AND 1 - (fe.embedding <=> g.embedding) > %s
            ORDER BY fe.embedding <=> g.embedding
            LIMIT 5
        """, (emb_id, LOG_THRESHOLD))

        matches = cur.fetchall()

        if not matches:
            no_match += 1
        else:
            for person_id, gallery_id, similarity, person_name in matches:
                if similarity >= AUTO_CONFIRM_THRESHOLD:
                    status = "auto_confirmed"
                    auto_confirmed += 1
                elif similarity >= PENDING_THRESHOLD:
                    status = "pending"
                    pending += 1
                else:
                    status = "low_confidence"
                    low_confidence += 1

                cur.execute("""
                    INSERT INTO face_matches
                    (detection_id, person_id, gallery_id, similarity, match_source, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (detection_id, person_id, gallery_id, similarity, "arcface", status))

                if similarity >= PENDING_THRESHOLD:
                    # Get detection info for logging
                    cur.execute("""
                        SELECT image_filename, doc_id FROM face_detections WHERE id = %s
                    """, (detection_id,))
                    det_info = cur.fetchone()
                    print(
                        f"  MATCH: {person_name} ({similarity:.3f}) "
                        f"[{status}] in {det_info[0]} ({det_info[1]})"
                    )

        # Batch commit
        if (i + 1) % BATCH_SIZE == 0:
            conn.commit()
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(embeddings) - i - 1) / rate / 60 if rate > 0 else 0
            print(
                f"[{i+1}/{len(embeddings)}] "
                f"auto={auto_confirmed}, pending={pending}, low={low_confidence}, none={no_match}, "
                f"{rate:.1f}/s, ETA {eta:.0f}min"
            )

    # Final commit
    conn.commit()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"MATCHING COMPLETE")
    print(f"Embeddings processed: {len(embeddings)}")
    print(f"Auto-confirmed: {auto_confirmed}")
    print(f"Pending review: {pending}")
    print(f"Low confidence: {low_confidence}")
    print(f"No match: {no_match}")
    print(f"Time: {elapsed/60:.1f} minutes")

    # Summary
    cur.execute("""
        SELECT status, COUNT(*) FROM face_matches
        GROUP BY status ORDER BY status
    """)
    print("\nFace matches by status:")
    for status, count in cur.fetchall():
        print(f"  {status}: {count}")

    cur.execute("""
        SELECT COUNT(DISTINCT person_id) FROM face_matches
        WHERE status IN ('auto_confirmed', 'pending')
    """)
    print(f"\nDistinct persons identified: {cur.fetchone()[0]}")

    cur.execute("""
        SELECT p.name, COUNT(*) as matches
        FROM face_matches fm
        JOIN persons p ON p.id = fm.person_id
        WHERE fm.status IN ('auto_confirmed', 'pending')
        GROUP BY p.name
        ORDER BY matches DESC
        LIMIT 20
    """)
    print("\nTop 20 persons by face matches:")
    for name, count in cur.fetchall():
        print(f"  {name}: {count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
