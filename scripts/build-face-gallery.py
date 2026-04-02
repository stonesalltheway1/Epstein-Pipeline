#!/usr/bin/env python3
"""
Build face gallery from existing person headshots.
Sources:
  1. Local files in public/images/persons/ (364 files, slug-based filenames)
  2. Wikimedia URLs from data/headshots.json (134 additional URLs)
Detects faces, generates ArcFace embeddings, inserts into face_gallery table.
"""

import os
import sys
import json
import tempfile
import requests
import numpy as np
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "epstein-index"
HEADSHOTS_DIR = PROJECT_DIR / "public" / "images" / "persons"
HEADSHOTS_JSON = PROJECT_DIR / "data" / "headshots.json"

# Load environment from the Next.js project
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env.local")

import psycopg2
import cv2

DATABASE_URL = os.environ["DATABASE_URL"]

# InsightFace setup
from insightface.app import FaceAnalysis

print("Loading InsightFace models...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Models loaded.")


def get_face_embedding(img: np.ndarray) -> tuple[np.ndarray, dict] | None:
    """Detect the largest face and return its ArcFace embedding."""
    faces = app.get(img)
    if not faces:
        return None

    # Pick the largest face (by bounding box area)
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    if largest.det_score < 0.5:
        return None

    face_width = largest.bbox[2] - largest.bbox[0]
    if face_width < 40:
        return None

    embedding = largest.normed_embedding  # 512-dim, L2-normalized
    info = {
        "det_score": float(largest.det_score),
        "face_width": float(face_width),
        "bbox": [float(x) for x in largest.bbox],
    }
    return embedding, info


def insert_embedding(cur, conn, person_id: str, source_url: str, embedding: np.ndarray, info: dict, notes_prefix: str):
    """Insert a face embedding into face_gallery."""
    embedding_list = embedding.tolist()
    embedding_str = "[" + ",".join(f"{x:.6f}" for x in embedding_list) + "]"

    cur.execute("""
        INSERT INTO face_gallery (person_id, source_url, embedding, model, notes)
        VALUES (%s, %s, %s::vector, %s, %s)
    """, (
        person_id,
        source_url,
        embedding_str,
        "arcface_r50",
        f"{notes_prefix}, det_score={info['det_score']:.2f}, width={info['face_width']:.0f}px"
    ))
    conn.commit()


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Get all persons from DB (slug -> person_id mapping)
    cur.execute('SELECT id, slug, name FROM persons ORDER BY name')
    persons_db = {row[1]: (row[0], row[2]) for row in cur.fetchall()}
    print(f"Persons in DB: {len(persons_db)}")

    # Check existing gallery entries to avoid duplicates
    cur.execute("SELECT person_id, source_url FROM face_gallery")
    existing = {(r[0], r[1]) for r in cur.fetchall()}
    print(f"Existing gallery entries: {len(existing)}")

    inserted = 0
    skipped = 0
    failed = 0
    not_found = 0

    # === Source 1: Local headshot files ===
    print(f"\n--- Source 1: Local headshots from {HEADSHOTS_DIR} ---")
    local_files = sorted(HEADSHOTS_DIR.glob("*.*"))
    print(f"Found {len(local_files)} local headshot files.")

    for i, filepath in enumerate(local_files):
        slug = filepath.stem.lower()  # e.g., "alan-dershowitz"
        source_url = f"/images/persons/{filepath.name}"

        if slug not in persons_db:
            not_found += 1
            continue

        person_id, name = persons_db[slug]

        if (person_id, source_url) in existing:
            skipped += 1
            continue

        print(f"[{i+1}/{len(local_files)}] {name} ({slug})")

        img = cv2.imread(str(filepath))
        if img is None:
            print(f"  X Could not read image")
            failed += 1
            continue

        result = get_face_embedding(img)
        if result is None:
            print(f"  X No face detected")
            failed += 1
            continue

        embedding, info = result
        insert_embedding(cur, conn, person_id, source_url, embedding, info, "Local headshot")
        existing.add((person_id, source_url))
        inserted += 1
        print(f"  OK Embedded (det={info['det_score']:.2f}, width={info['face_width']:.0f}px)")

    print(f"\nLocal: Inserted={inserted}, Skipped={skipped}, Failed={failed}, NotInDB={not_found}")

    # === Source 2: Wikimedia URLs from headshots.json ===
    print(f"\n--- Source 2: Wikimedia URLs from headshots.json ---")
    wiki_inserted = 0
    wiki_skipped = 0
    wiki_failed = 0

    if HEADSHOTS_JSON.exists():
        headshots_data = json.loads(HEADSHOTS_JSON.read_text(encoding="utf-8"))
        ok_items = [item for item in headshots_data["items"] if item.get("status") == "ok" and item.get("imageUrl")]
        print(f"Found {len(ok_items)} Wikimedia headshots with status=ok.")

        for i, item in enumerate(ok_items):
            slug = item["slug"]
            image_url = item["imageUrl"]

            if slug not in persons_db:
                continue

            person_id, name = persons_db[slug]

            if (person_id, image_url) in existing:
                wiki_skipped += 1
                continue

            print(f"[{i+1}/{len(ok_items)}] {name} ({slug}) [wiki]")

            try:
                headers = {"User-Agent": "EpsteinExposed-FaceGallery/1.0"}
                resp = requests.get(image_url, headers=headers, timeout=15)
                resp.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(resp.content)
                    tmp_path = f.name

                img = cv2.imread(tmp_path)
                os.unlink(tmp_path)

                if img is None:
                    print(f"  X Could not decode image")
                    wiki_failed += 1
                    continue

            except Exception as e:
                print(f"  X Download failed: {e}")
                wiki_failed += 1
                continue

            result = get_face_embedding(img)
            if result is None:
                print(f"  X No face detected")
                wiki_failed += 1
                continue

            embedding, info = result
            insert_embedding(cur, conn, person_id, image_url, embedding, info, "Wikimedia Commons")
            existing.add((person_id, image_url))
            wiki_inserted += 1
            print(f"  OK Embedded (det={info['det_score']:.2f}, width={info['face_width']:.0f}px)")

        print(f"\nWiki: Inserted={wiki_inserted}, Skipped={wiki_skipped}, Failed={wiki_failed}")

    # === Summary ===
    total_inserted = inserted + wiki_inserted
    print(f"\n{'='*50}")
    print(f"TOTAL INSERTED: {total_inserted}")

    cur.execute("SELECT COUNT(*) FROM face_gallery")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT person_id) FROM face_gallery")
    unique_persons = cur.fetchone()[0]
    print(f"Gallery total: {total} embeddings covering {unique_persons} persons")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
