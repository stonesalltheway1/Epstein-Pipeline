#!/usr/bin/env python3
"""
Quick test: detect faces in first 100 DS10 images to verify pipeline works.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "epstein-index"
IMAGES_DIR = Path("E:/epstein-ds10/images")

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env.local")

import psycopg2
from insightface.app import FaceAnalysis

print("Loading InsightFace models...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Models loaded.")

MIN_DET_SCORE = 0.65
MIN_FACE_WIDTH = 50
TEST_LIMIT = 200  # Only process first 200 images

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

# Get already processed
cur.execute("SELECT DISTINCT image_filename FROM face_detections")
processed = {r[0] for r in cur.fetchall()}
print(f"Already processed: {len(processed)}")

all_files = sorted(IMAGES_DIR.glob("*.png"))[:TEST_LIMIT]
remaining = [f for f in all_files if f.name not in processed]
print(f"Testing on {len(remaining)} images...")

faces_found = 0
images_with_faces = 0
start = time.time()

for filepath in remaining:
    filename = filepath.name
    parts = filename.replace(".png", "").split("_")
    doc_id = f"sd-10-{parts[0]}"
    page_num = int(parts[1].replace("p", ""))

    img = cv2.imread(str(filepath))
    if img is None:
        continue

    h, w = img.shape[:2]
    faces = app.get(img)

    has_face = False
    for face in (faces or []):
        det_score = float(face.det_score)
        if det_score < MIN_DET_SCORE:
            continue
        bbox = face.bbox
        fw = bbox[2] - bbox[0]
        if fw < MIN_FACE_WIDTH:
            continue

        has_face = True
        quality = det_score * 0.6 + min(fw / 200.0, 1.0) * 0.4

        cur.execute("""
            INSERT INTO face_detections
            (image_filename, doc_id, page_number, bbox_x, bbox_y, bbox_w, bbox_h,
             detection_confidence, face_quality_score, image_width, image_height)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            filename, doc_id, page_num,
            int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]),
            float(det_score), float(quality), w, h,
        ))
        detection_id = cur.fetchone()[0]

        emb_str = "[" + ",".join(f"{x:.6f}" for x in face.normed_embedding.tolist()) + "]"
        cur.execute("""
            INSERT INTO face_embeddings (detection_id, embedding, model)
            VALUES (%s, %s::vector, %s)
        """, (detection_id, emb_str, "arcface_r50"))

        faces_found += 1
        print(f"  FACE in {filename}: det={det_score:.2f}, width={fw:.0f}px")

    if has_face:
        images_with_faces += 1

conn.commit()

elapsed = time.time() - start
print(f"\nTest complete in {elapsed:.1f}s")
print(f"Processed: {len(remaining)} images")
print(f"Faces found: {faces_found} in {images_with_faces} images")
print(f"Rate: {len(remaining)/elapsed:.1f} img/s")
print(f"Hit rate: {images_with_faces/max(len(remaining),1)*100:.1f}%")

cur.execute("SELECT COUNT(*) FROM face_detections")
print(f"DB face_detections: {cur.fetchone()[0]}")
cur.execute("SELECT COUNT(*) FROM face_embeddings")
print(f"DB face_embeddings: {cur.fetchone()[0]}")

cur.close()
conn.close()
