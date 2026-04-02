#!/usr/bin/env python3
"""
Phase 2: Production face detection pipeline for DS10 document images.
GPU-accelerated, multi-scale, parallel I/O, bulk DB inserts.

Hardware target: GTX 1660 SUPER (6GB VRAM), Python 3.13, Neon Postgres.
"""

import os
import re
import sys
import time
import json
import queue
import threading
import traceback
import ctypes
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

IMAGES_DIR = Path("E:/epstein-ds10/images")
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "epstein-index"

MIN_DET_SCORE = 0.5
MIN_FACE_WIDTH = 40
BATCH_SIZE = 50
READER_WORKERS = 6
QUEUE_MAXSIZE = 200
DET_SIZES = [(320, 320), (640, 640), (1280, 1280)]
NMS_IOU_THRESHOLD = 0.4
ENABLE_PREPROCESSING = True
SAVE_FACE_CROPS = True
FACE_CROPS_DIR = Path("E:/epstein-ds10/face_crops")
PROGRESS_INTERVAL = 200

# ─── CUDA DLL Preloading (MUST happen before any onnxruntime import) ────────

def preload_cuda_dlls():
    """Preload CUDA DLLs from pip-installed nvidia packages for Windows."""
    import glob as _glob
    capi = os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime", "capi")
    if os.path.isdir(capi):
        os.add_dll_directory(capi)
        for dll in sorted(_glob.glob(os.path.join(capi, "*.dll"))):
            name = os.path.basename(dll)
            if name.startswith(("cu", "nv")):
                try:
                    ctypes.CDLL(dll)
                except OSError:
                    pass

    # Also try onnxruntime's built-in preload
    try:
        import onnxruntime
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()
    except Exception:
        pass

preload_cuda_dlls()

# ─── Environment Setup ──────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env.local")

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


def get_database_url() -> str:
    """Get direct (non-pooled) database URL for bulk operations."""
    url = os.environ["DATABASE_URL"]
    if "-pooler" in url:
        direct_url = url.replace("-pooler", "")
        masked = re.sub(r"://[^@]+@", "://***@", direct_url)
        print(f"INFO: Switching from pooled to direct connection: {masked}")
        return direct_url
    return url


def create_connection():
    """Create a psycopg2 connection with TCP keepalive for long-running jobs."""
    url = get_database_url()
    conn = psycopg2.connect(
        url,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    register_vector(conn)
    return conn


# ─── InsightFace Setup ──────────────────────────────────────────────────────

from insightface.app import FaceAnalysis


def setup_insightface():
    """
    Initialize one FaceAnalysis per det_size for multi-scale detection.
    Uses allowed_modules=["detection","recognition"] to skip landmark/genderage.
    """
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"Initializing InsightFace with providers: {providers}")

    apps = {}
    for det_size in DET_SIZES:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=det_size)
        apps[det_size] = app
        print(f"  Prepared det_size={det_size}")

    # Report which provider was actually used
    first_app = apps[DET_SIZES[0]]
    for model in first_app.models.values():
        if hasattr(model, "session"):
            active = model.session.get_providers()
            print(f"  Active provider: {active}")
            break

    return apps


# ─── Image Preprocessing ────────────────────────────────────────────────────


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE enhancement for low-contrast document scans."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if contrast < 50 or mean_brightness < 80 or mean_brightness > 200:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return img


# ─── Multi-Scale Detection ──────────────────────────────────────────────────


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def nms_faces(faces, iou_threshold=NMS_IOU_THRESHOLD):
    """Non-maximum suppression across multi-scale detections."""
    if len(faces) <= 1:
        return faces

    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    keep = []

    for face in faces:
        bbox = face.bbox
        suppressed = False
        for kept_face in keep:
            if compute_iou(bbox, kept_face.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(face)

    return keep


def detect_multiscale(apps, img):
    """Run face detection at multiple pre-prepared scales and merge results."""
    h, w = img.shape[:2]
    all_faces = []

    for det_size in DET_SIZES:
        # Skip large det_size on small images
        if max(h, w) < det_size[0] * 0.5 and det_size[0] > 640:
            continue
        if max(h, w) < 400 and det_size[0] > 320:
            continue

        app = apps[det_size]  # Already prepared, zero overhead
        faces = app.get(img)
        all_faces.extend(faces or [])

    if not all_faces:
        return []

    return nms_faces(all_faces)


# ─── Face Quality Scoring ───────────────────────────────────────────────────


def compute_quality_score(face, img) -> dict:
    """Compute multi-factor face quality score."""
    bbox = face.bbox
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    fw = x2 - x1
    fh = y2 - y1

    scores = {}

    # 1. Detection confidence
    scores["det_score"] = float(face.det_score)

    # 2. Size score (normalized by 200px)
    scores["size_score"] = min(fw / 200.0, 1.0)

    # 3. Aspect ratio score (ideal face ~0.75 w/h ratio)
    aspect = fw / max(fh, 1)
    scores["aspect_score"] = 1.0 - min(abs(aspect - 0.75) / 0.5, 1.0)

    # 4. Blur detection (Laplacian variance on face crop)
    h, w = img.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    if cx2 > cx1 + 10 and cy2 > cy1 + 10:
        crop = img[cy1:cy2, cx1:cx2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        scores["blur_score"] = min(lap_var / 500.0, 1.0)
    else:
        scores["blur_score"] = 0.0

    # 5. Pose estimation from 5 keypoints (yaw from eye-nose positions)
    if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 3:
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]

        eye_dist = np.linalg.norm(left_eye - right_eye)
        eye_mid = (left_eye + right_eye) / 2
        nose_offset = abs(nose[0] - eye_mid[0]) / max(eye_dist, 1)
        scores["pose_score"] = max(0, 1.0 - nose_offset * 2)
    else:
        scores["pose_score"] = 0.5

    # Weighted composite
    composite = (
        scores["det_score"] * 0.25
        + scores["size_score"] * 0.20
        + scores["aspect_score"] * 0.10
        + scores["blur_score"] * 0.20
        + scores["pose_score"] * 0.25
    )
    scores["composite"] = composite

    return scores


# ─── Face Crop Extraction ───────────────────────────────────────────────────


def extract_face_crop(face, img) -> bytes | None:
    """Extract 112x112 aligned face crop as JPEG bytes."""
    if not SAVE_FACE_CROPS:
        return None
    try:
        bbox = face.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 + 5 and y2 > y1 + 5:
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (112, 112))
            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes()
    except Exception:
        pass
    return None


def save_face_crop_to_disk(detection_id: int, crop_bytes: bytes | None):
    """Save pre-extracted crop bytes to disk."""
    if not crop_bytes:
        return
    try:
        crop_path = FACE_CROPS_DIR / f"{detection_id}.jpg"
        crop_path.write_bytes(crop_bytes)
    except Exception:
        pass


# ─── Index Management ───────────────────────────────────────────────────────


def manage_indexes(conn, action="check"):
    """Check, drop, or rebuild vector indexes for bulk insert performance."""
    cur = conn.cursor()

    cur.execute("""
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename IN ('face_embeddings', 'face_gallery')
          AND indexdef LIKE '%hnsw%'
        ORDER BY indexname
    """)
    indexes = cur.fetchall()

    if action == "check":
        if indexes:
            print(
                f"WARNING: {len(indexes)} HNSW vector indexes found (slows bulk insert):"
            )
            for name, defn in indexes:
                print(f"  {name}")
            return indexes
        else:
            print("No HNSW indexes found (good for bulk insert).")
            return []

    elif action == "drop":
        for name, _ in indexes:
            if "face_emb" in name:  # Only drop face_embeddings indexes
                print(f"  Dropping index: {name}")
                cur.execute(f"DROP INDEX IF EXISTS {name}")
        conn.commit()
        return indexes

    elif action == "rebuild":
        print("Rebuilding HNSW index on face_embeddings...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_emb_hnsw
            ON face_embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        conn.commit()
        print("  Index rebuilt.")

    cur.close()


# ─── Filename Parser ────────────────────────────────────────────────────────


def parse_filename(filename: str) -> tuple:
    """Parse DS10 filename: EFTA01262782_p0_i0.png -> ('sd-10-EFTA01262782', 0)"""
    parts = filename.replace(".png", "").split("_")
    efta_id = parts[0]
    page_num = int(parts[1].replace("p", ""))
    return f"sd-10-{efta_id}", page_num


# ─── Reader Pool ─────────────────────────────────────────────────────────────


def image_reader(filepath: Path, preprocess: bool) -> tuple:
    """Read and optionally preprocess an image (runs in thread pool)."""
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return filepath, None
        if preprocess:
            img = preprocess_image(img)
        return filepath, img
    except Exception:
        return filepath, None


def reader_producer(file_list, img_queue, preprocess, stop_event):
    """Producer: reads images from disk into queue in controlled batches."""
    with ThreadPoolExecutor(max_workers=READER_WORKERS) as pool:
        idx = 0
        pending = []

        while idx < len(file_list) and not stop_event.is_set():
            # Submit a small batch (bounded by worker count)
            while len(pending) < READER_WORKERS * 2 and idx < len(file_list):
                future = pool.submit(image_reader, file_list[idx], preprocess)
                pending.append(future)
                idx += 1

            # Drain completed futures into queue
            done = [f for f in pending if f.done()]
            for f in done:
                pending.remove(f)
                try:
                    result = f.result(timeout=5)
                    img_queue.put(result, timeout=30)
                except Exception:
                    pass

            if not done:
                time.sleep(0.01)

        # Drain remaining pending futures
        for f in pending:
            if stop_event.is_set():
                break
            try:
                result = f.result(timeout=30)
                img_queue.put(result, timeout=30)
            except Exception:
                pass

    img_queue.put(None)  # Sentinel


# ─── DB Writer ───────────────────────────────────────────────────────────────


class DBWriter:
    """Accumulates face data and bulk-inserts in batches.
    Includes Neon keepalive ping (every 4 min) and reconnect-on-failure."""

    def __init__(self, conn):
        self.conn = conn
        self.buffer = []
        self.total_inserted = 0
        self.lock = threading.Lock()
        self.last_flush_time = time.time()
        self._stop_ping = threading.Event()
        self._ping_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ping_thread.start()

    def _keepalive_loop(self):
        """Send SELECT 1 every 4 minutes to prevent Neon idle timeout."""
        while not self._stop_ping.wait(240):
            try:
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                    self.conn.commit()
            except Exception as e:
                print(f"Keepalive ping failed: {e} — will reconnect on next flush")

    def _reconnect(self):
        """Reconnect to Neon if connection is dead."""
        try:
            self.conn.close()
        except Exception:
            pass
        print("Reconnecting to Neon...")
        self.conn = create_connection()
        print("Reconnected.")

    def add(
        self,
        filename: str,
        doc_id: str,
        page_num: int,
        bbox: tuple,
        det_confidence: float,
        quality_composite: float,
        img_w: int,
        img_h: int,
        embedding: np.ndarray,
        crop_bytes: bytes | None,
    ):
        """Add a detected face to the buffer (no large image refs)."""
        with self.lock:
            self.buffer.append(
                {
                    "image_filename": filename,
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "detection_confidence": det_confidence,
                    "face_quality_score": quality_composite,
                    "image_width": img_w,
                    "image_height": img_h,
                    "embedding": np.array(embedding, dtype=np.float32),
                    "crop_bytes": crop_bytes,
                }
            )

            # Flush if batch full or 5 min since last flush
            time_since_flush = time.time() - self.last_flush_time
            if len(self.buffer) >= BATCH_SIZE or (self.buffer and time_since_flush > 300):
                self._flush()

    def _flush(self):
        """Bulk insert buffered detections and embeddings."""
        if not self.buffer:
            return

        batch = self.buffer[:]
        self.buffer = []
        self.last_flush_time = time.time()

        # Check connection health, reconnect if needed
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        except Exception:
            self._reconnect()

        cur = self.conn.cursor()
        try:
            # Phase 1: Bulk insert detections, returning id + filename + bbox
            # to reliably match back to batch entries
            det_values = [
                (
                    d["image_filename"],
                    d["doc_id"],
                    d["page_number"],
                    d["bbox_x"],
                    d["bbox_y"],
                    d["bbox_w"],
                    d["bbox_h"],
                    d["detection_confidence"],
                    d["face_quality_score"],
                    d["image_width"],
                    d["image_height"],
                )
                for d in batch
            ]

            det_sql = """
                INSERT INTO face_detections
                (image_filename, doc_id, page_number, bbox_x, bbox_y, bbox_w, bbox_h,
                 detection_confidence, face_quality_score, image_width, image_height)
                VALUES %s
                RETURNING id, image_filename, bbox_x, bbox_y
            """
            result = psycopg2.extras.execute_values(
                cur, det_sql, det_values, fetch=True
            )

            # Build mapping: (filename, bbox_x, bbox_y) -> det_id
            id_map = {}
            for row in result:
                key = (row[1], row[2], row[3])
                id_map[key] = row[0]

            # Phase 2: Bulk insert embeddings with pgvector
            emb_values = []
            crop_tasks = []
            for d in batch:
                key = (d["image_filename"], d["bbox_x"], d["bbox_y"])
                det_id = id_map.get(key)
                if det_id is None:
                    continue
                emb_values.append((det_id, d["embedding"], "w600k_r50_scrfd10gf"))
                if d["crop_bytes"]:
                    crop_tasks.append((det_id, d["crop_bytes"]))

            if emb_values:
                emb_sql = """
                    INSERT INTO face_embeddings (detection_id, embedding, model)
                    VALUES %s
                """
                psycopg2.extras.execute_values(cur, emb_sql, emb_values)

            self.conn.commit()
            self.total_inserted += len(emb_values)

            # Save face crops to disk (non-critical)
            if SAVE_FACE_CROPS:
                for det_id, crop_bytes in crop_tasks:
                    save_face_crop_to_disk(det_id, crop_bytes)

        except Exception as e:
            print(f"DB FLUSH ERROR: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            # Reconnect and fallback to individual inserts
            self._reconnect()
            cur = self.conn.cursor()
            for d in batch:
                try:
                    cur.execute(
                        """
                        INSERT INTO face_detections
                        (image_filename, doc_id, page_number, bbox_x, bbox_y, bbox_w, bbox_h,
                         detection_confidence, face_quality_score, image_width, image_height)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id
                    """,
                        (
                            d["image_filename"],
                            d["doc_id"],
                            d["page_number"],
                            d["bbox_x"],
                            d["bbox_y"],
                            d["bbox_w"],
                            d["bbox_h"],
                            d["detection_confidence"],
                            d["face_quality_score"],
                            d["image_width"],
                            d["image_height"],
                        ),
                    )
                    det_id = cur.fetchone()[0]
                    cur.execute(
                        """
                        INSERT INTO face_embeddings (detection_id, embedding, model)
                        VALUES (%s, %s, %s)
                    """,
                        (det_id, d["embedding"], "w600k_r50_scrfd10gf"),
                    )
                    self.conn.commit()
                    self.total_inserted += 1
                    if SAVE_FACE_CROPS and d["crop_bytes"]:
                        save_face_crop_to_disk(det_id, d["crop_bytes"])
                except Exception:
                    self.conn.rollback()
        finally:
            cur.close()

    def finalize(self):
        """Flush remaining buffer and stop keepalive."""
        self._stop_ping.set()
        with self.lock:
            self._flush()


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DS10 face detection pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    args = parser.parse_args()

    start_time = time.time()

    # ── Diagnostics ──
    print("=" * 60)
    print("FACE DETECTION PIPELINE - DS10 Document Images")
    print("=" * 60)

    # GPU info
    try:
        import subprocess

        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        print(f"GPU: {gpu_info}")
    except Exception:
        print("GPU: nvidia-smi not available")

    # Initialize InsightFace (one instance per det_size)
    apps = setup_insightface()
    print(f"InsightFace: {len(apps)} scale instances loaded.")

    # Database connection
    conn = create_connection()
    print("Database connected (direct, with keepalive).")

    # Index management
    existing_indexes = manage_indexes(conn, "check")
    if existing_indexes:
        print("Dropping HNSW indexes for bulk insert performance...")
        manage_indexes(conn, "drop")

    # Create face crops directory
    if SAVE_FACE_CROPS:
        FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Face crops dir: {FACE_CROPS_DIR}")

    # Resume: get already-processed filenames
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT image_filename) FROM face_detections")
    processed_count = cur.fetchone()[0]
    print(f"Already processed: {processed_count} images")

    processed = set()
    if processed_count > 0:
        cur.execute("SELECT DISTINCT image_filename FROM face_detections")
        processed = {r[0] for r in cur.fetchall()}
    cur.close()

    # Get all image files
    all_files = sorted(IMAGES_DIR.glob("*.png"))
    total_files = len(all_files)
    remaining = [f for f in all_files if f.name not in processed]
    if args.limit > 0:
        remaining = remaining[: args.limit]
    print(f"Total images: {total_files}, Remaining: {len(remaining)}"
          + (f" (limited to {args.limit})" if args.limit > 0 else ""))

    if not remaining:
        print("All images already processed!")
        if existing_indexes:
            manage_indexes(conn, "rebuild")
        conn.close()
        return

    # ── Pipeline Setup ──
    img_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    db_writer = DBWriter(conn)

    reader_thread = threading.Thread(
        target=reader_producer,
        args=(remaining, img_queue, ENABLE_PREPROCESSING, stop_event),
        daemon=True,
    )
    reader_thread.start()
    print(
        f"\nReader pool: {READER_WORKERS} workers, queue {QUEUE_MAXSIZE}"
    )
    print(f"Detection scales: {DET_SIZES}")
    print(f"Min det score: {MIN_DET_SCORE}, Min face width: {MIN_FACE_WIDTH}px")
    print("-" * 60)

    # ── Main Detection Loop ──
    images_processed = 0
    images_with_faces = 0
    total_faces = 0
    errors = 0

    try:
        while True:
            item = img_queue.get(timeout=120)
            if item is None:  # Sentinel from reader
                break

            filepath, img = item
            if img is None:
                errors += 1
                continue

            filename = filepath.name
            doc_id, page_num = parse_filename(filename)
            h, w = img.shape[:2]

            # Multi-scale detection (uses pre-prepared instances)
            faces = detect_multiscale(apps, img)
            images_processed += 1

            valid_faces = 0
            for face in faces:
                det_score = float(face.det_score)
                if det_score < MIN_DET_SCORE:
                    continue
                fw = float(face.bbox[2] - face.bbox[0])
                if fw < MIN_FACE_WIDTH:
                    continue

                # Compute quality + extract crop BEFORE buffering (no img ref stored)
                quality = compute_quality_score(face, img)
                crop_bytes = extract_face_crop(face, img)
                embedding = face.normed_embedding

                bbox = face.bbox
                db_writer.add(
                    filename,
                    doc_id,
                    page_num,
                    (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                    det_score,
                    float(quality["composite"]),
                    w,
                    h,
                    embedding,
                    crop_bytes,
                )
                valid_faces += 1
                total_faces += 1

            if valid_faces > 0:
                images_with_faces += 1

            # Progress reporting
            if images_processed % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = images_processed / elapsed
                eta_min = (
                    (len(remaining) - images_processed) / rate / 60 if rate > 0 else 0
                )
                hit_pct = (
                    images_with_faces / images_processed * 100
                    if images_processed > 0
                    else 0
                )
                queue_sz = img_queue.qsize()

                gpu_mem = ""
                try:
                    import subprocess

                    mem = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used,memory.total",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        .strip()
                        .split(", ")
                    )
                    gpu_mem = f", GPU {mem[0]}/{mem[1]}MB"
                except Exception:
                    pass

                print(
                    f"[{images_processed}/{len(remaining)}] "
                    f"{total_faces} faces in {images_with_faces} imgs "
                    f"({hit_pct:.1f}% hit), "
                    f"{rate:.1f} img/s, ETA {eta_min:.0f}min, "
                    f"q={queue_sz}, errs={errors}{gpu_mem}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted! Flushing buffer...")
        stop_event.set()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        stop_event.set()

    # ── Finalize ──
    db_writer.finalize()

    elapsed = time.time() - start_time
    hit_pct = images_with_faces / max(images_processed, 1) * 100

    print(f"\n{'=' * 60}")
    print("DETECTION COMPLETE")
    print(f"Images processed: {images_processed}")
    print(f"Images with faces: {images_with_faces} ({hit_pct:.1f}%)")
    print(f"Total faces detected: {total_faces}")
    print(f"Total inserted to DB: {db_writer.total_inserted}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hours)")
    print(f"Rate: {images_processed / max(elapsed, 1):.1f} images/sec")

    # Use the DBWriter's connection (may have been reconnected)
    conn = db_writer.conn

    # Rebuild indexes
    if existing_indexes:
        print("\nRebuilding HNSW indexes...")
        try:
            manage_indexes(conn, "rebuild")
        except Exception as e:
            print(f"Index rebuild error: {e} — reconnecting...")
            conn = create_connection()
            manage_indexes(conn, "rebuild")

    # Run ANALYZE
    try:
        cur = conn.cursor()
        print("Running ANALYZE on face tables...")
        cur.execute("ANALYZE face_detections")
        cur.execute("ANALYZE face_embeddings")
        conn.commit()

        # Final DB stats
        cur.execute("SELECT COUNT(*) FROM face_detections")
        print(f"\nDB face_detections: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM face_embeddings")
        print(f"DB face_embeddings: {cur.fetchone()[0]}")
        cur.close()
    except Exception as e:
        print(f"Final stats error: {e}")

    try:
        conn.close()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
