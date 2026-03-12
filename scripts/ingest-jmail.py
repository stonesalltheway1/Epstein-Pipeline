"""
Ingest jmail.world datasets into Neon Postgres.

Handles: iMessages, photos, people, photo_faces, and emails.
Downloads parquet files from data.jmail.world, maps to DB schema, batch upserts.

Usage:
  python scripts/ingest-jmail.py --all                    # Ingest everything
  python scripts/ingest-jmail.py --imessages              # Just iMessages
  python scripts/ingest-jmail.py --photos                 # Photos + people + faces
  python scripts/ingest-jmail.py --emails                 # 1.78M emails (large!)
  python scripts/ingest-jmail.py --emails --no-promo      # Skip promotional emails
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

# Load the production .env.local from the site repo
SITE_DIR = Path("C:/Users/Eric/OneDrive/Desktop/epstein-index")
load_dotenv(SITE_DIR / ".env.local")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL not set. Check .env.local")
    sys.exit(1)

JMAIL_DIR = Path(__file__).parent.parent / "data" / "jmail"
JMAIL_BASE = "https://data.jmail.world/v1"

BATCH_SIZE = 500


def download_if_missing(filename: str) -> Path:
    """Download parquet file from jmail if not already local."""
    path = JMAIL_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    JMAIL_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = f"{JMAIL_BASE}/{filename}"
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path} ({path.stat().st_size:,} bytes)")
    return path


def ingest_imessages(conn):
    """Ingest 4,509 iMessages into imessages + imessage_conversations tables."""
    print("\n=== Ingesting iMessages ===")

    # Create tables
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS imessage_conversations (
                id INT PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                bio TEXT,
                photo TEXT,
                last_message TEXT,
                last_message_time TEXT,
                pinned BOOLEAN DEFAULT FALSE,
                confirmed BOOLEAN DEFAULT FALSE,
                source_files JSONB DEFAULT '[]',
                message_count INT DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS imessages (
                id TEXT PRIMARY KEY,
                conversation_slug TEXT NOT NULL,
                message_index INT NOT NULL,
                text TEXT,
                sender TEXT NOT NULL,
                time TEXT,
                timestamp TIMESTAMPTZ,
                source_file TEXT,
                sender_name TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_imsg_conv ON imessages(conversation_slug);
            CREATE INDEX IF NOT EXISTS idx_imsg_ts ON imessages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_imsg_sender ON imessages(sender);
        """)
    conn.commit()

    # Load conversations
    convos = pd.read_parquet(download_if_missing("imessage_conversations.parquet"))
    print(f"  Conversations: {len(convos)}")
    with conn.cursor() as cur:
        for _, r in convos.iterrows():
            cur.execute("""
                INSERT INTO imessage_conversations (id, slug, name, bio, photo, last_message,
                    last_message_time, pinned, confirmed, source_files, message_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                int(r["id"]), r["slug"], r["name"], r.get("bio"),
                r.get("photo"), r.get("last_message"), r.get("last_message_time"),
                bool(r.get("pinned", False)), bool(r.get("confirmed", False)),
                json.dumps(r.get("source_files", "[]") if isinstance(r.get("source_files"), list) else json.loads(r.get("source_files", "[]"))),
                int(r.get("message_count", 0)),
            ))
    conn.commit()

    # Load messages
    msgs = pd.read_parquet(download_if_missing("imessage_messages.parquet"))
    print(f"  Messages: {len(msgs)}")
    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(msgs), BATCH_SIZE):
            batch = msgs.iloc[i:i+BATCH_SIZE]
            for _, r in batch.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO imessages (id, conversation_slug, message_index, text,
                            sender, time, timestamp, source_file, sender_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        r["id"], r["conversation_slug"], int(r["message_index"]),
                        r.get("text"), r["sender"], r.get("time"),
                        r.get("timestamp"), r.get("source_file"), r.get("sender_name"),
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"    Error on {r['id']}: {e}")
            conn.commit()
            print(f"\r  Messages: {min(i+BATCH_SIZE, len(msgs))}/{len(msgs)}", end="", flush=True)

    print(f"\n  Inserted: {inserted} messages")


def ingest_photos(conn):
    """Ingest 18K photos, 473 people, 975 face detections."""
    print("\n=== Ingesting Photos + People + Faces ===")

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                source TEXT,
                release_batch TEXT,
                original_filename TEXT,
                content_type TEXT,
                size INT,
                width INT,
                height INT,
                image_description TEXT,
                source_url TEXT
            );
            CREATE TABLE IF NOT EXISTS photo_people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT,
                photo_count INT DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS photo_faces (
                id SERIAL PRIMARY KEY,
                photo_id TEXT NOT NULL,
                person_id TEXT,
                person_name TEXT,
                bbox_left REAL,
                bbox_top REAL,
                bbox_width REAL,
                bbox_height REAL,
                age_low INT,
                age_high INT,
                gender TEXT,
                confidence REAL,
                celebrity_confidence REAL
            );
            CREATE INDEX IF NOT EXISTS idx_photo_faces_photo ON photo_faces(photo_id);
            CREATE INDEX IF NOT EXISTS idx_photo_faces_person ON photo_faces(person_id);
        """)
    conn.commit()

    # Photos
    photos = pd.read_parquet(download_if_missing("photos.parquet"))
    print(f"  Photos: {len(photos)}")
    with conn.cursor() as cur:
        for i in range(0, len(photos), BATCH_SIZE):
            batch = photos.iloc[i:i+BATCH_SIZE]
            for _, r in batch.iterrows():
                cur.execute("""
                    INSERT INTO photos (id, source, release_batch, original_filename,
                        content_type, size, width, height, image_description, source_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    r["id"], r.get("source"), r.get("release_batch"),
                    r.get("original_filename"), r.get("content_type"),
                    int(r.get("size", 0)) if pd.notna(r.get("size")) else None,
                    int(r.get("width", 0)) if pd.notna(r.get("width")) else None,
                    int(r.get("height", 0)) if pd.notna(r.get("height")) else None,
                    r.get("image_description"), r.get("source_url"),
                ))
            conn.commit()
            print(f"\r  Photos: {min(i+BATCH_SIZE, len(photos))}/{len(photos)}", end="", flush=True)

    # People
    people = pd.read_parquet(download_if_missing("people.parquet"))
    print(f"\n  People: {len(people)}")
    with conn.cursor() as cur:
        for _, r in people.iterrows():
            cur.execute("""
                INSERT INTO photo_people (id, name, source, photo_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (r["id"], r["name"], r.get("source"), int(r.get("photo_count", 0))))
    conn.commit()

    # Faces
    faces = pd.read_parquet(download_if_missing("photo_faces.parquet"))
    print(f"  Faces: {len(faces)}")
    with conn.cursor() as cur:
        for _, r in faces.iterrows():
            cur.execute("""
                INSERT INTO photo_faces (photo_id, person_id, person_name,
                    bbox_left, bbox_top, bbox_width, bbox_height,
                    age_low, age_high, gender, confidence, celebrity_confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                r["photo_id"], r.get("person_id"), r.get("person_name"),
                float(r.get("bbox_left", 0)), float(r.get("bbox_top", 0)),
                float(r.get("bbox_width", 0)), float(r.get("bbox_height", 0)),
                int(r.get("age_low", 0)) if pd.notna(r.get("age_low")) else None,
                int(r.get("age_high", 0)) if pd.notna(r.get("age_high")) else None,
                r.get("gender"),
                float(r.get("confidence", 0)) if pd.notna(r.get("confidence")) else None,
                float(r.get("celebrity_confidence", 0)) if pd.notna(r.get("celebrity_confidence")) else None,
            ))
    conn.commit()


def ingest_emails(conn, skip_promo=False):
    """Ingest 1.78M emails from jmail.world."""
    print("\n=== Ingesting Emails ===")
    print("  This will take several minutes for 1.78M rows...")

    # We need to expand the emails table or create a new one
    # The existing emails table has a different schema, so create jmail_emails
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jmail_emails (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                message_index INT DEFAULT 0,
                sender TEXT,
                subject TEXT,
                to_recipients JSONB DEFAULT '[]',
                cc_recipients JSONB DEFAULT '[]',
                bcc_recipients JSONB DEFAULT '[]',
                sent_at TIMESTAMPTZ,
                attachments INT DEFAULT 0,
                account_email TEXT,
                email_drop_id TEXT,
                folder_path TEXT,
                is_promotional BOOLEAN DEFAULT FALSE,
                release_batch INT,
                epstein_is_sender BOOLEAN DEFAULT FALSE,
                all_participants TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jmail_emails_sent ON jmail_emails(sent_at);
            CREATE INDEX IF NOT EXISTS idx_jmail_emails_sender ON jmail_emails(sender);
            CREATE INDEX IF NOT EXISTS idx_jmail_emails_account ON jmail_emails(account_email);
            CREATE INDEX IF NOT EXISTS idx_jmail_emails_promo ON jmail_emails(is_promotional);
            CREATE INDEX IF NOT EXISTS idx_jmail_emails_doc ON jmail_emails(doc_id);
        """)
    conn.commit()

    # Check existing count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jmail_emails")
        existing = cur.fetchone()[0]
    print(f"  Existing rows: {existing:,}")

    path = download_if_missing("emails-slim.parquet")
    df = pd.read_parquet(path)
    print(f"  Total in parquet: {len(df):,}")

    if skip_promo:
        before = len(df)
        df = df[~df.is_promotional.fillna(False).astype(bool)]
        print(f"  After removing promotional: {len(df):,} (removed {before - len(df):,})")

    if existing > 0:
        # Get existing IDs to skip
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM jmail_emails")
            existing_ids = {row[0] for row in cur.fetchall()}
        df = df[~df.id.isin(existing_ids)]
        print(f"  After skipping existing: {len(df):,}")

    if len(df) == 0:
        print("  Nothing to insert!")
        return

    inserted = 0
    errors = 0
    # Use a separate autocommit connection to avoid transaction-abort cascading
    import psycopg
    ac_conn = psycopg.connect(DB_URL, autocommit=True)
    with ac_conn.cursor() as cur:
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            for _, r in batch.iterrows():
                try:
                    # Parse JSON arrays from string safely
                    def safe_json(val):
                        if pd.isna(val) or val is None:
                            return "[]"
                        if isinstance(val, str):
                            try:
                                return json.dumps(json.loads(val))
                            except Exception:
                                return "[]"
                        return json.dumps(val) if val else "[]"

                    # Validate timestamp — skip bogus dates
                    sent_at = r.get("sent_at")
                    if pd.notna(sent_at) and isinstance(sent_at, str):
                        # Skip dates before 1990 or after 2025 (data errors)
                        if sent_at < "1990" or sent_at > "2025":
                            sent_at = None
                    else:
                        sent_at = None

                    cur.execute("""
                        INSERT INTO jmail_emails (id, doc_id, message_index, sender, subject,
                            to_recipients, cc_recipients, bcc_recipients, sent_at, attachments,
                            account_email, email_drop_id, folder_path, is_promotional,
                            release_batch, epstein_is_sender, all_participants)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        r["id"], r.get("doc_id"), int(r.get("message_index", 0)),
                        r.get("sender"), r.get("subject"),
                        safe_json(r.get("to_recipients")),
                        safe_json(r.get("cc_recipients")),
                        safe_json(r.get("bcc_recipients")),
                        sent_at,
                        int(r.get("attachments", 0)),
                        r.get("account_email"), r.get("email_drop_id"),
                        r.get("folder_path"),
                        bool(r.get("is_promotional", False)) if pd.notna(r.get("is_promotional")) else False,
                        int(r.get("release_batch", 0)) if pd.notna(r.get("release_batch")) else None,
                        bool(r.get("epstein_is_sender", False)) if pd.notna(r.get("epstein_is_sender")) else False,
                        r.get("all_participants"),
                    ))
                    inserted += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"\n    Error: {e}")
            pct = min(i + BATCH_SIZE, len(df)) / len(df) * 100
            print(f"\r  Progress: {min(i+BATCH_SIZE, len(df)):,}/{len(df):,} ({pct:.0f}%) inserted={inserted:,} errors={errors}", end="", flush=True)
    ac_conn.close()

    print(f"\n  Done! Inserted: {inserted:,}, Errors: {errors}")


def verify(conn):
    """Print verification counts."""
    print("\n=== Verification ===")
    tables = [
        "imessage_conversations", "imessages",
        "photos", "photo_people", "photo_faces",
        "jmail_emails",
    ]
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"  {table}: {count:,}")
            except Exception:
                conn.rollback()
                print(f"  {table}: (not created)")


def main():
    parser = argparse.ArgumentParser(description="Ingest jmail.world data into Neon")
    parser.add_argument("--all", action="store_true", help="Ingest everything")
    parser.add_argument("--imessages", action="store_true", help="Ingest iMessages")
    parser.add_argument("--photos", action="store_true", help="Ingest photos + people + faces")
    parser.add_argument("--emails", action="store_true", help="Ingest 1.78M emails")
    parser.add_argument("--no-promo", action="store_true", help="Skip promotional emails")
    parser.add_argument("--verify", action="store_true", help="Just print counts")
    args = parser.parse_args()

    if not any([args.all, args.imessages, args.photos, args.emails, args.verify]):
        parser.print_help()
        return

    with psycopg.connect(DB_URL) as conn:
        if args.verify:
            verify(conn)
            return

        if args.all or args.imessages:
            ingest_imessages(conn)
        if args.all or args.photos:
            ingest_photos(conn)
        if args.all or args.emails:
            ingest_emails(conn, skip_promo=args.no_promo)

        verify(conn)


if __name__ == "__main__":
    main()
