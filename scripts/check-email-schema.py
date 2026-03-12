import os
import psycopg
from dotenv import load_dotenv

load_dotenv("C:/Users/Eric/OneDrive/Desktop/epstein-index/.env.local")

db_url = os.environ["DATABASE_URL"]

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'emails'
            ORDER BY ordinal_position
        """)
        print("=== emails table schema ===")
        for row in cur.fetchall():
            print(f"  {row[0]} ({row[1]}) {'NULL' if row[2] == 'YES' else 'NOT NULL'}")

        cur.execute("SELECT COUNT(*) FROM emails")
        print(f"\nTotal rows: {cur.fetchone()[0]}")

        cur.execute("SELECT * FROM emails LIMIT 1")
        cols = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        if row:
            print("\nSample row:")
            for c, v in zip(cols, row):
                val = str(v)[:100] if v else 'NULL'
                print(f"  {c}: {val}")
