"""Batch AI vision enrichment for DS10 media items.

Uses Anthropic Batch API for 50% cost savings on large runs.
Falls back to synchronous API for small batches.

Generates descriptions, classifies content type, detects people,
and updates ds10-media.json with enriched metadata.

Usage:
    # Process 100 items synchronously (fast, full price)
    python scripts/enrich-ds10-vision-batch.py --limit 100

    # Process 5000 items via Batch API (50% cheaper, ~24h turnaround)
    python scripts/enrich-ds10-vision-batch.py --limit 5000 --batch-api

    # Use Haiku for cheap initial classification pass
    python scripts/enrich-ds10-vision-batch.py --model claude-haiku-4-5-20251001 --limit 10000 --batch-api

    # Resume from checkpoint
    python scripts/enrich-ds10-vision-batch.py --resume output/ds10-vision-checkpoint.json

Prerequisites:
    pip install anthropic httpx python-dotenv
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
DS10_JSON = SITE_DIR / "data" / "ds10-media.json"
OUTPUT_DIR = Path("output")
CHECKPOINT_PATH = OUTPUT_DIR / "ds10-vision-checkpoint.json"

VISION_PROMPT = """Analyze this image from the Jeffrey Epstein DOJ case files (EFTA evidence).

Respond ONLY with valid JSON (no markdown, no explanation):
{
  "description": "2-3 sentence factual description of what's shown",
  "content_type": "one of: legal_document, financial_record, letter, photograph, handwritten_note, form, receipt, passport, phone_record, email_printout, fax, property_record, court_filing, other",
  "people_mentioned": ["list of any person names visible in the document text or photo"],
  "date_visible": "any date visible in the document (YYYY-MM-DD format) or null",
  "key_entities": ["organizations, addresses, phone numbers, or account numbers visible"],
  "tags": ["3-5 relevant keywords"],
  "has_redactions": true/false,
  "contains_handwriting": true/false,
  "evidence_value": "high/medium/low - how significant is this for the Epstein investigation"
}

Be precise and factual. Only list people whose names are clearly visible."""


def load_media() -> list[dict]:
    return json.loads(DS10_JSON.read_text("utf-8"))


def needs_enrichment(item: dict) -> bool:
    """Check if item needs vision enrichment."""
    # Skip if already has a real description
    desc = item.get("description", "")
    if desc and len(desc) > 30 and "Evidence Photo" not in desc:
        return False
    return True


def fetch_image_bytes(url: str) -> bytes | None:
    """Download image from CDN."""
    import httpx
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def enrich_sync(items: list[dict], model: str, api_key: str) -> list[dict]:
    """Synchronous enrichment via Messages API."""
    import httpx

    results = []
    total = len(items)

    for i, item in enumerate(items):
        img_bytes = fetch_image_bytes(item["thumbnailUrl"])
        if not img_bytes or len(img_bytes) < 1000:
            continue

        b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Detect media type from URL
        url = item["thumbnailUrl"].lower()
        media_type = "image/png" if url.endswith(".png") else "image/jpeg"

        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 600,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                            {"type": "text", "text": VISION_PROMPT},
                        ],
                    }],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]

            # Parse JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            enrichment = json.loads(text.strip())
            results.append({"id": item["id"], "enrichment": enrichment})

        except Exception as e:
            print(f"  Failed {item['id']}: {str(e)[:80]}", flush=True)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} processed, {len(results)} enriched", flush=True)

        time.sleep(0.3)  # Rate limit

    return results


def create_batch_requests(items: list[dict], model: str) -> list[dict]:
    """Create Anthropic Batch API request objects."""
    import httpx

    requests = []
    for item in items:
        img_bytes = fetch_image_bytes(item["thumbnailUrl"])
        if not img_bytes or len(img_bytes) < 1000:
            continue

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        url = item["thumbnailUrl"].lower()
        media_type = "image/png" if url.endswith(".png") else "image/jpeg"

        requests.append({
            "custom_id": item["id"],
            "params": {
                "model": model,
                "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }],
            },
        })

    return requests


def submit_batch(requests: list[dict], api_key: str) -> str:
    """Submit batch to Anthropic Batch API. Returns batch_id."""
    import httpx

    # Write JSONL file
    jsonl_path = OUTPUT_DIR / f"ds10-batch-{int(time.time())}.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Batch JSONL: {jsonl_path} ({len(requests):,} requests)", flush=True)

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"requests": requests},
        timeout=120.0,
    )
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"  Batch submitted: {batch_id}", flush=True)
    return batch_id


def poll_batch(batch_id: str, api_key: str) -> list[dict] | None:
    """Check batch status and retrieve results if complete."""
    import httpx

    resp = httpx.get(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    status = data.get("processing_status")
    print(f"  Batch {batch_id}: {status}", flush=True)

    if status == "ended":
        # Fetch results
        results_url = data.get("results_url")
        if results_url:
            resp = httpx.get(
                results_url,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                timeout=120.0,
            )
            results = []
            for line in resp.text.strip().split("\n"):
                result = json.loads(line)
                custom_id = result["custom_id"]
                if result["result"]["type"] == "succeeded":
                    text = result["result"]["message"]["content"][0]["text"]
                    try:
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0]
                        enrichment = json.loads(text.strip())
                        results.append({"id": custom_id, "enrichment": enrichment})
                    except json.JSONDecodeError:
                        pass
            return results

    return None


def apply_enrichments(media: list[dict], enrichments: list[dict], persons_lookup: dict) -> int:
    """Apply vision enrichments to media items. Returns count of updated items."""
    enrichment_map = {e["id"]: e["enrichment"] for e in enrichments}

    updated = 0
    for item in media:
        if item["id"] not in enrichment_map:
            continue

        e = enrichment_map[item["id"]]

        # Update description
        if e.get("description"):
            item["description"] = e["description"]

        # Update tags
        if e.get("tags"):
            existing = set(item.get("tags", []))
            item["tags"] = list(existing | set(e["tags"]))

        # Add content_type tag
        if e.get("content_type"):
            ct = e["content_type"].replace("_", "-")
            if ct not in item.get("tags", []):
                item.setdefault("tags", []).append(ct)

        # Link persons from vision results
        if e.get("people_mentioned"):
            for name in e["people_mentioned"]:
                name_lower = name.lower().strip()
                if name_lower in persons_lookup:
                    pid = persons_lookup[name_lower]
                    if pid not in item.get("personIds", []):
                        item.setdefault("personIds", []).append(pid)

        updated += 1

    return updated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DS10 Vision Enrichment")
    parser.add_argument("--limit", type=int, default=100, help="Max items to process")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Vision model")
    parser.add_argument("--batch-api", action="store_true", help="Use Batch API (50% cheaper)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint JSON")
    parser.add_argument("--poll-batch", type=str, default=None, help="Poll existing batch ID for results")
    parser.add_argument("--dry-run", action="store_true", help="Don't update ds10-media.json")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrite ds10-media.json)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", flush=True)
        sys.exit(1)

    print("DS10 Vision Enrichment Pipeline", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Mode: {'Batch API' if args.batch_api else 'Synchronous'}", flush=True)

    # Load media
    media = load_media()
    print(f"  {len(media):,} media items loaded", flush=True)

    # Build person name -> ID lookup for linking
    import psycopg2
    db_url = None
    persons_lookup = {}
    try:
        for line in (SITE_DIR / ".env.local").read_text("utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                db_url = line.split("=", 1)[1].strip().strip('"')
        if db_url:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM persons")
            for pid, name in cur.fetchall():
                persons_lookup[name.lower()] = pid
                # Also map last name
                parts = name.lower().split()
                if len(parts) >= 2:
                    persons_lookup[f"{parts[0]} {parts[-1]}"] = pid
                    persons_lookup[f"{parts[-1]}, {parts[0]}"] = pid
            conn.close()
            print(f"  {len(persons_lookup):,} person name mappings loaded", flush=True)
    except Exception as e:
        print(f"  Warning: Could not load persons: {e}", flush=True)

    # Handle batch polling
    if args.poll_batch:
        results = poll_batch(args.poll_batch, api_key)
        if results:
            print(f"  Batch complete! {len(results):,} results", flush=True)
            updated = apply_enrichments(media, results, persons_lookup)
            if not args.dry_run:
                out = Path(args.output) if args.output else DS10_JSON
                out.write_text(json.dumps(media, indent=2, ensure_ascii=False), "utf-8")
                print(f"  Updated {updated:,} items, written to {out}", flush=True)
        else:
            print("  Batch not yet complete. Try again later.", flush=True)
        return

    # Load checkpoint if resuming
    completed_ids = set()
    if args.resume and Path(args.resume).exists():
        checkpoint = json.loads(Path(args.resume).read_text("utf-8"))
        completed_ids = {e["id"] for e in checkpoint}
        print(f"  Resuming: {len(completed_ids):,} already completed", flush=True)

    # Find items needing enrichment
    to_process = [item for item in media if needs_enrichment(item) and item["id"] not in completed_ids]
    to_process = to_process[:args.limit]
    print(f"  {len(to_process):,} items to enrich", flush=True)

    if not to_process:
        print("Nothing to process!", flush=True)
        return

    if args.batch_api:
        # Create and submit batch
        print("\nPreparing batch requests (downloading images)...", flush=True)
        requests = create_batch_requests(to_process, args.model)
        print(f"  {len(requests):,} valid requests prepared", flush=True)

        if requests:
            batch_id = submit_batch(requests, api_key)
            print(f"\nBatch submitted! ID: {batch_id}", flush=True)
            print(f"Poll for results with:", flush=True)
            print(f"  python scripts/enrich-ds10-vision-batch.py --poll-batch {batch_id}", flush=True)
    else:
        # Synchronous processing
        print(f"\nProcessing {len(to_process)} items synchronously...", flush=True)
        enrichments = enrich_sync(to_process, args.model, api_key)

        # Save checkpoint
        all_enrichments = enrichments
        if args.resume and Path(args.resume).exists():
            prev = json.loads(Path(args.resume).read_text("utf-8"))
            all_enrichments = prev + enrichments

        CHECKPOINT_PATH.write_text(json.dumps(all_enrichments, indent=2, ensure_ascii=False), "utf-8")
        print(f"  Checkpoint saved: {CHECKPOINT_PATH}", flush=True)

        # Apply enrichments
        updated = apply_enrichments(media, enrichments, persons_lookup)
        print(f"  {updated:,} items enriched", flush=True)

        if not args.dry_run:
            out = Path(args.output) if args.output else DS10_JSON
            out.write_text(json.dumps(media, indent=2, ensure_ascii=False), "utf-8")
            print(f"  Written to {out}", flush=True)

    # Summary
    all_people = set()
    for e in (enrichments if not args.batch_api else []):
        for p in e.get("enrichment", {}).get("people_mentioned", []):
            all_people.add(p)

    if all_people:
        print(f"\nPeople detected: {', '.join(sorted(all_people)[:20])}", flush=True)


if __name__ == "__main__":
    main()
