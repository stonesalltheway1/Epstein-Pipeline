"""Upload DS10 extracted images to Cloudflare R2 via direct API.

Uses async HTTP PUTs with rate limiting and retry for 429s.
Key structure: ds10/{efta_id}/{filename}

Usage:
    python scripts/upload-ds10-to-r2.py --only-json --workers 4
    python scripts/upload-ds10-to-r2.py --limit 1000
    python scripts/upload-ds10-to-r2.py  # upload all
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx

IMAGES_DIR = Path("E:/epstein-ds10/images")
BUCKET = "epstein-exposed-media"
SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
DS10_JSON = SITE_DIR / "data" / "ds10-media.json"

ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
API_TOKEN = os.environ.get("CF_API_TOKEN", "")
R2_API = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/r2/buckets/{BUCKET}/objects"

EXT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
}

uploaded = 0
failed = 0
total_bytes = 0
retries_total = 0


def get_r2_key(filename: str) -> str:
    stem = filename.split("_p")[0]
    return f"ds10/{stem}/{filename}"


async def upload_one(client: httpx.AsyncClient, img_path: Path, semaphore: asyncio.Semaphore):
    global uploaded, failed, total_bytes, retries_total
    key = get_r2_key(img_path.name)
    ext = img_path.suffix.lower()
    content_type = EXT_TYPES.get(ext, "application/octet-stream")
    data = img_path.read_bytes()

    for attempt in range(5):
        async with semaphore:
            try:
                resp = await client.put(
                    f"{R2_API}/{key}",
                    content=data,
                    headers={"Content-Type": content_type},
                    timeout=120,
                )
                if resp.status_code == 200:
                    uploaded += 1
                    total_bytes += len(data)
                    return
                elif resp.status_code == 429:
                    retries_total += 1
                    delay = 2 ** attempt + 1
                    await asyncio.sleep(delay)
                    continue
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"  FAIL {img_path.name}: HTTP {resp.status_code}", flush=True)
                    return
            except Exception as e:
                if attempt < 4:
                    retries_total += 1
                    await asyncio.sleep(2 ** attempt)
                    continue
                failed += 1
                if failed <= 5:
                    print(f"  FAIL {img_path.name}: {str(e)[:100]}", flush=True)
                return

    failed += 1
    if failed <= 5:
        print(f"  FAIL {img_path.name}: max retries exceeded", flush=True)


async def run(files: list[Path], max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    limits = httpx.Limits(max_connections=max_concurrent + 2, max_keepalive_connections=max_concurrent)

    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        start = time.time()
        total = len(files)
        batch_size = 50  # smaller batches to avoid overwhelming

        for batch_start in range(0, total, batch_size):
            batch = files[batch_start:batch_start + batch_size]
            tasks = [upload_one(client, f, semaphore) for f in batch]
            await asyncio.gather(*tasks)

            done = batch_start + len(batch)
            elapsed = time.time() - start
            rate = uploaded / elapsed if elapsed > 0 else 0
            mb_rate = total_bytes / elapsed / 1e6 if elapsed > 0 else 0
            remaining = (total - done) / (done / elapsed) / 60 if done > 0 and elapsed > 0 else 0
            print(f"  {done:,}/{total:,} | {uploaded:,} ok, {failed:,} fail, {retries_total} retries | {rate:.1f}/s ({mb_rate:.1f} MB/s) | ~{remaining:.0f}m left", flush=True)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Uploaded: {uploaded:,} ({total_bytes/1e9:.2f} GB)", flush=True)
    print(f"  Failed: {failed:,}", flush=True)
    print(f"  Retries: {retries_total:,}", flush=True)
    if elapsed > 0:
        print(f"  Rate: {uploaded/elapsed:.1f} files/s, {total_bytes/elapsed/1e6:.1f} MB/s", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent uploads")
    parser.add_argument("--only-json", action="store_true", help="Only upload files referenced in ds10-media.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.only_json:
        with open(DS10_JSON, "r", encoding="utf-8") as f:
            items = json.load(f)
        needed_files = set()
        for item in items:
            url = item.get("thumbnailUrl", "")
            filename = url.rsplit("/", 1)[-1] if "/" in url else ""
            if filename:
                needed_files.add(filename)
        print(f"ds10-media.json references {len(needed_files):,} unique files", flush=True)
        files = [IMAGES_DIR / f for f in sorted(needed_files) if (IMAGES_DIR / f).exists()]
        missing = len(needed_files) - len(files)
        if missing:
            print(f"  ({missing:,} files not yet extracted)", flush=True)
    else:
        files = sorted(IMAGES_DIR.glob("*.*"))

    print(f"Found {len(files):,} files to upload", flush=True)

    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {args.limit:,}", flush=True)

    if not files:
        print("Nothing to upload!", flush=True)
        return

    if args.dry_run:
        for f in files[:5]:
            print(f"  Would upload: {f.name} -> {get_r2_key(f.name)}", flush=True)
        print(f"  ... and {len(files)-5:,} more", flush=True)
        return

    print(f"Starting upload with {args.workers} concurrent connections...", flush=True)
    asyncio.run(run(files, args.workers))


if __name__ == "__main__":
    main()
