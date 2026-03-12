"""AI vision enrichment for media items.

Uses Claude/GPT-4o to generate descriptions, detect faces, and link persons
to photos that currently lack metadata.

Usage:
    # Enrich ingested-media.json items missing descriptions
    python scripts/enrich-media-vision.py --source ingested --limit 100

    # Enrich DS10 extracted photos
    python scripts/enrich-media-vision.py --source ds10 --input-dir E:/epstein-ds10/images --limit 50

    # Use Anthropic (Claude) instead of OpenAI
    python scripts/enrich-media-vision.py --provider anthropic --model claude-sonnet-4-6 --limit 50
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

logger = logging.getLogger(__name__)
console = Console()

SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")

VISION_PROMPT = """Analyze this image from the Jeffrey Epstein case files. Provide:

1. **Description** (2-3 sentences): What is shown? Be factual and specific.
2. **People** (if any): List any recognizable individuals with confidence level (high/medium/low).
3. **Location** (if identifiable): Where was this taken?
4. **Content type**: One of: portrait, group_photo, property, document, surveillance, aerial, evidence, art, other
5. **Tags**: 3-5 relevant keywords

Respond in JSON format:
{
  "description": "...",
  "people": [{"name": "...", "confidence": "high|medium|low"}],
  "location": "..." or null,
  "content_type": "...",
  "tags": ["...", "..."]
}

Be conservative with people identification — only name someone if you're genuinely confident.
If the image contains nudity or explicit content, note "contains_nudity: true" but still describe the context."""


def describe_image_anthropic(image_bytes: bytes, model: str = "claude-sonnet-4-6") -> dict | None:
    """Use Anthropic Claude for vision analysis."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")

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
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Anthropic vision failed: {e}")
        return None


def describe_image_openai(image_bytes: bytes, model: str = "gpt-4o-mini") -> dict | None:
    """Use OpenAI GPT-4o for vision analysis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set[/red]")
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ],
                }],
                "max_tokens": 500,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning(f"OpenAI vision failed: {e}")
        return None


def fetch_image_bytes(url: str) -> bytes | None:
    """Download image from URL."""
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


@click.command()
@click.option("--source", type=click.Choice(["ingested", "archive", "ds10"]), default="ingested")
@click.option("--input-dir", type=click.Path(exists=True), default=None, help="For ds10: directory with images")
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic")
@click.option("--model", default=None, help="Model override")
@click.option("--limit", type=int, default=50, help="Max items to process")
@click.option("--skip-described", is_flag=True, default=True, help="Skip items that already have descriptions")
@click.option("--output", type=click.Path(), default=None, help="Output enrichment JSON")
@click.option("--dry-run", is_flag=True, help="Don't modify source files")
def main(source: str, input_dir: str | None, provider: str, model: str | None, limit: int, skip_described: bool, output: str | None, dry_run: bool):
    """Enrich media items with AI vision analysis."""
    console.print(f"[bold cyan]Media Vision Enrichment[/bold cyan]")
    console.print(f"Source: {source}, Provider: {provider}, Limit: {limit}")

    if model is None:
        model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o-mini"

    describe_fn = describe_image_anthropic if provider == "anthropic" else describe_image_openai

    enrichments = []

    if source == "ingested":
        media_path = SITE_DIR / "data" / "ingested-media.json"
        media = json.loads(media_path.read_text("utf-8"))
        console.print(f"Loaded {len(media):,} ingested media items")

        # Find items needing descriptions
        to_process = []
        for item in media:
            if item["type"] != "photo":
                continue
            if skip_described and item.get("description") and len(item["description"]) > 20:
                continue
            if not item.get("thumbnailUrl"):
                continue
            to_process.append(item)

        to_process = to_process[:limit]
        console.print(f"Processing {len(to_process)} items needing enrichment")

        with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
            task = progress.add_task("Enriching", total=len(to_process))

            for item in to_process:
                img_bytes = fetch_image_bytes(item["thumbnailUrl"])
                if not img_bytes or len(img_bytes) < 1000:
                    progress.advance(task)
                    continue

                result = describe_fn(img_bytes, model)
                if result:
                    enrichments.append({
                        "id": item["id"],
                        "title": item["title"],
                        "enrichment": result,
                    })

                    # Update item in place
                    if not dry_run:
                        if result.get("description"):
                            item["description"] = result["description"]
                        if result.get("tags"):
                            existing_tags = set(item.get("tags", []))
                            item["tags"] = list(existing_tags | set(result["tags"]))

                progress.advance(task)
                time.sleep(0.5)  # Rate limit

        if not dry_run and enrichments:
            media_path.write_text(json.dumps(media, indent=2, ensure_ascii=False), encoding="utf-8")
            console.print(f"[green]Updated {len(enrichments)} items in {media_path.name}[/green]")

    elif source == "ds10" and input_dir:
        inp = Path(input_dir)
        images = sorted(list(inp.rglob("*.png")) + list(inp.rglob("*.jpg")))[:limit]
        console.print(f"Processing {len(images)} DS10 images")

        with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
            task = progress.add_task("Enriching DS10", total=len(images))

            for img_path in images:
                img_bytes = img_path.read_bytes()
                if len(img_bytes) < 1000:
                    progress.advance(task)
                    continue

                result = describe_fn(img_bytes, model)
                if result:
                    enrichments.append({
                        "file": str(img_path),
                        "efta_id": img_path.parent.name,
                        "enrichment": result,
                    })

                progress.advance(task)
                time.sleep(0.5)

    # Save enrichment results
    out_path = Path(output) if output else Path(f"output/enrichments-{source}-{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enrichments, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[bold green]Enriched {len(enrichments)} items[/bold green]")
    console.print(f"Results: {out_path.resolve()}")

    # Summary of detected people
    all_people = []
    for e in enrichments:
        for p in e.get("enrichment", {}).get("people", []):
            all_people.append(p["name"])

    if all_people:
        from collections import Counter
        top = Counter(all_people).most_common(20)
        console.print("\n[bold]Top detected people:[/bold]")
        for name, count in top:
            console.print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
