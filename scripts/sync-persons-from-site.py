#!/usr/bin/env python3
"""Sync persons registry from epstein-index site's data/persons.ts.

Reads the TypeScript persons file via regex, extracts fields, and writes
a clean persons-registry.json for use by the pipeline.

Usage:
    python scripts/sync-persons-from-site.py --from-site ../epstein-index/data/persons.ts
    python scripts/sync-persons-from-site.py --from-site ../epstein-index/data/persons.ts --output ./data/persons-registry.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click


def extract_persons(ts_path: Path) -> list[dict]:
    """Parse persons.ts and extract person entries via regex."""
    content = ts_path.read_text(encoding="utf-8")

    persons = []

    # Match person objects: { id: "p-NNNN", slug: "...", name: "...", ... }
    # This regex captures the full object between braces
    person_pattern = re.compile(
        r'\{\s*'
        r'id:\s*"(p-\d+)".*?'
        r'slug:\s*"([^"]+)".*?'
        r'name:\s*"([^"]+)".*?'
        r'(?:aliases:\s*\[(.*?)\].*?)?'
        r'category:\s*"([^"]+)".*?'
        r'(?:shortBio:\s*["`]([^"`]*?)["`])?',
        re.DOTALL,
    )

    for match in person_pattern.finditer(content):
        person_id = match.group(1)
        slug = match.group(2)
        name = match.group(3)
        aliases_raw = match.group(4) or ""
        category = match.group(5)
        short_bio = match.group(6) or None

        # Parse aliases from the array literal
        aliases = []
        if aliases_raw.strip():
            alias_pattern = re.compile(r'"([^"]+)"')
            aliases = alias_pattern.findall(aliases_raw)

        persons.append({
            "id": person_id,
            "slug": slug,
            "name": name,
            "aliases": aliases,
            "category": category,
            "shortBio": short_bio,
        })

    return persons


@click.command()
@click.option(
    "--from-site",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to epstein-index/data/persons.ts",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./data/persons-registry.json"),
    help="Output path for persons-registry.json",
)
def main(from_site: Path, output: Path) -> None:
    """Sync persons registry from the Epstein Exposed site."""
    click.echo(f"Reading persons from {from_site}...")

    persons = extract_persons(from_site)

    if not persons:
        click.echo("ERROR: No persons found. Check the file format.", err=True)
        sys.exit(1)

    click.echo(f"  Extracted {len(persons)} persons")

    # Check for existing registry and report diff
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        existing_ids = {p["id"] for p in existing}
        new_ids = {p["id"] for p in persons}
        added = new_ids - existing_ids
        removed = existing_ids - new_ids
        if added:
            click.echo(f"  New persons: {len(added)}")
        if removed:
            click.echo(f"  Removed persons: {len(removed)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(persons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(f"  Written to {output}")


if __name__ == "__main__":
    main()
