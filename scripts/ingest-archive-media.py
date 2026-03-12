"""Ingest new high-value audio/video items from Archive.org into the site's media gallery.

Fetches metadata from Archive.org, deduplicates against existing archive-media.json,
and appends new items with proper schema.

Usage:
    python scripts/ingest-archive-media.py [--dry-run] [--site-dir PATH]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

# ── Site media JSON path ──────────────────────────────────────────────────
DEFAULT_SITE_DIR = Path(r"C:\Users\Eric\OneDrive\Desktop\epstein-index")
ARCHIVE_MEDIA_PATH = "data/archive-media.json"

# ── Archive.org API ───────────────────────────────────────────────────────
METADATA_URL = "https://archive.org/metadata"

# ── Curated list of high-value items to ingest ────────────────────────────
# Each: (identifier, type, title_override, person_ids, tags, description_override)
# person_ids reference the site's person IDs
NEW_ITEMS = [
    # TIER 1 — Primary source depositions & testimony
    {
        "identifier": "jeffepsteindeposition",
        "type": "video",
        "title": "Jeffrey Epstein Deposition Footage",
        "personIds": ["p-001"],  # Epstein
        "tags": ["deposition", "primary-source", "court-proceedings"],
    },
    {
        "identifier": "ghislaine-maxwell-audio",
        "type": "audio",
        "title": "Ghislaine Maxwell Full Interview by Todd Blanche (2025)",
        "personIds": ["p-002"],  # Maxwell
        "tags": ["interview", "primary-source", "maxwell"],
    },
    {
        "identifier": "01-ghislaine-maxwells-deposition-02-09-2026",
        "type": "video",
        "title": "Ghislaine Maxwell Deposition (Feb 9, 2026)",
        "personIds": ["p-002"],
        "tags": ["deposition", "primary-source", "maxwell", "2026"],
    },
    # Les Wexner depositions (House Oversight, Feb 2026)
    {
        "identifier": "CSPAN_20260222_065800_House_Oversight_Cmte._Deposition_of_Billionaire_Les_Wexner_Over_Jeffrey_Epstein_Case",
        "type": "video",
        "title": "Les Wexner Deposition Part 1 — House Oversight (Feb 22, 2026)",
        "personIds": ["les-wexner"],
        "tags": ["deposition", "house-oversight", "wexner", "c-span", "2026"],
    },
    {
        "identifier": "CSPAN_20260222_080700_House_Oversight_Cmte._Deposition_of_Billionaire_Les_Wexner_Over_Jeffrey_Epstein_Case",
        "type": "video",
        "title": "Les Wexner Deposition Part 2 — House Oversight (Feb 22, 2026)",
        "personIds": ["les-wexner"],
        "tags": ["deposition", "house-oversight", "wexner", "c-span", "2026"],
    },
    {
        "identifier": "CSPAN_20260222_095900_House_Oversight_Cmte._Deposition_of_Billionaire_Les_Wexner_Over_Jeffrey_Epstein_Case",
        "type": "video",
        "title": "Les Wexner Deposition Part 3 — House Oversight (Feb 22, 2026)",
        "personIds": ["les-wexner"],
        "tags": ["deposition", "house-oversight", "wexner", "c-span", "2026"],
    },
    {
        "identifier": "CSPAN3_20260223_130400_House_Oversight_Cmte._Deposition_of_Billionaire_Les_Wexner_Over_Jeffrey_Epstein_Case",
        "type": "video",
        "title": "Les Wexner Deposition Full Replay — House Oversight (Feb 23, 2026)",
        "personIds": ["les-wexner"],
        "tags": ["deposition", "house-oversight", "wexner", "c-span", "2026"],
    },
    # Kash Patel FBI Director testimony
    {
        "identifier": "CSPAN_20250917_010100_FBI_Director_Kash_Patel_Testifies_at_Senate_Oversight_Hearing",
        "type": "video",
        "title": "FBI Director Kash Patel — Senate Oversight Testimony on Epstein Files (Sep 2025)",
        "personIds": [],
        "tags": ["testimony", "senate", "fbi", "c-span", "kash-patel", "2025"],
    },
    {
        "identifier": "CSPAN2_20250918_012900_FBI_Director_Kash_Patel_Testifies_at_House_Oversight_Hearing",
        "type": "video",
        "title": "FBI Director Kash Patel — House Oversight Testimony on Epstein Files (Sep 2025)",
        "personIds": [],
        "tags": ["testimony", "house-oversight", "fbi", "c-span", "kash-patel", "2025"],
    },
    {
        "identifier": "CSPAN2_20260220_065700_House_Oversight_Ranking_Member_Garcia_on_Les_Wexners_Financial_Support_to_Epstein",
        "type": "video",
        "title": "Rep. Garcia Press Stakeout — Wexner's Financial Support to Epstein (Feb 2026)",
        "personIds": ["les-wexner"],
        "tags": ["press-conference", "house-oversight", "wexner", "c-span", "2026"],
    },

    # TIER 2 — Key investigative journalism & trial coverage
    {
        "identifier": "epstein-documents-and-tapes",
        "type": "audio",
        "title": "Epstein Documents and Tapes — Audio Collection",
        "personIds": ["p-001"],
        "tags": ["primary-source", "audio-collection", "tapes"],
    },
    {
        "identifier": "jean-luc-brunel-60-minutes-american-girls-in-paris-1988-diane-sawyer-report",
        "type": "video",
        "title": "60 Minutes: American Girls in Paris — Diane Sawyer on Jean-Luc Brunel (1988)",
        "personIds": ["jean-luc-brunel"],
        "tags": ["investigative", "60-minutes", "cbs", "brunel", "modeling", "1988"],
    },
    {
        "identifier": "EpsteinCoverProjectVeritas",
        "type": "video",
        "title": "Leaked ABC News Hot Mic — Amy Robach: 'We Had Clinton, We Had Everything' (2019)",
        "personIds": ["bill-clinton", "p-001"],
        "tags": ["leaked", "abc-news", "media-coverup", "hot-mic", "2019"],
    },
    {
        "identifier": "2019JUL07_Perversion_Justice_Epstein_Story_Depth",
        "type": "video",
        "title": "Perversion of Justice — Miami Herald Investigation In Depth (Julie K. Brown)",
        "personIds": ["p-001"],
        "tags": ["investigative", "miami-herald", "julie-brown", "2019"],
    },
    {
        "identifier": "youtube-fORaRDiSBWk",
        "type": "video",
        "title": "Annie Farmer Testimony at Maxwell Trial (Dec 2021)",
        "personIds": ["p-002"],
        "tags": ["trial", "testimony", "maxwell-trial", "victim-testimony", "2021"],
    },
    {
        "identifier": "youtube-vUEQFWhLalg",
        "type": "video",
        "title": "Maxwell Trial: Epstein's Housekeeper Testifies (Dec 2021)",
        "personIds": ["p-002", "p-001"],
        "tags": ["trial", "testimony", "maxwell-trial", "2021"],
    },

    # TIER 3 — Drone footage & supplemental
    {
        "identifier": "epsteindrone",
        "type": "video",
        "title": "Rusty Shackleford Drone Footage — Epstein Island (65 Videos)",
        "personIds": ["p-001"],
        "locationIds": ["little-st-james"],
        "tags": ["drone", "surveillance", "little-st-james", "island", "rusty-shackleford"],
    },
    {
        "identifier": "youtube-HHemTz3u7IQ",
        "type": "video",
        "title": "Epstein Island Unseen Home Video — Early Days (2026 Release)",
        "personIds": ["p-001"],
        "locationIds": ["little-st-james"],
        "tags": ["home-video", "island", "primary-source", "2026"],
    },
    {
        "identifier": "youtube-LsOXdki3ZEw",
        "type": "video",
        "title": "The Epstein Home Party Tape — Rare Footage",
        "personIds": ["p-001"],
        "tags": ["party", "home-video", "primary-source", "rare"],
    },
    {
        "identifier": "01-trump-epsteins-friendship-02-06-2026",
        "type": "video",
        "title": "Trump & Epstein's Friendship — Documentary (Feb 2026)",
        "personIds": ["p-001", "donald-trump"],
        "tags": ["documentary", "trump", "2026"],
    },
    {
        "identifier": "audio-of-epstein-speaking-about-his-good-friend-trump",
        "type": "video",
        "title": "Audio of Epstein Speaking About 'His Good Friend Trump'",
        "personIds": ["p-001", "donald-trump"],
        "tags": ["audio-recording", "primary-source", "trump"],
    },
    {
        "identifier": "youtube-XzHj-za_6J8",
        "type": "video",
        "title": "Virginia Giuffre Deposition Excerpt — Naming Tom Pritzker",
        "personIds": ["virginia-giuffre", "tom-pritzker"],
        "tags": ["deposition", "giuffre", "pritzker", "testimony"],
    },
    {
        "identifier": "youtube-D7WG4KkL7jM",
        "type": "video",
        "title": "Law & Crime: Ghislaine Maxwell Denied Bail — Full Coverage",
        "personIds": ["p-002"],
        "tags": ["court", "bail-hearing", "maxwell", "law-and-crime"],
    },
    {
        "identifier": "youtube-UzHCFud7nhM",
        "type": "video",
        "title": "Ghislaine Maxwell Sentencing Hearing — Full Coverage",
        "personIds": ["p-002"],
        "tags": ["sentencing", "court", "maxwell", "2022"],
    },
    {
        "identifier": "the-epstein-files-07-07-2025a",
        "type": "video",
        "title": "The Epstein Files — Multi-Episode Documentary Series (2025)",
        "personIds": ["p-001"],
        "tags": ["documentary", "series", "2025"],
    },
    {
        "identifier": "CSPAN_20190712_185600_President_Trump__Outgoing_Labor_Secretary_Departure_Remarks",
        "type": "video",
        "title": "Trump & Acosta Departure Remarks — Acosta Resignation Over Epstein Plea Deal (2019)",
        "personIds": ["p-001", "donald-trump", "alexander-acosta"],
        "tags": ["press-conference", "acosta", "resignation", "c-span", "plea-deal", "2019"],
    },
    {
        "identifier": "2019Nov10_60_Minutes_Australia_Exposing_Jeffrey_Epsteins_International_Sex_Trafficking_Ring",
        "type": "video",
        "title": "60 Minutes Australia: Exposing Epstein's International Sex Trafficking Ring (2019)",
        "personIds": ["p-001"],
        "tags": ["investigative", "60-minutes", "australia", "trafficking", "2019"],
    },
    {
        "identifier": "2010DEC06_Prince_Andrew_Peeking_around_Door_Epsteins_Manhattan_Home_Young_Girls_Seen",
        "type": "video",
        "title": "Prince Andrew at Epstein's NYC Mansion — Daily Mail Footage (Dec 2010)",
        "personIds": ["prince-andrew", "p-001"],
        "locationIds": ["nyc-townhouse"],
        "tags": ["surveillance", "prince-andrew", "manhattan", "daily-mail", "2010"],
    },
    {
        "identifier": "2019OCT09_Epsteins_Female_Enablers_Named_Maxwell_Fontanilla_Groff_Espinosa",
        "type": "video",
        "title": "NBC: Epstein's Female Enablers Named — Maxwell, Groff, Espinosa (2019)",
        "personIds": ["p-002", "p-001"],
        "tags": ["news-report", "nbc", "enablers", "2019"],
    },
    {
        "identifier": "10-3-berman",
        "type": "audio",
        "title": "Geoffrey Berman (SDNY) on Battling Trump DOJ Over Epstein Case",
        "personIds": ["p-001"],
        "tags": ["interview", "sdny", "berman", "prosecution", "doj"],
    },
    {
        "identifier": "jeffrey-epstein-the-game-of-the-global-elite-full-investigative-documentary",
        "type": "video",
        "title": "Jeffrey Epstein: The Game of the Global Elite — Full Documentary",
        "personIds": ["p-001"],
        "tags": ["documentary", "full-length", "investigative"],
    },
    {
        "identifier": "epstein0",
        "type": "video",
        "title": "Jeffrey Epstein Deposition Excerpt",
        "personIds": ["p-001"],
        "tags": ["deposition", "primary-source"],
    },
    {
        "identifier": "youtube-j-P94ObSOVk",
        "type": "video",
        "title": "Epstein Personal Videos Compilation — Rare & Unseen",
        "personIds": ["p-001"],
        "tags": ["home-video", "personal", "rare", "compilation"],
    },
]


def load_existing(site_dir: Path) -> tuple[list[dict], set[str], int]:
    """Load existing archive-media.json, return items, existing IDs, and max numeric ID."""
    path = site_dir / ARCHIVE_MEDIA_PATH
    if not path.exists():
        return [], set(), 0

    items = json.loads(path.read_text("utf-8"))
    existing_ids = set()
    max_num = 0

    for item in items:
        # Track Archive.org identifiers via sourceUrl
        source_url = item.get("sourceUrl", "")
        if "archive.org/details/" in source_url:
            ident = source_url.split("archive.org/details/")[-1].split("/")[0].split("?")[0]
            existing_ids.add(ident)

        # Track max media ID number
        mid = item.get("id", "")
        if mid.startswith("m-"):
            try:
                num = int(mid[2:])
                max_num = max(max_num, num)
            except ValueError:
                pass

    return items, existing_ids, max_num


def fetch_archive_metadata(identifier: str) -> dict | None:
    """Fetch metadata for a single Archive.org item."""
    url = f"{METADATA_URL}/{identifier}"
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  WARNING: Failed to fetch {identifier}: {e}")
        return None


def build_media_item(
    item_def: dict,
    meta: dict | None,
    next_id: int,
) -> dict:
    """Build a media item in the archive-media.json schema."""
    identifier = item_def["identifier"]
    metadata = (meta or {}).get("metadata", {})

    # Extract date
    date_str = metadata.get("date", metadata.get("addeddate", ""))
    if date_str and "T" in date_str:
        date_str = date_str.split("T")[0]
    if not date_str:
        date_str = ""

    # Extract description
    desc = metadata.get("description", "")
    if isinstance(desc, list):
        desc = " ".join(desc)
    # Strip HTML tags
    import re
    desc = re.sub(r"<[^>]+>", "", desc).strip()
    if len(desc) > 500:
        desc = desc[:497] + "..."

    # Use override title or Archive.org title
    title = item_def.get("title") or metadata.get("title", identifier)

    return {
        "id": f"m-{next_id}",
        "title": title,
        "type": item_def["type"],
        "description": desc,
        "date": date_str,
        "source": "Internet Archive",
        "sourceUrl": f"https://archive.org/details/{identifier}",
        "thumbnailUrl": f"https://media.epsteinexposed.com/archive/services/img/{identifier}",
        "fullUrl": f"https://archive.org/embed/{identifier}",
        "personIds": item_def.get("personIds", []),
        "locationIds": item_def.get("locationIds", []),
        "documentIds": [],
        "tags": ["internet-archive"] + item_def.get("tags", []),
        "verified": True,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Archive.org media into the site")
    parser.add_argument("--dry-run", action="store_true", help="Don't write, just show what would be added")
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    args = parser.parse_args()

    site_dir = args.site_dir
    print(f"Site directory: {site_dir}")
    print(f"Media file: {site_dir / ARCHIVE_MEDIA_PATH}")

    # Load existing
    existing_items, existing_idents, max_id = load_existing(site_dir)
    print(f"Existing items: {len(existing_items)}, max ID: m-{max_id}")
    print(f"Known Archive.org identifiers: {len(existing_idents)}")

    # Filter to new items only
    new_defs = [d for d in NEW_ITEMS if d["identifier"] not in existing_idents]
    skipped = len(NEW_ITEMS) - len(new_defs)
    if skipped:
        print(f"Skipping {skipped} already-ingested items")

    if not new_defs:
        print("Nothing new to ingest!")
        return

    print(f"\nFetching metadata for {len(new_defs)} new items...")

    new_items = []
    next_id = max_id + 1

    for i, item_def in enumerate(new_defs):
        ident = item_def["identifier"]
        print(f"  [{i+1}/{len(new_defs)}] {ident[:70]}...")

        meta = fetch_archive_metadata(ident)
        media_item = build_media_item(item_def, meta, next_id)
        new_items.append(media_item)
        next_id += 1

        time.sleep(0.5)  # Rate limit

    print(f"\nPrepared {len(new_items)} new media items:")
    for item in new_items:
        print(f"  {item['id']} | {item['type']:6} | {item['title'][:70]}")

    if args.dry_run:
        print("\n[DRY RUN] Would write to archive-media.json")
        return

    # Append and write
    all_items = existing_items + new_items
    output_path = site_dir / ARCHIVE_MEDIA_PATH
    output_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(all_items)} total items to {output_path}")
    print(f"Added {len(new_items)} new items ({sum(1 for i in new_items if i['type']=='video')} video, {sum(1 for i in new_items if i['type']=='audio')} audio)")


if __name__ == "__main__":
    main()
