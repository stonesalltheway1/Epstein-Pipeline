"""FEC data quality: clean false positives, enrich candidate names, link unmatched donors."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from pathlib import Path

import httpx
import psycopg
from rich.console import Console
from rich.table import Table

console = Console()

_API_BASE = "https://api.open.fec.gov/v1"

# ---------------------------------------------------------------------------
# Task 1: Clean false positives
# ---------------------------------------------------------------------------

_FALSE_POSITIVE_PATTERNS = [
    # Title + single surname ONLY: "Mr. Johnson", "MR. DAVIS", "Ms. George"
    # Must be exactly title + one word (not "Mr. Jack Goldberger")
    re.compile(r'^(?:Mr|Mrs|Ms|Miss|MR|MRS|MS)\.?\s+[A-Za-z]+$'),
    # Redacted/generic markers
    re.compile(r'^\(b\)'),
    # Single word only
    re.compile(r'^[A-Za-z]+$'),
]

# Explicit blocklist of known false positives found in the data
_FALSE_POSITIVE_NAMES = frozenset({
    "mr. johnson", "mr. davis", "mr. jones", "mr. perry", "mr. berry",
    "mr. leopold", "ms. davis", "ms. jane", "ms. maxwell", "ms. mccarthy",
    "ms. brune", "mr. gair", "mr. rotert", "mr. schoeman", "mr. daugerdas",
    "mr. benhamou", "mr. berke", "mr. nardello", "earl iv",
})


def _is_false_positive(name: str) -> bool:
    """Check if a person_name is a false positive (generic/non-identifiable)."""
    name = name.strip()
    if not name or len(name) < 4:
        return True
    if name.lower() in _FALSE_POSITIVE_NAMES:
        return True
    for pat in _FALSE_POSITIVE_PATTERNS:
        if pat.match(name):
            return True
    return False


def clean_false_positives(database_url: str, *, dry_run: bool = False) -> None:
    """Remove false positive donation records from political_donations."""
    console.print("[bold]FEC Data Cleanup: False Positives[/bold]")

    with psycopg.connect(database_url, autocommit=True) as conn:
        cur = conn.cursor()

        # Get all distinct person names
        cur.execute("SELECT DISTINCT person_id, person_name FROM political_donations")
        all_donors = cur.fetchall()

        false_positives = []
        for pid, pname in all_donors:
            if _is_false_positive(pname):
                false_positives.append((pid, pname))

        if not false_positives:
            console.print("[green]No false positives found.[/green]")
            return

        console.print(f"Found [red]{len(false_positives)}[/red] false positive donors:")
        for pid, pname in false_positives[:20]:
            cur.execute(
                "SELECT COUNT(*), SUM(amount) FROM political_donations WHERE person_id = %s",
                (pid,),
            )
            cnt, total = cur.fetchone()
            console.print(f"  [yellow]{pname}[/yellow] ({pid}): {cnt} donations, ${(total or 0)/100:,.0f}")

        if len(false_positives) > 20:
            console.print(f"  ... and {len(false_positives) - 20} more")

        if dry_run:
            console.print("\n[dim]Dry run -- no changes made.[/dim]")
            return

        # Delete donations
        total_deleted = 0
        for pid, pname in false_positives:
            cur.execute("DELETE FROM political_donations WHERE person_id = %s", (pid,))
            total_deleted += cur.rowcount

            # Reset person record if it exists
            cur.execute(
                "UPDATE persons SET fec_donor = false, fec_total_donated = 0 WHERE id = %s",
                (pid,),
            )

        console.print(f"\n[green]Deleted {total_deleted} false positive donation records.[/green]")
        console.print(f"[green]Cleaned {len(false_positives)} donor entries.[/green]")


# ---------------------------------------------------------------------------
# Task 2: Enrich candidate names (3-phase: bulk CSV + batch API + name parse)
# ---------------------------------------------------------------------------

# FEC bulk data URLs — committee master, candidate master, linkage
_FEC_BULK_BASE = "https://www.fec.gov/files/bulk-downloads"
_BULK_CYCLES = [2026, 2024, 2022, 2020, 2018, 2016, 2014, 2012, 2010, 2008, 2006, 2004]

# Committee name patterns that reveal candidate names
_NAME_FOR_OFFICE_RE = re.compile(
    r'^(.+?)\s+FOR\s+(?:CONGRESS|SENATE|PRESIDENT|GOVERNOR|ASSEMBLY|'
    r'STATE\s+SENATE|STATE\s+HOUSE|MAYOR|COUNCIL|JUDGE|SHERIFF|'
    r'REPRESENTATIVE|SUPERVISOR|COMMISSIONER|ATTORNEY\s+GENERAL|'
    r'COMPTROLLER|TREASURER|SECRETARY\s+OF\s+STATE|LIEUTENANT\s+GOVERNOR|'
    r'U\.?S\.?\s+SENATE|U\.?S\.?\s+CONGRESS|US\s+SENATE|US\s+CONGRESS|'
    r'AMERICA|THE\s+PEOPLE|CHANGE|TEXAS|FLORIDA|CALIFORNIA|NEW\s+YORK|'
    r'OHIO|PENNSYLVANIA|VIRGINIA|MICHIGAN|GEORGIA|ARIZONA|'
    r'[A-Z]{2})\b',
    re.IGNORECASE,
)

# Office keyword detection from committee name
_OFFICE_KEYWORDS = {
    "PRESIDENT": "P", "PRESIDENTIAL": "P", "WHITE HOUSE": "P", "AMERICA": "P",
    "SENATE": "S", "SENATORIAL": "S", "U.S. SENATE": "S", "US SENATE": "S",
    "CONGRESS": "H", "CONGRESSIONAL": "H", "REPRESENTATIVE": "H",
    "U.S. CONGRESS": "H", "US CONGRESS": "H", "HOUSE": "H",
}

# Committee type -> office mapping
_CMTE_TYPE_OFFICE = {"P": "P", "S": "S", "H": "H"}


def _title_case_name(name: str) -> str:
    """Convert 'SMITH, JOHN Q' or 'JOHN SMITH' to 'John Smith'."""
    if not name:
        return ""
    # FEC uses "LAST, FIRST MIDDLE" format
    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ""
        # Remove suffixes like "/ RUNNING MATE" or "JR"
        first = re.sub(r'\s*/\s*.*$', '', first).strip()
        name = f"{first} {last}" if first else last
    return name.title()


def _parse_candidate_from_committee_name(comm_name: str) -> tuple[str, str]:
    """Try to extract candidate name and office from committee name.

    Returns (candidate_name, office_code) where office_code is P/S/H or ''.
    """
    if not comm_name:
        return "", ""

    # Detect office from keywords
    office = ""
    upper = comm_name.upper()
    for kw, code in _OFFICE_KEYWORDS.items():
        if kw in upper:
            office = code
            break

    # Try "NAME FOR OFFICE" pattern
    m = _NAME_FOR_OFFICE_RE.match(comm_name)
    if m:
        raw = m.group(1).strip()
        # Skip if the "name" part is clearly not a person name
        # (e.g., "FRIENDS OF", "CITIZENS FOR", "COMMITTEE TO ELECT")
        skip_prefixes = (
            "FRIENDS OF", "CITIZENS FOR", "COMMITTEE TO", "COMMITTEE FOR",
            "PEOPLE FOR", "AMERICANS FOR", "VOTE FOR", "TEAM",
            "VICTORY", "VOLUNTEERS FOR", "SUPPORTERS OF",
        )
        raw_upper = raw.upper()
        is_person = not any(raw_upper.startswith(p) for p in skip_prefixes)

        # "FRIENDS OF JOHN SMITH" -> extract "JOHN SMITH"
        if not is_person:
            for prefix in ("FRIENDS OF ", "CITIZENS FOR ", "COMMITTEE TO ELECT ",
                           "PEOPLE FOR ", "VOLUNTEERS FOR ", "SUPPORTERS OF "):
                if raw_upper.startswith(prefix):
                    raw = raw[len(prefix):].strip()
                    is_person = bool(raw)
                    break

        if is_person and len(raw) >= 3 and " " in raw:
            return _title_case_name(raw), office

    # Try "FRIENDS OF NAME" or "ELECT NAME" patterns
    friends_re = re.match(
        r'^(?:FRIENDS OF|CITIZENS FOR|ELECT|RE-ELECT|COMMITTEE TO (?:ELECT|RE-ELECT))\s+(.+?)(?:\s+(?:INC|LLC|PAC|COMMITTEE))?$',
        comm_name, re.IGNORECASE,
    )
    if friends_re:
        raw = friends_re.group(1).strip()
        if len(raw) >= 3 and " " in raw:
            return _title_case_name(raw), office

    return "", office


def _download_fec_bulk_file(
    client: httpx.Client,
    file_type: str,
    cycle: int,
    cache_dir: Path,
) -> list[list[str]] | None:
    """Download and parse a single FEC bulk file (CM, CN, or CCL).

    Returns list of pipe-delimited rows, or None on failure.
    """
    yy = str(cycle)[-2:]
    filename = f"{file_type}{yy}.zip"
    cache_file = cache_dir / filename

    # Use cached file if exists
    if cache_file.exists():
        raw_bytes = cache_file.read_bytes()
    else:
        url = f"{_FEC_BULK_BASE}/{cycle}/{filename}"
        try:
            resp = client.get(url, follow_redirects=True, timeout=60.0)
            if resp.status_code != 200:
                return None
            raw_bytes = resp.content
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(raw_bytes)
        except Exception:
            return None

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            # Usually one file inside
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as f:
                text = f.read().decode("latin-1", errors="replace")
                reader = csv.reader(io.StringIO(text), delimiter="|")
                return list(reader)
    except Exception:
        return None


def _build_bulk_lookup(
    client: httpx.Client,
    cache_dir: Path,
) -> dict[str, dict]:
    """Build committee_id -> {candidate_name, party, office, state} from FEC bulk files.

    Downloads Committee Master (CM), Candidate Master (CN), and
    Candidate-Committee Linkage (CCL) files across multiple election cycles.
    """
    console.print("  [dim]Phase 1: Downloading FEC bulk data files...[/dim]")

    # Step 1: Build candidate_id -> {name, party, office, state} from CN files
    candidate_lookup: dict[str, dict] = {}
    for cycle in _BULK_CYCLES:
        rows = _download_fec_bulk_file(client, "cn", cycle, cache_dir)
        if not rows:
            continue
        for row in rows:
            if len(row) < 10:
                continue
            cand_id = row[0].strip()
            if cand_id and cand_id not in candidate_lookup:
                candidate_lookup[cand_id] = {
                    "name": _title_case_name(row[1].strip()),
                    "party": row[2].strip(),
                    "office": row[5].strip(),  # P/S/H
                    "state": row[4].strip(),
                }
    console.print(f"    Candidates loaded: {len(candidate_lookup)}")

    # Step 2: Build committee_id -> candidate_id from CM files (column 15 = CAND_ID)
    cmte_to_cand: dict[str, str] = {}
    cmte_party: dict[str, str] = {}
    cmte_type: dict[str, str] = {}
    for cycle in _BULK_CYCLES:
        rows = _download_fec_bulk_file(client, "cm", cycle, cache_dir)
        if not rows:
            continue
        for row in rows:
            if len(row) < 15:
                continue
            cmte_id = row[0].strip()
            cand_id = row[14].strip() if len(row) > 14 else ""
            party = row[10].strip() if len(row) > 10 else ""
            ctype = row[9].strip() if len(row) > 9 else ""
            if cmte_id:
                if cand_id and cmte_id not in cmte_to_cand:
                    cmte_to_cand[cmte_id] = cand_id
                if party:
                    cmte_party[cmte_id] = party
                if ctype:
                    cmte_type[cmte_id] = ctype
    console.print(f"    Committee->Candidate links from CM: {len(cmte_to_cand)}")

    # Step 3: Add more links from CCL files
    ccl_added = 0
    for cycle in _BULK_CYCLES:
        rows = _download_fec_bulk_file(client, "ccl", cycle, cache_dir)
        if not rows:
            continue
        for row in rows:
            if len(row) < 4:
                continue
            cand_id = row[0].strip()
            cmte_id = row[3].strip()
            if cmte_id and cand_id and cmte_id not in cmte_to_cand:
                cmte_to_cand[cmte_id] = cand_id
                ccl_added += 1
    console.print(f"    Additional links from CCL: {ccl_added}")

    # Step 4: Resolve committee_id -> full candidate info
    result: dict[str, dict] = {}
    for cmte_id, cand_id in cmte_to_cand.items():
        cand = candidate_lookup.get(cand_id, {})
        if cand.get("name"):
            result[cmte_id] = {
                "candidate_name": cand["name"],
                "candidate_party": cand.get("party", "") or cmte_party.get(cmte_id, ""),
                "candidate_office": cand.get("office", ""),
                "candidate_state": cand.get("state", ""),
                "source": "bulk",
            }
        elif cmte_party.get(cmte_id):
            result[cmte_id] = {
                "candidate_name": "",
                "candidate_party": cmte_party[cmte_id],
                "candidate_office": _CMTE_TYPE_OFFICE.get(cmte_type.get(cmte_id, ""), ""),
                "candidate_state": "",
                "source": "bulk_party_only",
            }

    # Also add party info for committees without candidate links
    for cmte_id, party in cmte_party.items():
        if cmte_id not in result and party:
            result[cmte_id] = {
                "candidate_name": "",
                "candidate_party": party,
                "candidate_office": _CMTE_TYPE_OFFICE.get(cmte_type.get(cmte_id, ""), ""),
                "candidate_state": "",
                "source": "bulk_party_only",
            }

    console.print(f"    [green]Bulk lookup built: {len(result)} committees ({sum(1 for v in result.values() if v.get('candidate_name'))} with candidate names)[/green]")
    return result


def _batch_resolve_committees(
    client: httpx.Client,
    committee_ids: list[str],
    api_key: str,
) -> dict[str, dict]:
    """Resolve committees via batch API calls (100 IDs per request)."""
    console.print(f"  [dim]Phase 2: Batch API resolution for {len(committee_ids)} committees...[/dim]")

    result: dict[str, dict] = {}
    errors = 0
    batch_size = 100
    batches = [committee_ids[i:i + batch_size] for i in range(0, len(committee_ids), batch_size)]

    for batch_idx, batch in enumerate(batches):
        if batch_idx % 10 == 0 and batch_idx > 0:
            console.print(f"    Batch {batch_idx}/{len(batches)}...")

        try:
            # Build params with multiple committee_id values
            params: list[tuple[str, str]] = [("api_key", api_key), ("per_page", "100")]
            for cid in batch:
                params.append(("committee_id", cid))

            resp = client.get(
                f"{_API_BASE}/committees/",
                params=params,
                timeout=30.0,
            )

            if resp.status_code == 429:
                console.print(f"    [yellow]Rate limited at batch {batch_idx}, waiting 30s...[/yellow]")
                time.sleep(30.0)
                resp = client.get(f"{_API_BASE}/committees/", params=params, timeout=30.0)

            if resp.status_code != 200:
                errors += 1
                time.sleep(2.0)
                continue

            data = resp.json()
            committees_data = data.get("results", [])

            # Also need to paginate if there are more results
            total_pages = data.get("pagination", {}).get("pages", 1)
            if total_pages > 1:
                for page in range(2, min(total_pages + 1, 5)):
                    time.sleep(1.0)
                    page_params = params + [("page", str(page))]
                    page_resp = client.get(f"{_API_BASE}/committees/", params=page_params, timeout=30.0)
                    if page_resp.status_code == 200:
                        committees_data.extend(page_resp.json().get("results", []))

            for r in committees_data:
                cmte_id = r.get("committee_id", "")
                if not cmte_id:
                    continue

                info = {
                    "committee_name": r.get("name", ""),
                    "candidate_name": "",
                    "candidate_party": r.get("party", "") or "",
                    "candidate_office": "",
                    "candidate_state": "",
                    "source": "api_batch",
                }

                candidate_ids = r.get("candidate_ids", [])
                if candidate_ids:
                    # Committee has linked candidates
                    info["candidate_name"] = _title_case_name(r.get("candidate_name", "") or "")
                    info["candidate_office"] = r.get("office", "") or ""
                    info["candidate_state"] = r.get("state", "") or ""

                result[cmte_id] = info

            time.sleep(1.5)  # Respect rate limits between batches

        except Exception as e:
            errors += 1
            if errors <= 3:
                console.print(f"    [yellow]Batch error: {e}[/yellow]")
            time.sleep(5.0)

    console.print(f"    [green]API resolved: {len(result)} committees ({sum(1 for v in result.values() if v.get('candidate_name'))} with names), {errors} errors[/green]")

    # Phase 2b: batch-resolve candidate_ids that came back
    # Collect all committees that have candidate_ids but empty candidate_name
    unresolved_cand_ids: set[str] = set()
    cmte_to_cand_ids: dict[str, list[str]] = {}
    for cmte_id, info in result.items():
        if not info.get("candidate_name"):
            # Re-check the raw data for candidate_ids
            pass  # candidate_ids aren't stored in result, only resolved name

    return result


def _apply_name_parsing(
    committee_names: dict[str, str],
    existing: dict[str, dict],
) -> dict[str, dict]:
    """Phase 3: Parse candidate names from committee names for unresolved committees."""
    console.print(f"  [dim]Phase 3: Name parsing for {len(committee_names)} remaining committees...[/dim]")

    parsed = 0
    party_only = 0
    result: dict[str, dict] = {}

    for cmte_id, comm_name in committee_names.items():
        if cmte_id in existing and existing[cmte_id].get("candidate_name"):
            continue  # Already resolved

        cand_name, office = _parse_candidate_from_committee_name(comm_name)
        if cand_name:
            parsed += 1
            # Try to get party from existing partial data
            party = ""
            if cmte_id in existing:
                party = existing[cmte_id].get("candidate_party", "")
            result[cmte_id] = {
                "candidate_name": cand_name,
                "candidate_party": party,
                "candidate_office": office,
                "candidate_state": "",
                "source": "name_parse",
            }
        elif office:
            # Got office but not name — still useful
            party_only += 1

    console.print(f"    [green]Parsed {parsed} candidate names from committee names, {party_only} office-only[/green]")
    return result


def enrich_candidate_names(
    database_url: str,
    *,
    api_key: str,
    cache_dir: Path | None = None,
) -> None:
    """Resolve empty candidate_name fields using 3-phase strategy:

    Phase 1: FEC bulk CSV files (committee master, candidate master, linkage)
    Phase 2: Batch API calls (100 committees per request)
    Phase 3: Committee name parsing ("X FOR CONGRESS" patterns)
    """
    console.print("[bold]FEC Data Enrichment: Candidate Names (3-Phase)[/bold]")

    effective_cache_dir = cache_dir or Path("./output/fec/fec-cache")
    cache_path = effective_cache_dir / "committees.json"
    committee_cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            committee_cache = json.load(f)
        console.print(f"Loaded {len(committee_cache)} cached committees")

    with psycopg.connect(database_url, autocommit=True) as conn:
        cur = conn.cursor()

        # Get distinct committee IDs with missing candidate info
        cur.execute("""
            SELECT DISTINCT fec_committee_id, fec_committee_name, COUNT(*) as cnt
            FROM political_donations
            WHERE (candidate_name IS NULL OR candidate_name = '')
              AND fec_committee_id IS NOT NULL AND fec_committee_id != ''
            GROUP BY fec_committee_id, fec_committee_name
            ORDER BY cnt DESC
        """)
        committees = cur.fetchall()

        if not committees:
            console.print("[green]All candidate names already filled.[/green]")
            return

        # Filter out committees already fully cached
        uncached_ids = []
        committee_names: dict[str, str] = {}
        for cid, cname, cnt in committees:
            committee_names[cid] = cname or ""
            if cid not in committee_cache or not committee_cache[cid].get("candidate_name"):
                uncached_ids.append(cid)

        console.print(f"Committees needing resolution: [cyan]{len(committees)}[/cyan] ({len(uncached_ids)} uncached)")

        # ── Phase 1: Bulk CSV lookup ──
        bulk_lookup: dict[str, dict] = {}
        with httpx.Client(timeout=60.0) as dl_client:
            bulk_lookup = _build_bulk_lookup(dl_client, effective_cache_dir / "bulk")

        # Apply bulk results
        bulk_resolved = 0
        for cid in uncached_ids:
            if cid in bulk_lookup:
                info = bulk_lookup[cid]
                # Merge with existing cache (don't overwrite better data)
                if cid not in committee_cache or not committee_cache[cid].get("candidate_name"):
                    committee_cache[cid] = info
                    if info.get("candidate_name"):
                        bulk_resolved += 1

        console.print(f"  Phase 1 resolved: [green]{bulk_resolved}[/green] candidate names from bulk files")

        # ── Phase 2: Batch API for remaining ──
        still_uncached = [
            cid for cid in uncached_ids
            if cid not in committee_cache or not committee_cache[cid].get("candidate_name")
        ]

        if still_uncached and api_key:
            with httpx.Client(timeout=30.0) as api_client:
                api_results = _batch_resolve_committees(api_client, still_uncached, api_key)

            api_resolved = 0
            for cid, info in api_results.items():
                if cid not in committee_cache or not committee_cache[cid].get("candidate_name"):
                    # Merge: keep party from bulk if API didn't provide one
                    existing = committee_cache.get(cid, {})
                    if not info.get("candidate_party") and existing.get("candidate_party"):
                        info["candidate_party"] = existing["candidate_party"]
                    committee_cache[cid] = info
                    if info.get("candidate_name"):
                        api_resolved += 1

            console.print(f"  Phase 2 resolved: [green]{api_resolved}[/green] candidate names from API")

        # ── Phase 3: Name parsing for remaining ──
        still_unresolved = {
            cid: committee_names.get(cid, "")
            for cid in uncached_ids
            if not committee_cache.get(cid, {}).get("candidate_name") and committee_names.get(cid)
        }

        if still_unresolved:
            parsed = _apply_name_parsing(still_unresolved, committee_cache)
            for cid, info in parsed.items():
                if not committee_cache.get(cid, {}).get("candidate_name"):
                    # Merge party from earlier phases
                    existing = committee_cache.get(cid, {})
                    if not info.get("candidate_party") and existing.get("candidate_party"):
                        info["candidate_party"] = existing["candidate_party"]
                    committee_cache[cid] = info

        # ── Apply all results to DB ──
        console.print("\n  [dim]Applying results to database...[/dim]")
        resolved = 0
        updated_rows = 0
        party_updated = 0

        for cid, cname, cnt in committees:
            info = committee_cache.get(cid, {})
            if not info:
                continue

            cand_name = info.get("candidate_name", "")
            cand_party = info.get("candidate_party", "")
            cand_office = info.get("candidate_office", "")
            cand_state = info.get("candidate_state", "")
            comm_name = info.get("committee_name", "")

            if cand_name or cand_party or comm_name:
                cur.execute("""
                    UPDATE political_donations
                    SET candidate_name = CASE WHEN candidate_name IS NULL OR candidate_name = '' THEN %s ELSE candidate_name END,
                        candidate_party = CASE WHEN candidate_party IS NULL OR candidate_party = '' THEN %s ELSE candidate_party END,
                        candidate_office = CASE WHEN candidate_office IS NULL OR candidate_office = '' THEN %s ELSE candidate_office END,
                        candidate_state = CASE WHEN candidate_state IS NULL OR candidate_state = '' THEN %s ELSE candidate_state END,
                        fec_committee_name = CASE WHEN fec_committee_name IS NULL OR fec_committee_name = '' THEN %s ELSE fec_committee_name END
                    WHERE fec_committee_id = %s
                      AND (candidate_name IS NULL OR candidate_name = '')
                """, (cand_name, cand_party, cand_office, cand_state, comm_name, cid))
                updated_rows += cur.rowcount
                if cand_name:
                    resolved += 1
                elif cand_party:
                    party_updated += 1

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(committee_cache, f, indent=2)

        table = Table(title="Enrichment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Committees processed", str(len(committees)))
        table.add_row("Candidate names resolved", str(resolved))
        table.add_row("Party-only updates", str(party_updated))
        table.add_row("Donation rows updated", str(updated_rows))
        table.add_row("Cache entries", str(len(committee_cache)))
        console.print(table)


# ---------------------------------------------------------------------------
# Task 3: Link unmatched donors
# ---------------------------------------------------------------------------


# Known name variations that should be linked (short name -> full name in DB)
_KNOWN_ALIASES = {
    "alex acosta": "alexander acosta",
    "ken starr": "kenneth starr",
    "juan molyneux": "juan pablo molyneux",
    "bill gates": "bill gates",
    "al gore": "al gore",
}


def link_unmatched_donors(database_url: str, *, dry_run: bool = False) -> None:
    """Match unlinked donation records to persons in the Neon persons table."""
    console.print("[bold]FEC Data Linking: Unmatched Donors[/bold]")

    try:
        from rapidfuzz import fuzz
    except ImportError:
        console.print("[red]rapidfuzz required: pip install rapidfuzz[/red]")
        return

    with psycopg.connect(database_url, autocommit=True) as conn:
        cur = conn.cursor()

        # Get all Neon persons
        cur.execute("SELECT id, name FROM persons")
        neon_persons = {row[1].lower().strip(): row[0] for row in cur.fetchall()}
        console.print(f"Neon persons: [cyan]{len(neon_persons)}[/cyan]")

        # Get unlinked donors (person_id not in persons table)
        cur.execute("""
            SELECT DISTINCT pd.person_id, pd.person_name, COUNT(*) as cnt
            FROM political_donations pd
            LEFT JOIN persons p ON p.id = pd.person_id
            WHERE p.id IS NULL
            GROUP BY pd.person_id, pd.person_name
            ORDER BY cnt DESC
        """)
        unlinked = cur.fetchall()
        console.print(f"Unlinked donors: [cyan]{len(unlinked)}[/cyan]")

        if not unlinked:
            console.print("[green]All donors are linked.[/green]")
            return

        exact_matches = []
        fuzzy_matches = []
        no_match = []

        neon_names = list(neon_persons.keys())

        for pid, pname, cnt in unlinked:
            pname_lower = pname.lower().strip()

            # Exact match
            if pname_lower in neon_persons:
                exact_matches.append((pid, pname, neon_persons[pname_lower], cnt))
                continue

            # Known alias match
            if pname_lower in _KNOWN_ALIASES:
                alias_target = _KNOWN_ALIASES[pname_lower]
                if alias_target in neon_persons:
                    exact_matches.append((pid, pname, neon_persons[alias_target], cnt))
                    continue

            # Fuzzy match
            best_score = 0
            best_match = None
            for nname in neon_names:
                score = fuzz.token_sort_ratio(pname_lower, nname)
                if score > best_score:
                    best_score = score
                    best_match = nname

            if best_score >= 95:
                exact_matches.append((pid, pname, neon_persons[best_match], cnt))
            elif best_score >= 80:
                fuzzy_matches.append((pid, pname, neon_persons[best_match], best_match, best_score, cnt))
            else:
                no_match.append((pid, pname, cnt))

        console.print(f"\n  Exact/high matches (auto-link): [green]{len(exact_matches)}[/green]")
        console.print(f"  Fuzzy matches (review): [yellow]{len(fuzzy_matches)}[/yellow]")
        console.print(f"  No match: [dim]{len(no_match)}[/dim]")

        if fuzzy_matches:
            console.print("\n[yellow]Fuzzy matches for review:[/yellow]")
            for pid, pname, neon_id, neon_name, score, cnt in fuzzy_matches[:15]:
                console.print(f"  {pname} -> {neon_name} (score={score:.0f}, {cnt} donations)")

        if dry_run:
            console.print("\n[dim]Dry run -- no changes made.[/dim]")
            return

        # Apply exact matches
        linked = 0
        for old_pid, pname, new_pid, cnt in exact_matches:
            try:
                # Check for conflict
                cur.execute(
                    "SELECT COUNT(*) FROM political_donations WHERE person_id = %s",
                    (new_pid,),
                )
                existing = cur.fetchone()[0]
                if existing > 0:
                    # Merge: delete old, the target already has records
                    cur.execute(
                        "DELETE FROM political_donations WHERE person_id = %s",
                        (old_pid,),
                    )
                else:
                    cur.execute(
                        "UPDATE political_donations SET person_id = %s WHERE person_id = %s",
                        (new_pid, old_pid),
                    )
                linked += 1
            except Exception as e:
                console.print(f"  [yellow]Error linking {pname}: {e}[/yellow]")

        # Update fec_donor flags
        cur.execute("""
            UPDATE persons SET fec_donor = true, fec_checked_at = NOW(),
                fec_total_donated = COALESCE(
                    (SELECT SUM(amount) FROM political_donations WHERE person_id = persons.id), 0
                )
            WHERE id IN (SELECT DISTINCT person_id FROM political_donations)
        """)

        console.print(f"\n[green]Linked {linked} donors to Neon persons.[/green]")
