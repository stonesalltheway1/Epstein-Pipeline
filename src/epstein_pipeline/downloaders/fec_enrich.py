"""FEC data quality: clean false positives, enrich candidate names, link unmatched donors."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import psycopg
from rich.console import Console
from rich.table import Table

console = Console()

_API_BASE = "https://api.open.fec.gov/v1"
_RATE_LIMIT_DELAY = 0.55

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
    # "EARL IV", "TOWN DRIVER" etc. — ALL CAPS 2-word but not real names
    # Don't catch this — too many false hits on real names in ALL CAPS
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
            console.print("\n[dim]Dry run — no changes made.[/dim]")
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
# Task 2: Enrich candidate names from committee API
# ---------------------------------------------------------------------------


def enrich_candidate_names(
    database_url: str,
    *,
    api_key: str,
    cache_dir: Path | None = None,
) -> None:
    """Resolve empty candidate_name fields via FEC committee API."""
    console.print("[bold]FEC Data Enrichment: Candidate Names[/bold]")

    cache_path = (cache_dir or Path("./output/fec/fec-cache")) / "committees.json"
    committee_cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            committee_cache = json.load(f)
        console.print(f"Loaded {len(committee_cache)} cached committees")

    with psycopg.connect(database_url, autocommit=True) as conn:
        cur = conn.cursor()

        # Get distinct committee IDs with missing candidate info
        cur.execute("""
            SELECT DISTINCT fec_committee_id, COUNT(*) as cnt
            FROM political_donations
            WHERE (candidate_name IS NULL OR candidate_name = '')
              AND fec_committee_id IS NOT NULL AND fec_committee_id != ''
            GROUP BY fec_committee_id
            ORDER BY cnt DESC
        """)
        committees = cur.fetchall()

        console.print(f"Committees needing resolution: [cyan]{len(committees)}[/cyan]")

        if not committees:
            console.print("[green]All candidate names already filled.[/green]")
            return

        resolved = 0
        updated_rows = 0
        errors = 0

        with httpx.Client(timeout=10.0) as client:
            for i, (cid, cnt) in enumerate(committees):
                if i % 50 == 0 and i > 0:
                    console.print(f"  Progress: {i}/{len(committees)} committees...")
                    # Save cache periodically
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(committee_cache, f, indent=2)

                # Check cache first
                if cid in committee_cache and committee_cache[cid].get("candidate_name"):
                    info = committee_cache[cid]
                else:
                    # Call FEC API
                    try:
                        resp = client.get(
                            f"{_API_BASE}/committee/{cid}/",
                            params={"api_key": api_key},
                        )
                        if resp.status_code == 429:
                            # Rate limited — back off and retry once
                            time.sleep(10.0)
                            resp = client.get(
                                f"{_API_BASE}/committee/{cid}/",
                                params={"api_key": api_key},
                            )
                        resp.raise_for_status()
                        data = resp.json()
                        results = data.get("results", [])

                        info = {}
                        if results:
                            r = results[0]
                            info = {
                                "committee_name": r.get("name", ""),
                                "candidate_name": "",
                                "candidate_party": r.get("party", "") or "",
                                "candidate_office": "",
                                "candidate_state": "",
                            }
                            # Many committees have candidate info directly
                            candidate_ids = r.get("candidate_ids", [])
                            if candidate_ids:
                                info["candidate_name"] = r.get("candidate_name", "") or ""
                                info["candidate_office"] = r.get("office", "") or ""
                                info["candidate_state"] = r.get("state", "") or ""

                        committee_cache[cid] = info
                        time.sleep(0.6)  # Slightly slower for bulk enrichment
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            console.print(f"  [yellow]Error for {cid}: {e}[/yellow]")
                        committee_cache[cid] = {}
                        continue

                # Update DB if we got useful info
                cname = info.get("candidate_name", "")
                cparty = info.get("candidate_party", "")
                coffice = info.get("candidate_office", "")
                cstate = info.get("candidate_state", "")
                commname = info.get("committee_name", "")

                if cname or cparty or commname:
                    cur.execute("""
                        UPDATE political_donations
                        SET candidate_name = CASE WHEN candidate_name IS NULL OR candidate_name = '' THEN %s ELSE candidate_name END,
                            candidate_party = CASE WHEN candidate_party IS NULL OR candidate_party = '' THEN %s ELSE candidate_party END,
                            candidate_office = CASE WHEN candidate_office IS NULL OR candidate_office = '' THEN %s ELSE candidate_office END,
                            candidate_state = CASE WHEN candidate_state IS NULL OR candidate_state = '' THEN %s ELSE candidate_state END,
                            fec_committee_name = CASE WHEN fec_committee_name IS NULL OR fec_committee_name = '' THEN %s ELSE fec_committee_name END
                        WHERE fec_committee_id = %s
                          AND (candidate_name IS NULL OR candidate_name = '')
                    """, (cname, cparty, coffice, cstate, commname, cid))
                    updated_rows += cur.rowcount
                    if cname:
                        resolved += 1

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(committee_cache, f, indent=2)

        table = Table(title="Enrichment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Committees queried", str(len(committees)))
        table.add_row("Candidate names resolved", str(resolved))
        table.add_row("Donation rows updated", str(updated_rows))
        table.add_row("Errors", str(errors))
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
            console.print("\n[dim]Dry run — no changes made.[/dim]")
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
