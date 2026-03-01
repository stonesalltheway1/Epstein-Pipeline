"""OpenSanctions API integration for sanctions/PEP cross-referencing.

Downloads sanctions data for all persons in the Epstein database and flags
matches against OFAC SDN, EU sanctions, UN Security Council, Interpol,
PEP registries, and 100+ other datasets via the OpenSanctions API.

API docs: https://www.opensanctions.org/docs/api/
Requires: EPSTEIN_OPENSANCTIONS_API_KEY environment variable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# OpenSanctions API
# ---------------------------------------------------------------------------
_API_BASE = "https://api.opensanctions.org"
_SEARCH_ENDPOINT = f"{_API_BASE}/search/default"
_MATCH_ENDPOINT = f"{_API_BASE}/match/default"

# Rate limit: be respectful (max ~2 req/sec)
_RATE_LIMIT_DELAY = 0.5


@dataclass
class SanctionsMatch:
    """A match from OpenSanctions for a person."""

    entity_id: str
    caption: str
    schema_type: str
    score: float
    datasets: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    first_seen: str | None = None
    last_seen: str | None = None
    referents: list[str] = field(default_factory=list)


@dataclass
class PersonSanctionsResult:
    """Sanctions check result for one person."""

    person_id: str
    person_name: str
    matches: list[SanctionsMatch] = field(default_factory=list)
    best_score: float = 0.0
    is_sanctioned: bool = False
    is_pep: bool = False
    datasets: list[str] = field(default_factory=list)
    checked_at: str = ""
    error: str | None = None


def _search_person(
    client: httpx.Client,
    name: str,
    *,
    schema: str = "Person",
    limit: int = 10,
) -> list[SanctionsMatch]:
    """Search OpenSanctions for a person by name."""
    try:
        resp = client.get(
            _SEARCH_ENDPOINT,
            params={"q": name, "schema": schema, "limit": limit, "fuzzy": "true"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [
            SanctionsMatch(
                entity_id=r.get("id", ""),
                caption=r.get("caption", r.get("name", "")),
                schema_type=r.get("schema", "Unknown"),
                score=r.get("score", 0.0),
                datasets=r.get("datasets", []),
                properties=r.get("properties", {}),
                first_seen=r.get("first_seen"),
                last_seen=r.get("last_seen"),
                referents=r.get("referents", []),
            )
            for r in results
        ]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise RuntimeError(
                "OpenSanctions API key invalid or missing. "
                "Set EPSTEIN_OPENSANCTIONS_API_KEY in your environment."
            ) from e
        raise
    except Exception:
        return []


def _match_person(
    client: httpx.Client,
    name: str,
    *,
    birth_date: str | None = None,
    nationality: str | None = None,
) -> list[SanctionsMatch]:
    """Use the match endpoint for higher-quality entity matching."""
    properties: dict[str, list[str]] = {"name": [name]}
    if birth_date:
        properties["birthDate"] = [birth_date]
    if nationality:
        properties["nationality"] = [nationality]

    try:
        resp = client.post(
            _MATCH_ENDPOINT,
            params={"threshold": 0.5, "algorithm": "best"},
            json={
                "queries": {
                    "q": {
                        "schema": "Person",
                        "properties": properties,
                    }
                }
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("responses", {}).get("q", {}).get("results", [])
        return [
            SanctionsMatch(
                entity_id=r.get("id", ""),
                caption=r.get("caption", ""),
                schema_type=r.get("schema", ""),
                score=r.get("score", 0.0),
                datasets=r.get("datasets", []),
                properties=r.get("properties", {}),
                first_seen=r.get("first_seen"),
                last_seen=r.get("last_seen"),
                referents=r.get("referents", []),
            )
            for r in results
        ]
    except Exception:
        return []


def download_opensanctions(
    output_dir: Path,
    *,
    api_key: str,
    persons_registry_path: Path | None = None,
    match_threshold: float = 0.5,
    use_match_api: bool = True,
) -> None:
    """Cross-reference all persons against OpenSanctions.

    Checks every person in the registry against OFAC, EU sanctions,
    UN SC, Interpol, PEP lists, and 100+ other datasets.

    Args:
        output_dir: Where to save results JSON
        api_key: OpenSanctions API key
        persons_registry_path: Path to persons-registry.json
        match_threshold: Minimum score to consider a match (0-1)
        use_match_api: Use /match endpoint (better quality) vs /search
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load persons registry
    registry_path = persons_registry_path or Path("./data/persons-registry.json")
    if not registry_path.exists():
        console.print(f"[red]Persons registry not found at {registry_path}[/red]")
        return

    with open(registry_path, encoding="utf-8") as f:
        persons = json.load(f)

    if isinstance(persons, dict):
        persons_list = list(persons.values()) if not isinstance(next(iter(persons.values()), None), str) else [persons]
    elif isinstance(persons, list):
        persons_list = persons
    else:
        console.print("[red]Unexpected persons registry format[/red]")
        return

    console.print(f"[bold]OpenSanctions Cross-Reference[/bold]")
    console.print(f"Persons to check: [cyan]{len(persons_list)}[/cyan]")
    console.print(f"API: [cyan]{_API_BASE}[/cyan]")
    console.print(f"Match threshold: [cyan]{match_threshold}[/cyan]")
    console.print(f"Method: [cyan]{'match' if use_match_api else 'search'}[/cyan]")
    console.print()

    headers = {
        "Authorization": f"ApiKey {api_key}",
        "Accept": "application/json",
    }

    results: list[PersonSanctionsResult] = []
    total_matches = 0
    sanctioned_count = 0
    pep_count = 0

    with httpx.Client(headers=headers, timeout=30.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Checking persons...", total=len(persons_list))

            for person in persons_list:
                person_id = person.get("id", person.get("slug", ""))
                person_name = person.get("name", "")

                if not person_name or len(person_name) < 3:
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Checking {person_name[:30]}...")

                try:
                    if use_match_api:
                        matches = _match_person(client, person_name)
                    else:
                        matches = _search_person(client, person_name)

                    # Filter by threshold
                    good_matches = [m for m in matches if m.score >= match_threshold]

                    # Determine flags
                    all_datasets: set[str] = set()
                    for m in good_matches:
                        all_datasets.update(m.datasets)

                    is_sanctioned = bool(
                        all_datasets
                        & {
                            "us_ofac_sdn",
                            "eu_fsf",
                            "un_sc_sanctions",
                            "gb_hmt_sanctions",
                            "au_dfat_sanctions",
                            "ca_sema_sanctions",
                        }
                    )
                    is_pep = bool(
                        all_datasets
                        & {
                            "everypolitician",
                            "wd_peps",
                            "us_cia_world_leaders",
                            "ru_rupep",
                        }
                    )

                    result = PersonSanctionsResult(
                        person_id=person_id,
                        person_name=person_name,
                        matches=good_matches,
                        best_score=max((m.score for m in good_matches), default=0.0),
                        is_sanctioned=is_sanctioned,
                        is_pep=is_pep,
                        datasets=sorted(all_datasets),
                        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )

                    if good_matches:
                        total_matches += len(good_matches)
                    if is_sanctioned:
                        sanctioned_count += 1
                    if is_pep:
                        pep_count += 1

                    results.append(result)

                except Exception as e:
                    results.append(
                        PersonSanctionsResult(
                            person_id=person_id,
                            person_name=person_name,
                            error=str(e),
                            checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    )

                progress.advance(task)
                time.sleep(_RATE_LIMIT_DELAY)

    # Save results
    output_file = output_dir / "opensanctions-results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": "OpenSanctions API",
                    "api_url": _API_BASE,
                    "total_persons_checked": len(results),
                    "total_matches": total_matches,
                    "sanctioned_count": sanctioned_count,
                    "pep_count": pep_count,
                    "match_threshold": match_threshold,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "results": [
                    {
                        "person_id": r.person_id,
                        "person_name": r.person_name,
                        "best_score": r.best_score,
                        "is_sanctioned": r.is_sanctioned,
                        "is_pep": r.is_pep,
                        "datasets": r.datasets,
                        "matches": [
                            {
                                "entity_id": m.entity_id,
                                "caption": m.caption,
                                "schema_type": m.schema_type,
                                "score": m.score,
                                "datasets": m.datasets,
                                "first_seen": m.first_seen,
                                "last_seen": m.last_seen,
                            }
                            for m in r.matches
                        ],
                        "checked_at": r.checked_at,
                        "error": r.error,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    console.print()
    console.print(f"[green]Results saved to {output_file}[/green]")
    console.print()

    # Summary table
    table = Table(title="OpenSanctions Cross-Reference Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Persons checked", str(len(results)))
    table.add_row("Total matches", str(total_matches))
    table.add_row("Sanctioned persons", str(sanctioned_count))
    table.add_row("PEPs (Politically Exposed)", str(pep_count))
    table.add_row("Errors", str(sum(1 for r in results if r.error)))
    console.print(table)

    # Show top matches
    flagged = [r for r in results if r.matches]
    if flagged:
        console.print()
        match_table = Table(title=f"Top Matches (score >= {match_threshold})")
        match_table.add_column("Person", style="white")
        match_table.add_column("Best Match", style="cyan")
        match_table.add_column("Score", style="yellow")
        match_table.add_column("Datasets", style="dim")
        match_table.add_column("Flags", style="red")

        for r in sorted(flagged, key=lambda x: x.best_score, reverse=True)[:30]:
            best = r.matches[0] if r.matches else None
            flags = []
            if r.is_sanctioned:
                flags.append("SANCTIONED")
            if r.is_pep:
                flags.append("PEP")
            match_table.add_row(
                r.person_name,
                best.caption if best else "",
                f"{r.best_score:.2f}",
                ", ".join(r.datasets[:3]),
                " ".join(flags) if flags else "",
            )

        console.print(match_table)
