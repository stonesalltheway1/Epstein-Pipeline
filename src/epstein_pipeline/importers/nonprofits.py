"""Import IRS Form 990 nonprofit results into Neon Postgres.

Reads nonprofit-990-results.json from the downloader and writes org,
filing, officer, and grant records to Neon, then updates person records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg
from rich.console import Console
from rich.table import Table

console = Console()


def _slugify(name: str) -> str:
    """Convert org name to URL slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:120].strip("-")


def _ensure_nonprofit_tables(conn: psycopg.Connection) -> None:
    """Create nonprofit tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nonprofit_orgs (
            id VARCHAR PRIMARY KEY,
            ein VARCHAR(10) UNIQUE NOT NULL,
            name VARCHAR NOT NULL,
            slug VARCHAR UNIQUE,
            city VARCHAR,
            state VARCHAR(2),
            subsection_code VARCHAR(10),
            ntee_code VARCHAR(10),
            ruling_date VARCHAR,
            category VARCHAR(50),
            latest_assets BIGINT,
            latest_revenue BIGINT,
            latest_expenses BIGINT,
            latest_grants_paid BIGINT,
            latest_officer_comp BIGINT,
            filing_count INT DEFAULT 0,
            person_id VARCHAR,
            organization_id VARCHAR,
            propublica_url VARCHAR,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nonprofit_filings (
            id SERIAL PRIMARY KEY,
            org_id VARCHAR NOT NULL REFERENCES nonprofit_orgs(id),
            tax_period VARCHAR(6),
            tax_year INT,
            form_type VARCHAR(10),
            total_revenue BIGINT,
            total_expenses BIGINT,
            total_assets BIGINT,
            total_liabilities BIGINT,
            grants_paid BIGINT,
            contributions_received BIGINT,
            officer_comp_total BIGINT,
            pdf_url VARCHAR,
            irs_object_id VARCHAR,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(org_id, tax_period)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nonprofit_officers (
            id SERIAL PRIMARY KEY,
            org_id VARCHAR NOT NULL REFERENCES nonprofit_orgs(id),
            tax_year INT,
            person_name VARCHAR NOT NULL,
            title VARCHAR,
            hours_per_week REAL,
            compensation BIGINT DEFAULT 0,
            compensation_related BIGINT DEFAULT 0,
            other_compensation BIGINT DEFAULT 0,
            is_former BOOLEAN DEFAULT false,
            person_id VARCHAR,
            match_score REAL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(org_id, tax_year, person_name)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_np_officers_person
        ON nonprofit_officers(person_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_np_officers_name
        ON nonprofit_officers(person_name)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nonprofit_grants (
            id SERIAL PRIMARY KEY,
            org_id VARCHAR NOT NULL REFERENCES nonprofit_orgs(id),
            tax_year INT,
            recipient_name VARCHAR NOT NULL,
            recipient_ein VARCHAR,
            recipient_city VARCHAR,
            recipient_state VARCHAR(2),
            amount BIGINT NOT NULL,
            purpose VARCHAR,
            recipient_org_id VARCHAR,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(org_id, tax_year, recipient_name, amount)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_np_grants_recipient
        ON nonprofit_grants(recipient_name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_np_grants_org
        ON nonprofit_grants(org_id)
    """)

    # Add columns to persons table
    for col, dtype, default in [
        ("nonprofit_officer", "BOOLEAN", "false"),
        ("nonprofit_org_count", "INT", "0"),
        ("nonprofit_total_comp", "BIGINT", "0"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE persons ADD COLUMN IF NOT EXISTS {col} {dtype} DEFAULT {default}"
            )
        except Exception:
            pass


def import_nonprofits(
    results_path: Path,
    database_url: str,
    *,
    min_match_score: float = 0.80,
) -> None:
    """Import nonprofit 990 results into Neon Postgres.

    Args:
        results_path: Path to nonprofit-990-results.json
        database_url: Neon Postgres connection URL
        min_match_score: Minimum fuzzy match score to link officer→person
    """
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        return

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    organizations = data.get("organizations", [])
    person_matches = data.get("person_matches", [])
    metadata = data.get("metadata", {})

    console.print("[bold]Importing IRS Form 990 Nonprofit Results[/bold]")
    console.print(f"Results file: [cyan]{results_path}[/cyan]")
    console.print(f"Organizations: [cyan]{len(organizations)}[/cyan]")
    console.print(f"Person matches: [cyan]{len(person_matches)}[/cyan]")
    console.print(f"Min match score: [cyan]{min_match_score}[/cyan]")
    console.print()

    imported_orgs = 0
    imported_filings = 0
    imported_officers = 0
    imported_grants = 0
    updated_persons = 0
    errors = 0

    # Pre-compute slugs to handle duplicates (e.g., multiple "Epstein Family Foundation")
    slug_counts: dict[str, int] = {}
    org_slugs: dict[str, str] = {}  # EIN → slug
    for org_data in organizations:
        ein = org_data.get("ein", "")
        name = org_data.get("name", "")
        if not ein or not name:
            continue
        base_slug = _slugify(name)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        if slug_counts[base_slug] > 1:
            # Disambiguate with state abbreviation, or EIN suffix
            state = org_data.get("state", "").lower().strip()
            if state:
                slug = f"{base_slug}-{state}"
            else:
                slug = f"{base_slug}-{ein[-4:]}"
        else:
            slug = base_slug
        org_slugs[ein] = slug

    with psycopg.connect(database_url, autocommit=True) as conn:
        _ensure_nonprofit_tables(conn)

        # Build person slug→id lookup from Neon DB (pipeline registry IDs may differ)
        neon_person_ids: dict[str, str] = {}  # slug → Neon person id
        try:
            rows = conn.execute("SELECT id, slug FROM persons").fetchall()
            for row in rows:
                neon_person_ids[row[1]] = row[0]
            console.print(f"Loaded [cyan]{len(neon_person_ids)}[/cyan] persons from Neon for ID resolution")
        except Exception:
            pass

        # Build person match lookup: officer_name+org_ein → person match
        # Resolve pipeline person IDs to Neon IDs via slug
        match_lookup: dict[str, dict] = {}
        for pm in person_matches:
            score = pm.get("match_score", 0.0)
            if score >= min_match_score:
                key = f"{pm.get('officer_name', '').lower()}|{pm.get('org_ein', '')}"
                # Resolve person_id: try slug lookup in Neon first
                pipeline_id = pm.get("person_id", "")
                person_name = pm.get("person_name", "")
                # Generate slug from person name
                person_slug = _slugify(person_name)
                neon_id = neon_person_ids.get(person_slug, pipeline_id)
                pm["resolved_person_id"] = neon_id
                match_lookup[key] = pm

        for org_data in organizations:
            ein = org_data.get("ein", "")
            name = org_data.get("name", "")
            if not ein or not name:
                continue

            org_id = f"np-{ein}"
            slug = org_slugs.get(ein, _slugify(name))

            # Get latest filing for summary stats
            filings = org_data.get("filings", [])
            latest = filings[0] if filings else {}

            try:
                conn.execute(
                    """
                    INSERT INTO nonprofit_orgs
                        (id, ein, name, slug, city, state, subsection_code, ntee_code,
                         ruling_date, category, latest_assets, latest_revenue,
                         latest_expenses, latest_grants_paid, latest_officer_comp,
                         filing_count, propublica_url, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        slug = EXCLUDED.slug,
                        city = EXCLUDED.city,
                        state = EXCLUDED.state,
                        latest_assets = EXCLUDED.latest_assets,
                        latest_revenue = EXCLUDED.latest_revenue,
                        latest_expenses = EXCLUDED.latest_expenses,
                        latest_grants_paid = EXCLUDED.latest_grants_paid,
                        latest_officer_comp = EXCLUDED.latest_officer_comp,
                        filing_count = EXCLUDED.filing_count,
                        updated_at = NOW()
                    """,
                    (
                        org_id,
                        ein,
                        name,
                        slug,
                        org_data.get("city", ""),
                        org_data.get("state", ""),
                        org_data.get("subsection_code", ""),
                        org_data.get("ntee_code", ""),
                        org_data.get("ruling_date", ""),
                        org_data.get("category", "other"),
                        int(latest.get("total_assets", 0) or 0),
                        int(latest.get("total_revenue", 0) or 0),
                        int(latest.get("total_expenses", 0) or 0),
                        int(latest.get("grants_paid", 0) or 0),
                        int(latest.get("officer_comp_total", 0) or 0),
                        len(filings),
                        org_data.get("propublica_url", ""),
                        json.dumps({
                            "ntee_code": org_data.get("ntee_code", ""),
                            "subsection_code": org_data.get("subsection_code", ""),
                        }),
                    ),
                )
                imported_orgs += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    console.print(f"[yellow]Org error ({name}): {e}[/yellow]")
                continue

            # Import filings
            for filing in filings:
                try:
                    conn.execute(
                        """
                        INSERT INTO nonprofit_filings
                            (org_id, tax_period, tax_year, form_type,
                             total_revenue, total_expenses, total_assets,
                             total_liabilities, grants_paid, contributions_received,
                             officer_comp_total, pdf_url, irs_object_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (org_id, tax_period) DO UPDATE SET
                            total_revenue = EXCLUDED.total_revenue,
                            total_expenses = EXCLUDED.total_expenses,
                            total_assets = EXCLUDED.total_assets,
                            grants_paid = EXCLUDED.grants_paid,
                            officer_comp_total = EXCLUDED.officer_comp_total
                        """,
                        (
                            org_id,
                            filing.get("tax_period", ""),
                            int(filing.get("tax_year", 0) or 0),
                            filing.get("form_type", ""),
                            int(filing.get("total_revenue", 0) or 0),
                            int(filing.get("total_expenses", 0) or 0),
                            int(filing.get("total_assets", 0) or 0),
                            int(filing.get("total_liabilities", 0) or 0),
                            int(filing.get("grants_paid", 0) or 0),
                            int(filing.get("contributions_received", 0) or 0),
                            int(filing.get("officer_comp_total", 0) or 0),
                            filing.get("pdf_url", ""),
                            filing.get("irs_object_id", ""),
                        ),
                    )
                    imported_filings += 1
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        console.print(f"[yellow]Filing error: {e}[/yellow]")

            # Import officers
            for officer in org_data.get("all_officers", []):
                officer_name = officer.get("name", "")
                if not officer_name:
                    continue

                # Look up person match — use resolved Neon ID
                key = f"{officer_name.lower()}|{ein}"
                pm = match_lookup.get(key)
                person_id = pm.get("resolved_person_id") if pm else None
                match_score = pm.get("match_score") if pm else None

                try:
                    conn.execute(
                        """
                        INSERT INTO nonprofit_officers
                            (org_id, tax_year, person_name, title, hours_per_week,
                             compensation, compensation_related, other_compensation,
                             is_former, person_id, match_score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (org_id, tax_year, person_name) DO UPDATE SET
                            title = EXCLUDED.title,
                            compensation = EXCLUDED.compensation,
                            person_id = EXCLUDED.person_id,
                            match_score = EXCLUDED.match_score
                        """,
                        (
                            org_id,
                            int(officer.get("filing_year", 0) or 0),
                            officer_name,
                            officer.get("title", ""),
                            float(officer.get("hours_per_week", 0) or 0),
                            int(officer.get("compensation", 0) or 0),
                            int(officer.get("compensation_related", 0) or 0),
                            int(officer.get("other_compensation", 0) or 0),
                            officer.get("is_former", False),
                            person_id,
                            match_score,
                        ),
                    )
                    imported_officers += 1
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        console.print(f"[yellow]Officer error: {e}[/yellow]")

            # Import grants
            for grant in org_data.get("all_grants", []):
                recipient = grant.get("recipient_name", "")
                amount = int(grant.get("amount", 0) or 0)
                if not recipient or amount == 0:
                    continue

                try:
                    conn.execute(
                        """
                        INSERT INTO nonprofit_grants
                            (org_id, tax_year, recipient_name, recipient_ein,
                             recipient_city, recipient_state, amount, purpose)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (org_id, tax_year, recipient_name, amount) DO NOTHING
                        """,
                        (
                            org_id,
                            int(grant.get("filing_year", 0) or 0),
                            recipient,
                            grant.get("recipient_ein", ""),
                            grant.get("recipient_city", ""),
                            grant.get("recipient_state", ""),
                            amount,
                            grant.get("purpose", ""),
                        ),
                    )
                    imported_grants += 1
                except Exception as e:
                    errors += 1
                    if errors <= 15:
                        console.print(f"[yellow]Grant error: {e}[/yellow]")

        # Update person records
        console.print()
        console.print("[bold]Updating person records...[/bold]")

        # Group matches by resolved Neon person_id
        person_orgs: dict[str, dict] = {}
        for pm in person_matches:
            pid = pm.get("resolved_person_id") or pm.get("person_id", "")
            score = pm.get("match_score", 0.0)
            if not pid or score < min_match_score:
                continue
            if pid not in person_orgs:
                person_orgs[pid] = {"orgs": set(), "total_comp": 0}
            person_orgs[pid]["orgs"].add(pm.get("org_ein", ""))
            person_orgs[pid]["total_comp"] += int(pm.get("compensation", 0) or 0)

        for pid, info in person_orgs.items():
            try:
                conn.execute(
                    """
                    UPDATE persons SET
                        nonprofit_officer = true,
                        nonprofit_org_count = %s,
                        nonprofit_total_comp = %s
                    WHERE id = %s OR slug = %s
                    """,
                    (len(info["orgs"]), info["total_comp"], pid, pid),
                )
                updated_persons += 1
            except Exception:
                pass

    console.print()

    # Summary
    table = Table(title="Nonprofit 990 Import Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Organizations imported", str(imported_orgs))
    table.add_row("Filings imported", str(imported_filings))
    table.add_row("Officers imported", str(imported_officers))
    table.add_row("Grants imported", str(imported_grants))
    table.add_row("Persons updated", str(updated_persons))
    table.add_row("Errors", str(errors))
    console.print(table)

    console.print(f"\n[green]Import complete.[/green]")
