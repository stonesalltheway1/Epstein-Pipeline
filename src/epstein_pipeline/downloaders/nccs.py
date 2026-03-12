"""NCCS (National Center for Charitable Statistics) e-file data downloader.

Downloads pre-parsed IRS Form 990 data from NCCS S3 buckets.
Used to backfill financial summaries for 2013-2018 that ProPublica
lacks extracted data for (the "PDF-only gap years").

Tables used:
  F9-P01-T00-SUMMARY     — Financial summary (revenue, assets, expenses, grants)
  F9-P07-T00-DIR-TRUST-KEY — Officer/trustee aggregate compensation
  SJ-P01-T00-COMPENSATION — Schedule J compensation details

Data source: https://nccs.urban.org/nccs/datasets/efile/
License: Public domain (IRS data)
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from epstein_pipeline.models.nonprofit import Nonprofit990Filing

console = Console()

_S3_BASE = "https://nccs-efile.s3.us-east-1.amazonaws.com/parsed"

# Column mappings from NCCS field names to our Nonprofit990Filing fields
_SUMMARY_FIELD_MAP = {
    "F9_01_REV_TOT_CY": "total_revenue",
    "F9_01_EXP_TOT_CY": "total_expenses",
    "F9_01_NAFB_ASSET_TOT_EOY": "total_assets",
    "F9_01_NAFB_LIAB_TOT_EOY": "total_liabilities",
    "F9_01_EXP_GRANT_SIMILAR_CY": "grants_paid",
    "F9_01_REV_CONTR_TOT_CY": "contributions_received",
    "F9_01_EXP_SAL_ETC_CY": "officer_comp_total",  # Approximation: total salaries
}

# Compensation table fields
_COMP_FIELD_MAP = {
    "F9_07_COMP_DTK_COMP_ORG_TOT": "comp_from_org",
    "F9_07_COMP_DTK_COMP_RLTD_TOT": "comp_from_related",
    "F9_07_COMP_DTK_COMP_OTH_TOT": "comp_other",
}


def _format_ein(ein: str) -> str:
    """Normalize EIN to 9-digit string without dashes."""
    return ein.replace("-", "").strip().zfill(9)


def download_nccs_financials(
    target_eins: list[str],
    years: range | list[int] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, list[Nonprofit990Filing]]:
    """Download NCCS financial summaries for target EINs.

    Streams large CSV files (~185MB each) and filters to only target EINs
    to avoid downloading full datasets into memory.

    Args:
        target_eins: List of EINs (with or without dashes) to fetch.
        years: Year range to cover. Defaults to 2013-2018.
        cache_dir: Optional cache directory for downloaded data.

    Returns:
        Dict mapping EIN → list of Nonprofit990Filing with financial data.
    """
    if years is None:
        years = range(2013, 2019)  # 2013-2018 inclusive

    normalized = {_format_ein(e) for e in target_eins}
    results: dict[str, list[Nonprofit990Filing]] = {}
    cache_file = cache_dir / "nccs_financials.json" if cache_dir else None

    # Check cache
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            console.print(f"  [dim]Using cached NCCS data ({len(cached)} EINs)[/dim]")
            for ein, filings_data in cached.items():
                if ein in normalized:
                    results[ein] = [Nonprofit990Filing(**fd) for fd in filings_data]
            return results
        except Exception:
            pass

    console.print("[bold cyan]NCCS Historical Data Download[/bold cyan]")
    console.print(f"  Target EINs: {len(normalized)}, Years: {min(years)}-{max(years)}")

    client = httpx.Client(
        headers={"User-Agent": "EpsteinPipeline/1.0 (research)"},
        follow_redirects=True,
        timeout=120.0,
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading NCCS tables...", total=len(list(years)) * 2)

            for year in years:
                # ── Financial Summary ──
                progress.update(task, description=f"Summary {year}...")
                summary_url = f"{_S3_BASE}/F9-P01-T00-SUMMARY-{year}.csv"
                _stream_and_filter(client, summary_url, normalized, year, results, "summary")
                progress.advance(task)
                time.sleep(0.3)

                # ── Compensation Totals ──
                progress.update(task, description=f"Compensation {year}...")
                comp_url = f"{_S3_BASE}/F9-P07-T00-DIR-TRUST-KEY-{year}.csv"
                _stream_and_filter(client, comp_url, normalized, year, results, "compensation")
                progress.advance(task)
                time.sleep(0.3)

    finally:
        client.close()

    total_filings = sum(len(v) for v in results.values())
    console.print(f"  Found [cyan]{total_filings}[/cyan] filings across [cyan]{len(results)}[/cyan] EINs")

    # Save cache
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for ein, filings in results.items():
            serializable[ein] = [
                {
                    "tax_period": f.tax_period,
                    "tax_year": f.tax_year,
                    "form_type": f.form_type,
                    "total_revenue": f.total_revenue,
                    "total_expenses": f.total_expenses,
                    "total_assets": f.total_assets,
                    "total_liabilities": f.total_liabilities,
                    "grants_paid": f.grants_paid,
                    "contributions_received": f.contributions_received,
                    "officer_comp_total": f.officer_comp_total,
                    "pdf_url": f.pdf_url,
                    "irs_object_id": f.irs_object_id,
                }
                for f in filings
            ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
        console.print(f"  [dim]Cached to {cache_file}[/dim]")

    return results


def _stream_and_filter(
    client: httpx.Client,
    url: str,
    target_eins: set[str],
    year: int,
    results: dict[str, list[Nonprofit990Filing]],
    table_type: str,
) -> None:
    """Stream a large NCCS CSV and extract rows matching target EINs.

    Files are ~185MB each, so we stream line-by-line rather than
    loading into memory.
    """
    try:
        with client.stream("GET", url, timeout=120.0) as resp:
            if resp.status_code != 200:
                return

            buffer = ""
            header: list[str] | None = None
            ein_col_idx = -1
            matches = 0

            for chunk in resp.iter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    if header is None:
                        # Parse header
                        reader = csv.reader(io.StringIO(line))
                        header = next(reader)
                        header = [h.strip('"').strip() for h in header]
                        try:
                            ein_col_idx = header.index("ORG_EIN")
                        except ValueError:
                            return  # No EIN column
                        continue

                    # Fast EIN check before full CSV parse
                    # EIN is typically in position 4 (0-indexed 3)
                    parts = line.split(",", ein_col_idx + 2)
                    if len(parts) <= ein_col_idx:
                        continue
                    ein_raw = parts[ein_col_idx].strip('"').strip()
                    ein = _format_ein(ein_raw)

                    if ein not in target_eins:
                        continue

                    # Full parse for matching row
                    reader = csv.reader(io.StringIO(line))
                    try:
                        values = next(reader)
                    except StopIteration:
                        continue

                    row = dict(zip(header, values))
                    matches += 1

                    if table_type == "summary":
                        _merge_summary(ein, year, row, results)
                    elif table_type == "compensation":
                        _merge_compensation(ein, year, row, results)

            if matches > 0:
                console.print(f"    [dim]{table_type} {year}: {matches} matches[/dim]")

    except httpx.TimeoutException:
        console.print(f"    [yellow]Timeout downloading {url}[/yellow]")
    except Exception as e:
        console.print(f"    [yellow]Error: {e}[/yellow]")


def _safe_int(val: str) -> int:
    """Parse a string to int, handling NA/empty/floats."""
    if not val or val == "NA" or val == '""':
        return 0
    try:
        return int(float(val.strip('"')))
    except (ValueError, TypeError):
        return 0


def _merge_summary(
    ein: str, year: int, row: dict[str, str],
    results: dict[str, list[Nonprofit990Filing]],
) -> None:
    """Merge NCCS summary row into results, creating or updating a filing."""
    # Find or create filing for this EIN+year
    filings = results.setdefault(ein, [])
    existing = next((f for f in filings if f.tax_year == year), None)

    tax_period = str(row.get("TAX_YEAR", year)) + "12"  # Approximate to December
    form_type = row.get("RETURN_TYPE", "990")
    object_id = row.get("OBJECTID", "")

    revenue = _safe_int(row.get("F9_01_REV_TOT_CY", "0"))
    expenses = _safe_int(row.get("F9_01_EXP_TOT_CY", "0"))
    assets = _safe_int(row.get("F9_01_NAFB_ASSET_TOT_EOY", "0"))
    liabilities = _safe_int(row.get("F9_01_NAFB_LIAB_TOT_EOY", "0"))
    grants = _safe_int(row.get("F9_01_EXP_GRANT_SIMILAR_CY", "0"))
    contributions = _safe_int(row.get("F9_01_REV_CONTR_TOT_CY", "0"))
    salaries = _safe_int(row.get("F9_01_EXP_SAL_ETC_CY", "0"))

    if existing:
        # Only fill in zero fields (don't overwrite ProPublica data)
        if existing.total_revenue == 0:
            existing.total_revenue = revenue
        if existing.total_expenses == 0:
            existing.total_expenses = expenses
        if existing.total_assets == 0:
            existing.total_assets = assets
        if existing.total_liabilities == 0:
            existing.total_liabilities = liabilities
        if existing.grants_paid == 0:
            existing.grants_paid = grants
        if existing.contributions_received == 0:
            existing.contributions_received = contributions
        if existing.officer_comp_total == 0:
            existing.officer_comp_total = salaries
        if not existing.irs_object_id:
            existing.irs_object_id = object_id
    else:
        filings.append(
            Nonprofit990Filing(
                tax_period=tax_period,
                tax_year=year,
                form_type=form_type,
                total_revenue=revenue,
                total_expenses=expenses,
                total_assets=assets,
                total_liabilities=liabilities,
                grants_paid=grants,
                contributions_received=contributions,
                officer_comp_total=salaries,
                irs_object_id=object_id,
            )
        )


def _merge_compensation(
    ein: str, year: int, row: dict[str, str],
    results: dict[str, list[Nonprofit990Filing]],
) -> None:
    """Merge compensation totals from Part VII into existing filing."""
    filings = results.get(ein, [])
    existing = next((f for f in filings if f.tax_year == year), None)
    if not existing:
        return

    comp_org = _safe_int(row.get("F9_07_COMP_DTK_COMP_ORG_TOT", "0"))
    comp_related = _safe_int(row.get("F9_07_COMP_DTK_COMP_RLTD_TOT", "0"))

    # Use the more specific officer compensation total if available
    total_comp = comp_org + comp_related
    if total_comp > 0 and (existing.officer_comp_total == 0 or total_comp > existing.officer_comp_total):
        existing.officer_comp_total = total_comp
