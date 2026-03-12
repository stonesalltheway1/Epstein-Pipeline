"""IRS Form 990 nonprofit cross-reference via ProPublica API + IRS bulk XML.

Phase 1: ProPublica Nonprofit Explorer API for discovery + aggregate financials.
Phase 2: IRS bulk 990 XML for officer names, grants, related orgs.
         Uses IRS index CSVs to find OBJECT_IDs, then extracts individual XMLs
         from remote ZIP files via HTTP range requests (no full ZIP download needed).
Phase 3: Person cross-reference via fuzzy matching.

ProPublica API: https://projects.propublica.org/nonprofits/api/v2/
IRS bulk data: https://www.irs.gov/charities-non-profits/form-990-series-downloads
"""

from __future__ import annotations

import csv
import io
import json
import re
import struct
import time
import zlib
from dataclasses import asdict
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

from epstein_pipeline.models.nonprofit import (
    Nonprofit990Filing,
    Nonprofit990Grant,
    Nonprofit990Officer,
    Nonprofit990Org,
    Nonprofit990RelatedOrg,
    NonprofitPersonMatch,
    NonprofitSearchResult,
)

console = Console()

# ---------------------------------------------------------------------------
# ProPublica API
# ---------------------------------------------------------------------------
_PP_BASE = "https://projects.propublica.org/nonprofits/api/v2"
_PP_SEARCH = f"{_PP_BASE}/search.json"
_PP_ORG = f"{_PP_BASE}/organizations"

# Rate limit: be polite (no documented limit, but ~1 req/s is safe)
_PP_DELAY = 1.0

# IRS bulk XML base
_IRS_XML_BASE = "https://apps.irs.gov/pub/epostcard/990/xml"

# NTEE code to category mapping
_NTEE_CATEGORIES: dict[str, str] = {
    "A": "arts",
    "B": "education",
    "C": "environment",
    "D": "animal",
    "E": "health",
    "F": "mental-health",
    "G": "disease",
    "H": "medical-research",
    "I": "crime",
    "J": "employment",
    "K": "food",
    "L": "housing",
    "M": "disaster",
    "N": "recreation",
    "O": "youth",
    "P": "human-services",
    "Q": "international",
    "R": "civil-rights",
    "S": "community",
    "T": "foundation",
    "U": "science",
    "V": "social-science",
    "W": "public-benefit",
    "X": "religion",
    "Y": "mutual-benefit",
    "Z": "unknown",
}


def _format_ein(ein: str) -> str:
    """Normalize EIN to 9 digits without hyphens."""
    return re.sub(r"\D", "", ein).zfill(9)


def _display_ein(ein: str) -> str:
    """Format EIN for display: XX-XXXXXXX."""
    clean = _format_ein(ein)
    return f"{clean[:2]}-{clean[2:]}"


def _title_case_name(name: str) -> str:
    """Normalize 990 names: 'SMITH, JOHN Q' → 'John Q Smith'."""
    if not name:
        return ""
    name = name.strip()
    # Handle "LAST, FIRST" format
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    # Title case, preserving Roman numerals and common suffixes
    words = []
    for word in name.split():
        upper = word.upper()
        if upper in ("II", "III", "IV", "JR", "SR", "MD", "PHD", "ESQ"):
            words.append(upper)
        elif len(word) <= 1:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _classify_org(ntee_code: str, name: str) -> str:
    """Classify org into broad category."""
    if ntee_code:
        prefix = ntee_code[0].upper()
        cat = _NTEE_CATEGORIES.get(prefix, "other")
        if cat in ("foundation",):
            return "foundation"
        if cat in ("education",):
            return "education"
        if cat in ("health", "medical-research", "disease", "mental-health"):
            return "health"
        if cat in ("arts",):
            return "arts"
        if cat in ("science", "social-science"):
            return "science"
        return "charity"
    # Fallback to name heuristics
    name_lower = name.lower()
    if "foundation" in name_lower or "fund" in name_lower:
        return "foundation"
    if "university" in name_lower or "school" in name_lower or "institute" in name_lower:
        return "education"
    if "hospital" in name_lower or "health" in name_lower or "medical" in name_lower:
        return "health"
    return "other"


# ---------------------------------------------------------------------------
# ProPublica helpers
# ---------------------------------------------------------------------------


def _search_propublica(
    client: httpx.Client,
    query: str,
    *,
    max_pages: int = 3,
) -> list[dict]:
    """Search ProPublica Nonprofit Explorer, return org summaries."""
    all_orgs: list[dict] = []
    seen_eins: set[str] = set()

    for page in range(max_pages):
        try:
            resp = client.get(
                _PP_SEARCH,
                params={"q": query, "page": str(page)},
                timeout=15.0,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            orgs = data.get("organizations", [])
            if not orgs:
                break

            for org in orgs:
                ein = str(org.get("ein", ""))
                if ein and ein not in seen_eins:
                    seen_eins.add(ein)
                    all_orgs.append(org)

            # Check if more pages
            total = data.get("total_results", 0)
            if len(all_orgs) >= total:
                break

        except Exception:
            break

        time.sleep(_PP_DELAY)

    return all_orgs


def _get_org_detail(
    client: httpx.Client,
    ein: str,
) -> dict | None:
    """Get ProPublica org detail including filing summaries."""
    clean_ein = _format_ein(ein)
    try:
        resp = client.get(
            f"{_PP_ORG}/{clean_ein}.json",
            timeout=15.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _parse_propublica_filings(
    org_data: dict,
    *,
    max_filings: int = 25,
) -> list[Nonprofit990Filing]:
    """Extract filing summaries from ProPublica org detail.

    Parses both filings_with_data (full financials, typically 2011-present)
    and filings_without_data (PDF-only, typically 2001-2018) to maximize
    historical coverage.
    """
    filings_data = org_data.get("filings_with_data", [])
    filings: list[Nonprofit990Filing] = []
    seen_periods: set[str] = set()

    # First: filings with full financial data
    for fd in filings_data[:max_filings]:
        tax_period = str(fd.get("tax_prd", ""))
        tax_year = int(tax_period[:4]) if len(tax_period) >= 4 else 0
        seen_periods.add(tax_period)
        filings.append(
            Nonprofit990Filing(
                tax_period=tax_period,
                tax_year=tax_year,
                form_type=fd.get("formtype_str", "") or fd.get("formtype", ""),
                total_revenue=int(fd.get("totrevenue", 0) or 0),
                total_expenses=int(fd.get("totfuncexpns", 0) or 0),
                total_assets=int(fd.get("totassetsend", 0) or 0),
                total_liabilities=int(fd.get("totliabend", 0) or 0),
                grants_paid=int(fd.get("grntstogovt", 0) or 0) + int(fd.get("grntstoindiv", 0) or 0),
                contributions_received=int(fd.get("totcntrbgfts", 0) or 0),
                officer_comp_total=int(fd.get("compnsatncurrofcr", 0) or 0),
                pdf_url=fd.get("pdf_url", ""),
            )
        )

    # Second: filings without data (PDF-only, no financials) — fill gaps
    filings_no_data = org_data.get("filings_without_data", [])
    for fd in filings_no_data:
        tax_period = str(fd.get("tax_prd", ""))
        if tax_period in seen_periods:
            continue  # Already have full data for this period
        tax_year = int(tax_period[:4]) if len(tax_period) >= 4 else 0
        if tax_year < 2013:
            continue  # Only backfill to 2013 (pre-arrest era)
        seen_periods.add(tax_period)
        filings.append(
            Nonprofit990Filing(
                tax_period=tax_period,
                tax_year=tax_year,
                form_type=fd.get("formtype_str", "") or str(fd.get("formtype", "")),
                pdf_url=fd.get("pdf_url", ""),
                # Financial fields default to 0 — PDF-only filing
            )
        )
        if len(filings) >= max_filings:
            break

    return filings


# ---------------------------------------------------------------------------
# IRS bulk XML index + remote ZIP extraction
# ---------------------------------------------------------------------------

# IRS index CSV URL pattern and ZIP base
_IRS_INDEX_URL = "https://apps.irs.gov/pub/epostcard/990/xml/{year}/index_{year}.csv"
_IRS_ZIP_URL = "https://apps.irs.gov/pub/epostcard/990/xml/{year}/{zip_name}.zip"

# Type for index entries: maps EIN → list of (object_id, zip_name, tax_period, return_type, year)
IrsIndexEntry = tuple[str, str, str, str, int]  # object_id, zip_name, tax_period, return_type, year


def _download_irs_index(
    client: httpx.Client,
    year: int,
    target_eins: set[str],
    cache_dir: Path,
) -> dict[str, list[IrsIndexEntry]]:
    """Download IRS index CSV for a year and find entries matching target EINs.

    Returns dict mapping EIN → list of (object_id, zip_name, tax_period, return_type, year).
    The index is cached locally for reuse.
    """
    index_cache = cache_dir / f"index_{year}.json"

    # Check cache
    if index_cache.exists():
        try:
            with open(index_cache, encoding="utf-8") as f:
                cached = json.load(f)
            # Filter to only target EINs
            result: dict[str, list[IrsIndexEntry]] = {}
            for ein, entries in cached.items():
                if ein in target_eins:
                    result[ein] = [tuple(e) for e in entries]
            return result
        except Exception:
            pass

    url = _IRS_INDEX_URL.format(year=year)
    console.print(f"  Downloading IRS index for {year}...")

    try:
        # Stream the CSV — these are large (50-200MB)
        result: dict[str, list[IrsIndexEntry]] = {}

        with client.stream("GET", url, timeout=120.0) as resp:
            if resp.status_code != 200:
                console.print(f"  [yellow]Index {year} not available (HTTP {resp.status_code})[/yellow]")
                return {}

            text_buffer = ""
            header_parsed = False
            col_indices: dict[str, int] = {}

            for chunk in resp.iter_text():
                text_buffer += chunk
                lines = text_buffer.split("\n")
                # Keep the last (potentially incomplete) line in buffer
                text_buffer = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue

                    if not header_parsed:
                        # Parse header
                        cols = line.split(",")
                        for i, col in enumerate(cols):
                            col_indices[col.strip()] = i
                        header_parsed = True
                        continue

                    # Parse data row — CSV with possible commas in taxpayer name
                    try:
                        parts = list(csv.reader([line]))[0]
                    except Exception:
                        parts = line.split(",")

                    ein_idx = col_indices.get("EIN", 2)
                    ein = parts[ein_idx].strip() if len(parts) > ein_idx else ""
                    ein = _format_ein(ein)

                    if ein not in target_eins:
                        continue

                    obj_idx = col_indices.get("OBJECT_ID", 8)
                    tp_idx = col_indices.get("TAX_PERIOD", 3)
                    rt_idx = col_indices.get("RETURN_TYPE", 6)

                    object_id = parts[obj_idx].strip() if len(parts) > obj_idx else ""
                    tax_period = parts[tp_idx].strip() if len(parts) > tp_idx else ""
                    return_type = parts[rt_idx].strip() if len(parts) > rt_idx else ""

                    # ZIP name column exists in 2024+ indexes (column index 9)
                    zip_name = ""
                    if len(parts) > 9:
                        candidate = parts[9].strip()
                        if candidate.startswith("20") and "TEOS" in candidate:
                            zip_name = candidate

                    if object_id:
                        entry: IrsIndexEntry = (object_id, zip_name, tax_period, return_type, year)
                        result.setdefault(ein, []).append(entry)

        # Cache the results for this year
        cache_data = {ein: [list(e) for e in entries] for ein, entries in result.items()}
        with open(index_cache, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        total_found = sum(len(v) for v in result.values())
        console.print(f"  Found [cyan]{total_found}[/cyan] filings for [cyan]{len(result)}[/cyan] EINs in {year}")
        return result

    except Exception as e:
        console.print(f"  [yellow]Error reading {year} index: {e}[/yellow]")
        return {}


def _extract_xml_from_remote_zip(
    client: httpx.Client,
    zip_url: str,
    object_id: str,
    cache_dir: Path,
) -> str:
    """Extract a single XML file from a remote ZIP using HTTP range requests.

    Downloads only the central directory (~1-2MB) + the specific file data (~5-50KB),
    avoiding downloading the full ZIP (100MB-1GB+).

    Returns the XML text, or empty string on failure.
    """
    xml_cache = cache_dir / f"{object_id}.xml"
    if xml_cache.exists():
        return xml_cache.read_text(encoding="utf-8", errors="replace")

    target_filename = f"{object_id}_public.xml"

    try:
        # Step 1: Get ZIP file size
        head = client.head(zip_url, timeout=15.0)
        if head.status_code != 200:
            return ""
        file_size = int(head.headers.get("content-length", 0))
        if file_size == 0:
            return ""

        # Step 2: Download the tail to find EOCD (End of Central Directory)
        tail_size = min(256 * 1024, file_size)
        r = client.get(
            zip_url,
            headers={"Range": f"bytes={file_size - tail_size}-{file_size - 1}"},
            timeout=30.0,
        )
        tail = r.content

        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            return ""

        eocd = tail[eocd_pos : eocd_pos + 22]
        cd_size = struct.unpack("<I", eocd[12:16])[0]
        cd_offset = struct.unpack("<I", eocd[16:20])[0]

        # Step 3: Download central directory
        r2 = client.get(
            zip_url,
            headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
            timeout=30.0,
        )
        cd_data = r2.content

        # Step 4: Find our target file in the central directory
        offset = 0
        target_info = None
        while offset < len(cd_data) - 46:
            sig = cd_data[offset : offset + 4]
            if sig != b"PK\x01\x02":
                break
            fname_len = struct.unpack("<H", cd_data[offset + 28 : offset + 30])[0]
            extra_len = struct.unpack("<H", cd_data[offset + 30 : offset + 32])[0]
            comment_len = struct.unpack("<H", cd_data[offset + 32 : offset + 34])[0]
            comp_size = struct.unpack("<I", cd_data[offset + 20 : offset + 24])[0]
            comp_method = struct.unpack("<H", cd_data[offset + 10 : offset + 12])[0]
            local_offset = struct.unpack("<I", cd_data[offset + 42 : offset + 46])[0]

            fname = cd_data[offset + 46 : offset + 46 + fname_len].decode("utf-8", errors="replace")

            if target_filename in fname:
                target_info = (comp_size, comp_method, local_offset)
                break

            offset += 46 + fname_len + extra_len + comment_len

        if not target_info:
            return ""

        comp_size, comp_method, local_offset = target_info

        # Step 5: Download the local file header + compressed data
        # Local header: 30 bytes fixed + variable filename + extra field
        header_margin = 300  # generous margin for filename + extra
        r3 = client.get(
            zip_url,
            headers={
                "Range": f"bytes={local_offset}-{local_offset + header_margin + comp_size}"
            },
            timeout=30.0,
        )
        local_data = r3.content

        # Parse local file header
        if local_data[:4] != b"PK\x03\x04":
            return ""
        local_fname_len = struct.unpack("<H", local_data[26:28])[0]
        local_extra_len = struct.unpack("<H", local_data[28:30])[0]
        data_start = 30 + local_fname_len + local_extra_len
        comp_data = local_data[data_start : data_start + comp_size]

        # Step 6: Decompress
        if comp_method == 8:  # Deflate
            xml_bytes = zlib.decompress(comp_data, -zlib.MAX_WBITS)
        elif comp_method == 0:  # Stored
            xml_bytes = comp_data
        else:
            return ""

        xml_text = xml_bytes.decode("utf-8", errors="replace")

        # Cache locally
        xml_cache.write_text(xml_text, encoding="utf-8")
        return xml_text

    except Exception as e:
        console.print(f"  [dim]ZIP extract error for {object_id}: {e}[/dim]")
        return ""


def _guess_zip_name(object_id: str, year: int) -> str:
    """Try to guess the ZIP filename when the index doesn't include it.

    For pre-2024 indexes that don't have the ZIP column, we try common patterns.
    Returns empty string if we can't determine it.
    """
    # Pre-2024 indexes don't have a ZIP column. We'd need to try each monthly ZIP.
    # Since there are typically 4-12 ZIPs per year, we'll build a small lookup.
    return ""


def _find_xml_in_year_zips(
    client: httpx.Client,
    object_id: str,
    year: int,
    cache_dir: Path,
) -> str:
    """Search through year's ZIP files to find and extract a specific XML.

    Used when the index doesn't include the ZIP filename (pre-2024).
    Caches the ZIP central directories to avoid re-downloading them.
    """
    # Cache central directory listings per ZIP
    cd_cache_dir = cache_dir / "zip-listings"
    cd_cache_dir.mkdir(exist_ok=True)

    target_filename = f"{object_id}_public.xml"

    # Try common ZIP suffixes for this year
    suffixes = []
    for month in range(1, 13):
        for letter in "ABCDE":
            suffixes.append(f"{month:02d}{letter}")

    for suffix in suffixes:
        zip_name = f"{year}_TEOS_XML_{suffix}"
        listing_cache = cd_cache_dir / f"{zip_name}.json"

        # Check if we've already cached this ZIP's file listing
        if listing_cache.exists():
            try:
                with open(listing_cache, encoding="utf-8") as f:
                    listing = json.load(f)
                if target_filename in listing.get("files", {}):
                    zip_url = _IRS_ZIP_URL.format(year=year, zip_name=zip_name)
                    return _extract_xml_from_remote_zip(client, zip_url, object_id, cache_dir)
                continue  # Not in this ZIP
            except Exception:
                pass

        # Check if ZIP exists
        zip_url = _IRS_ZIP_URL.format(year=year, zip_name=zip_name)
        try:
            head = client.head(zip_url, timeout=10.0)
            if head.status_code != 200:
                continue
        except Exception:
            continue

        file_size = int(head.headers.get("content-length", 0))
        if file_size == 0:
            continue

        # Download central directory
        try:
            tail_size = min(2 * 1024 * 1024, file_size)
            r = client.get(
                zip_url,
                headers={"Range": f"bytes={file_size - tail_size}-{file_size - 1}"},
                timeout=30.0,
            )
            tail = r.content

            eocd_pos = tail.rfind(b"PK\x05\x06")
            if eocd_pos < 0:
                continue

            eocd = tail[eocd_pos : eocd_pos + 22]
            cd_size = struct.unpack("<I", eocd[12:16])[0]
            cd_offset = struct.unpack("<I", eocd[16:20])[0]

            r2 = client.get(
                zip_url,
                headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
                timeout=30.0,
            )
            cd_data = r2.content

            # Build file listing from central directory
            files: dict[str, dict] = {}
            offset = 0
            while offset < len(cd_data) - 46:
                sig = cd_data[offset : offset + 4]
                if sig != b"PK\x01\x02":
                    break
                fname_len = struct.unpack("<H", cd_data[offset + 28 : offset + 30])[0]
                extra_len = struct.unpack("<H", cd_data[offset + 30 : offset + 32])[0]
                comment_len = struct.unpack("<H", cd_data[offset + 32 : offset + 34])[0]
                fname = cd_data[offset + 46 : offset + 46 + fname_len].decode("utf-8", errors="replace")
                # Store just the base filename for lookup
                base = fname.rsplit("/", 1)[-1] if "/" in fname else fname
                files[base] = {"offset": offset}
                offset += 46 + fname_len + extra_len + comment_len

            # Cache the listing
            with open(listing_cache, "w", encoding="utf-8") as f:
                json.dump({"zip_name": zip_name, "file_count": len(files), "files": {k: True for k in files}}, f)

            if target_filename in files:
                return _extract_xml_from_remote_zip(client, zip_url, object_id, cache_dir)

        except Exception:
            continue

    return ""


# ---------------------------------------------------------------------------
# IRS XML parsing (Phase 2 — extract officers, grants, related orgs)
# ---------------------------------------------------------------------------


def _try_parse_990_xml(
    client: httpx.Client,
    ein: str,
    object_id: str,
    cache_dir: Path,
    *,
    zip_name: str = "",
    year: int = 0,
) -> tuple[list[Nonprofit990Officer], list[Nonprofit990Grant], list[Nonprofit990RelatedOrg]]:
    """Download and parse a 990 XML filing from IRS.

    Uses remote ZIP extraction via HTTP range requests. Falls back gracefully.
    Returns (officers, grants, related_orgs).
    """
    officers: list[Nonprofit990Officer] = []
    grants: list[Nonprofit990Grant] = []
    related_orgs: list[Nonprofit990RelatedOrg] = []

    if not object_id:
        return officers, grants, related_orgs

    # Check cache first
    xml_cache = cache_dir / f"{object_id}.xml"
    xml_text = ""

    if xml_cache.exists():
        xml_text = xml_cache.read_text(encoding="utf-8", errors="replace")
    else:
        # Strategy 1: Extract from known ZIP (if zip_name provided from 2024+ index)
        if zip_name:
            zip_url = _IRS_ZIP_URL.format(year=year, zip_name=zip_name)
            xml_text = _extract_xml_from_remote_zip(client, zip_url, object_id, cache_dir)

        # Strategy 2: Search year's ZIPs (for pre-2024 indexes without zip_name column)
        if not xml_text and year:
            xml_text = _find_xml_in_year_zips(client, object_id, year, cache_dir)

    if not xml_text:
        return officers, grants, related_orgs

    # Parse with lxml if available, otherwise basic regex
    try:
        from lxml import etree

        root = etree.fromstring(xml_text.encode("utf-8"))
        # Remove namespaces for easier querying
        for elem in root.iter():
            if "}" in str(elem.tag):
                elem.tag = elem.tag.split("}", 1)[1]

        def _find_first(parent, *paths):
            """Find first matching element. Uses 'is not None' (not bool) to avoid
            lxml's FutureWarning where childless elements evaluate as False."""
            for p in paths:
                el = parent.find(p)
                if el is not None:
                    return el
            return None

        def _el_text(el) -> str:
            """Get element text safely."""
            return el.text.strip() if el is not None and el.text else ""

        # --- Officers (Form990PartVIISectionAGrp or OfficerDirTrstKeyEmplGrp) ---
        for grp_tag in [
            ".//Form990PartVIISectionAGrp",
            ".//OfficerDirTrstKeyEmplGrp",
            ".//OfficerDirTrstEmplGrp",
            ".//OfcrDirTrstKeyEmplInfoGrp",
        ]:
            for grp in root.findall(grp_tag):
                name_el = _find_first(grp, "PersonNm", "BusinessName/BusinessNameLine1Txt", "OfficerNm/PersonNm")
                name = _el_text(name_el)
                if not name:
                    continue

                title_el = _find_first(grp, "TitleTxt", "Title", "PersonTitleTxt")
                hours_el = _find_first(grp, "AverageHoursPerWeekRt", "AvgHoursPerWkDevotedToPosRt", "AverageHoursPerWeekDevotedToPosRt")
                comp_el = _find_first(grp, "ReportableCompFromOrgAmt", "CompensationAmt", "Compensation")
                comp_related_el = _find_first(grp, "ReportableCompFromRltdOrgAmt")
                other_comp_el = _find_first(grp, "OtherCompensationAmt")
                former_el = _find_first(grp, "FormerOfcrDirectorTrusteeInd")

                officers.append(
                    Nonprofit990Officer(
                        name=_title_case_name(name),
                        title=_el_text(title_el),
                        hours_per_week=float(_el_text(hours_el)) if _el_text(hours_el) else 0.0,
                        compensation=int(float(_el_text(comp_el))) if _el_text(comp_el) else 0,
                        compensation_related=int(float(_el_text(comp_related_el))) if _el_text(comp_related_el) else 0,
                        other_compensation=int(float(_el_text(other_comp_el))) if _el_text(other_comp_el) else 0,
                        is_former=_el_text(former_el) in ("true", "1", "X"),
                    )
                )

        # --- Grants paid (GrantOrContributionPdDurYrGrp / Schedule I) ---
        for grp_tag in [
            ".//GrantOrContributionPdDurYrGrp",
            ".//RecipientTable",
            ".//GrantsOtherAsstToIndivInUSGrp",
        ]:
            for grp in root.findall(grp_tag):
                recip_name_el = _find_first(
                    grp,
                    "RecipientPersonNm",
                    "RecipientBusinessName/BusinessNameLine1Txt",
                    "BusinessNameLine1",
                    "BusinessNameLine1Txt",
                )
                recip_name = _el_text(recip_name_el)
                if not recip_name:
                    continue

                ein_el = _find_first(grp, "RecipientEIN", "EINOfRecipient")
                amt_el = _find_first(grp, "Amt", "CashGrantAmt", "GrantOrContributionAmt")
                purpose_el = _find_first(grp, "PurposeOfGrantTxt", "GrantOrContributionPurposeTxt")
                city_el = _find_first(grp, "USAddress/CityNm", "RecipientUSAddress/CityNm")
                state_el = _find_first(grp, "USAddress/StateAbbreviationCd", "RecipientUSAddress/StateAbbreviationCd")

                grants.append(
                    Nonprofit990Grant(
                        recipient_name=recip_name,
                        recipient_ein=_el_text(ein_el),
                        recipient_city=_el_text(city_el),
                        recipient_state=_el_text(state_el),
                        amount=int(float(_el_text(amt_el))) if _el_text(amt_el) else 0,
                        purpose=_el_text(purpose_el),
                    )
                )

        # --- Related organizations (Schedule R) ---
        for grp in root.findall(".//IdRelatedTaxExemptOrgGrp"):
            name_el = _find_first(grp, "DisregardedEntityName/BusinessNameLine1Txt", "RelatedOrganizationName/BusinessNameLine1Txt")
            ein_el = _find_first(grp, "EIN")
            rel_el = _find_first(grp, "DirectControllingEntityName/BusinessNameLine1Txt")
            related_orgs.append(
                Nonprofit990RelatedOrg(
                    name=_el_text(name_el),
                    ein=_el_text(ein_el),
                    relationship_type="related-exempt",
                    direct_controlling_entity=_el_text(rel_el),
                )
            )

    except ImportError:
        # No lxml — fallback to regex extraction for key fields
        officers.extend(_regex_extract_officers(xml_text))
        grants.extend(_regex_extract_grants(xml_text))
    except Exception as e:
        console.print(f"[dim]XML parse warning for {object_id}: {e}[/dim]")

    return officers, grants, related_orgs


def _regex_extract_officers(xml_text: str) -> list[Nonprofit990Officer]:
    """Fallback regex extraction for officer names from XML."""
    officers: list[Nonprofit990Officer] = []
    # Match <PersonNm>NAME</PersonNm> within officer groups
    pattern = re.compile(
        r"<PersonNm>([^<]+)</PersonNm>.*?"
        r"(?:<TitleTxt>([^<]*)</TitleTxt>)?"
        r".*?(?:<ReportableCompFromOrgAmt>([^<]*)</ReportableCompFromOrgAmt>)?",
        re.DOTALL,
    )
    for m in pattern.finditer(xml_text):
        name = _title_case_name(m.group(1))
        if name:
            officers.append(
                Nonprofit990Officer(
                    name=name,
                    title=m.group(2) or "",
                    compensation=int(float(m.group(3))) if m.group(3) else 0,
                )
            )
    return officers


def _regex_extract_grants(xml_text: str) -> list[Nonprofit990Grant]:
    """Fallback regex extraction for grants from XML."""
    grants: list[Nonprofit990Grant] = []
    pattern = re.compile(
        r"<RecipientBusinessName>.*?<BusinessNameLine1Txt>([^<]+)</BusinessNameLine1Txt>.*?"
        r"(?:<CashGrantAmt>([^<]*)</CashGrantAmt>|<Amt>([^<]*)</Amt>)",
        re.DOTALL,
    )
    for m in pattern.finditer(xml_text):
        name = m.group(1).strip()
        amount = m.group(2) or m.group(3) or "0"
        if name:
            grants.append(
                Nonprofit990Grant(
                    recipient_name=name,
                    amount=int(float(amount)),
                )
            )
    return grants


# ---------------------------------------------------------------------------
# Person cross-reference
# ---------------------------------------------------------------------------


def _match_officers_to_persons(
    organizations: list[Nonprofit990Org],
    registry: dict,
    *,
    threshold: int = 85,
) -> list[NonprofitPersonMatch]:
    """Fuzzy-match officer names against persons registry."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        console.print("[yellow]rapidfuzz not installed — skipping person matching[/yellow]")
        return []

    matches: list[NonprofitPersonMatch] = []
    seen: set[tuple[str, str, int]] = set()

    # Build lookup from registry
    person_names: dict[str, dict] = {}
    if isinstance(registry, list):
        for p in registry:
            name = p.get("name", "")
            if name:
                person_names[name.lower()] = p
    elif isinstance(registry, dict):
        for pid, p in registry.items():
            if isinstance(p, dict):
                name = p.get("name", "")
                if name:
                    person_names[name.lower()] = p

    if not person_names:
        return []

    person_name_list = list(person_names.keys())

    for org in organizations:
        for officer in org.all_officers:
            officer_lower = officer.name.lower().strip()
            if not officer_lower or len(officer_lower) < 5:
                continue

            best_score = 0.0
            best_person: dict | None = None

            # Exact match first
            if officer_lower in person_names:
                best_score = 100.0
                best_person = person_names[officer_lower]
            else:
                # Fuzzy match
                for pname in person_name_list:
                    score = fuzz.token_sort_ratio(officer_lower, pname)
                    if score > best_score:
                        best_score = score
                        best_person = person_names[pname]

            if best_person and best_score >= threshold:
                key = (best_person.get("id", ""), org.ein, officer.filing_year)
                if key not in seen:
                    seen.add(key)
                    officer.match_person_id = best_person.get("id", "")
                    officer.match_score = best_score / 100.0
                    matches.append(
                        NonprofitPersonMatch(
                            person_id=best_person.get("id", ""),
                            person_name=best_person.get("name", ""),
                            officer_name=officer.name,
                            org_ein=org.ein,
                            org_name=org.name,
                            title=officer.title,
                            filing_year=officer.filing_year,
                            compensation=officer.compensation,
                            match_score=best_score / 100.0,
                        )
                    )

    return matches


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------


def download_nonprofits(
    output_dir: Path,
    *,
    persons_registry_path: Path | None = None,
    seed_eins: list[str] | None = None,
    search_terms: list[str] | None = None,
    max_filings_per_org: int = 10,
    fuzzy_threshold: int = 85,
    skip_xml: bool = False,
) -> None:
    """Cross-reference Epstein network against IRS Form 990 nonprofit data.

    Algorithm:
    1. Seed discovery: Start with known EINs + search ProPublica for names
    2. ProPublica summary: Aggregate financials per filing year
    3. IRS XML deep parse: Officers, grants, related orgs (when available)
    4. Person cross-reference: Fuzzy-match officers against persons registry

    Args:
        output_dir: Where to save results JSON
        persons_registry_path: Path to persons-registry.json
        seed_eins: Known EINs to start with
        search_terms: Additional org name searches on ProPublica
        max_filings_per_org: Years of 990 data to parse
        fuzzy_threshold: Minimum rapidfuzz score for person matching
        skip_xml: Skip IRS XML download (ProPublica only)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "990-cache"
    cache_dir.mkdir(exist_ok=True)
    xml_cache_dir = cache_dir / "xml"
    xml_cache_dir.mkdir(exist_ok=True)

    # Default seed EINs (known Epstein-linked nonprofits)
    default_eins = [
        "223496220",  # J Epstein Foundation
        "133996471",  # COUQ Foundation
        "660789697",  # Gratitude America
        "455091884",  # TerraMar Project
        "133947890",  # Debra & Leon Black Foundation
        "237320631",  # Wexner Foundation
        "311306419",  # Wexner Center Foundation
        "311318013",  # Wexner Family Charitable Fund
        "137265141",  # Dubin Family Foundation
        "474634539",  # Schwarzman Foundation (Stephen A. Schwarzman)
        "454757735",  # Schwarzman Scholars
        "850325494",  # Santa Fe Institute
    ]
    eins_to_check = list(set(seed_eins or default_eins))

    # Default search terms for discovery
    default_searches = [
        "Epstein Foundation",
        "COUQ Foundation",
        "Gratitude America",
        "TerraMar Project",
        "Wexner Foundation",
        "Leon Black Foundation",
    ]
    searches = search_terms or default_searches

    # Load persons registry for cross-reference
    registry: dict | list = {}
    reg_path = persons_registry_path or Path("./data/persons-registry.json")
    if reg_path.exists():
        with open(reg_path, encoding="utf-8") as f:
            registry = json.load(f)
        console.print(f"Loaded persons registry: [cyan]{len(registry) if isinstance(registry, (list, dict)) else 0}[/cyan] entries")
    else:
        console.print(f"[yellow]No persons registry at {reg_path} — skipping person matching[/yellow]")

    console.print("[bold]IRS Form 990 Nonprofit Cross-Reference[/bold]")
    console.print(f"Seed EINs: [cyan]{len(eins_to_check)}[/cyan]")
    console.print(f"Search terms: [cyan]{len(searches)}[/cyan]")
    console.print(f"Max filings per org: [cyan]{max_filings_per_org}[/cyan]")
    console.print(f"XML parsing: [cyan]{'disabled' if skip_xml else 'enabled'}[/cyan]")
    console.print()

    organizations: list[Nonprofit990Org] = []
    seen_eins: set[str] = set()
    errors = 0

    with httpx.Client(
        timeout=30.0,
        headers={"User-Agent": "EpsteinPipeline/1.0 (research; https://epsteinexposed.com)"},
        follow_redirects=True,
    ) as client:
        # ── Phase 1: Discovery via ProPublica ──
        console.print("[bold cyan]Phase 1: ProPublica Discovery[/bold cyan]")

        # Search by terms first (may discover additional EINs)
        for term in searches:
            console.print(f"  Searching: [dim]{term}[/dim]")
            results = _search_propublica(client, term, max_pages=2)
            for org in results:
                ein = str(org.get("ein", ""))
                if ein and ein not in seen_eins and _format_ein(ein) not in {_format_ein(e) for e in eins_to_check}:
                    # Only add if it seems relevant
                    name = org.get("name", "").lower()
                    if any(kw in name for kw in ["epstein", "couq", "gratitude", "terramar", "wexner", "black", "dubin", "schwarzman"]):
                        eins_to_check.append(ein)

        console.print(f"  Total EINs to process: [cyan]{len(eins_to_check)}[/cyan]")
        console.print()

        # ── Phase 2a: IRS Index Lookup (find OBJECT_IDs for XML download) ──
        irs_index: dict[str, list[IrsIndexEntry]] = {}  # EIN → entries
        if not skip_xml:
            console.print("[bold cyan]Phase 2a: IRS Index Lookup[/bold cyan]")
            normalized_eins = {_format_ein(e) for e in eins_to_check}
            for year in range(2024, 2012, -1):
                year_entries = _download_irs_index(
                    client, year, normalized_eins, cache_dir
                )
                for ein, entries in year_entries.items():
                    irs_index.setdefault(ein, []).extend(entries)
                time.sleep(0.5)
            total_xml_entries = sum(len(v) for v in irs_index.values())
            console.print(
                f"  Total XML filings found: [cyan]{total_xml_entries}[/cyan] "
                f"across [cyan]{len(irs_index)}[/cyan] EINs"
            )
            console.print()

        # ── Phase 2b: Detail + Filings ──
        console.print("[bold cyan]Phase 2b: Organization Details + Filings[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing orgs...", total=len(eins_to_check))

            for ein in eins_to_check:
                clean_ein = _format_ein(ein)
                if clean_ein in seen_eins:
                    progress.advance(task)
                    continue
                seen_eins.add(clean_ein)

                # Check cache — but only use if it has officers (skip stale caches from Phase 1 only runs)
                cache_file = cache_dir / f"{clean_ein}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, encoding="utf-8") as f:
                            cached = json.load(f)
                        cached_officers = cached.get("all_officers", [])
                        # Use cache only if XML was already processed OR we're skipping XML
                        if cached_officers or skip_xml or clean_ein not in irs_index:
                            org = Nonprofit990Org(**{k: v for k, v in cached.items() if k in Nonprofit990Org.__dataclass_fields__})
                            org.filings = [Nonprofit990Filing(**fd) for fd in cached.get("filings", [])]
                            org.all_officers = [Nonprofit990Officer(**od) for od in cached_officers]
                            org.all_grants = [Nonprofit990Grant(**gd) for gd in cached.get("all_grants", [])]
                            org.all_related_orgs = [Nonprofit990RelatedOrg(**rd) for rd in cached.get("all_related_orgs", [])]
                            organizations.append(org)
                            progress.update(task, description=f"[dim](cached) {org.name[:40]}[/dim]")
                            progress.advance(task)
                            continue
                    except Exception:
                        pass

                progress.update(task, description=f"Fetching {_display_ein(clean_ein)}...")

                try:
                    org_data = _get_org_detail(client, clean_ein)
                    if not org_data:
                        progress.advance(task)
                        continue

                    org_info = org_data.get("organization", {})
                    name = org_info.get("name", "") or ""
                    if not name:
                        progress.advance(task)
                        continue

                    # Parse filings from ProPublica
                    filings = _parse_propublica_filings(org_data, max_filings=max_filings_per_org)

                    # Build org
                    ntee = org_info.get("ntee_code", "") or ""
                    org = Nonprofit990Org(
                        ein=clean_ein,
                        name=name,
                        city=org_info.get("city", "") or "",
                        state=org_info.get("state", "") or "",
                        ntee_code=ntee,
                        subsection_code=str(org_info.get("subsection_code", "") or ""),
                        ruling_date=str(org_info.get("ruling_date", "") or ""),
                        category=_classify_org(ntee, name),
                        propublica_url=f"https://projects.propublica.org/nonprofits/organizations/{clean_ein}",
                        filings=filings,
                    )

                    # ── Phase 2c: IRS XML deep parse for officers + grants ──
                    if not skip_xml and clean_ein in irs_index:
                        ein_entries = irs_index[clean_ein]
                        # Limit to max_filings_per_org most recent
                        ein_entries_sorted = sorted(
                            ein_entries, key=lambda e: e[2], reverse=True  # sort by tax_period desc
                        )[:max_filings_per_org]

                        for entry in ein_entries_sorted:
                            object_id, zip_name, tax_period, return_type, idx_year = entry
                            tax_year = int(tax_period[:4]) if len(tax_period) >= 4 else 0

                            progress.update(
                                task,
                                description=f"XML {name[:25]}… {tax_year}",
                            )

                            officers, grants, related = _try_parse_990_xml(
                                client,
                                clean_ein,
                                object_id,
                                xml_cache_dir,
                                zip_name=zip_name,
                                year=idx_year,
                            )

                            # Tag with filing year
                            for o in officers:
                                o.filing_year = tax_year
                            for g in grants:
                                g.filing_year = tax_year

                            # Link to ProPublica filing if matching tax_period exists
                            for filing in filings:
                                if filing.tax_period == tax_period:
                                    filing.irs_object_id = object_id
                                    filing.officers = officers
                                    filing.grants = grants
                                    filing.related_orgs = related
                                    break

                            org.all_officers.extend(officers)
                            org.all_grants.extend(grants)
                            org.all_related_orgs.extend(related)

                    organizations.append(org)

                    # Cache org (with XML data included)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(asdict(org), f, indent=2)

                    progress.update(task, description=f"{name[:40]}")

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        console.print(f"[yellow]Error for {_display_ein(clean_ein)}: {e}[/yellow]")

                progress.advance(task)
                time.sleep(_PP_DELAY)

        # ── Phase 3: Person Cross-Reference ──
        console.print()
        console.print("[bold cyan]Phase 3: Person Cross-Reference[/bold cyan]")

        person_matches: list[NonprofitPersonMatch] = []
        if registry:
            person_matches = _match_officers_to_persons(
                organizations, registry, threshold=fuzzy_threshold
            )
            console.print(f"  Matched [cyan]{len(person_matches)}[/cyan] officers to persons in registry")
        else:
            console.print("  [dim]Skipped (no registry)[/dim]")

    # ── Save results ──
    total_officers = sum(len(o.all_officers) for o in organizations)
    total_grants = sum(len(o.all_grants) for o in organizations)
    total_filings = sum(len(o.filings) for o in organizations)

    result = NonprofitSearchResult(
        total_orgs_found=len(organizations),
        total_filings=total_filings,
        total_officers=total_officers,
        total_grants=total_grants,
        total_person_matches=len(person_matches),
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        organizations=organizations,
        person_matches=person_matches,
    )

    output_file = output_dir / "nonprofit-990-results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": "IRS Form 990 via ProPublica + IRS Bulk XML",
                    "propublica_api": _PP_BASE,
                    "total_orgs": len(organizations),
                    "total_filings": total_filings,
                    "total_officers": total_officers,
                    "total_grants": total_grants,
                    "total_person_matches": len(person_matches),
                    "checked_at": result.checked_at,
                },
                "organizations": [asdict(o) for o in organizations],
                "person_matches": [asdict(m) for m in person_matches],
            },
            f,
            indent=2,
        )

    console.print()
    console.print(f"[green]Results saved to {output_file}[/green]")
    console.print()

    # Summary table
    table = Table(title="IRS Form 990 Nonprofit Cross-Reference Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Organizations found", str(len(organizations)))
    table.add_row("Total filings", str(total_filings))
    table.add_row("Officers/Trustees extracted", str(total_officers))
    table.add_row("Grants extracted", str(total_grants))
    table.add_row("Person matches", str(len(person_matches)))
    table.add_row("Errors", str(errors))
    console.print(table)

    # Show organizations
    if organizations:
        console.print()
        org_table = Table(title="Organizations Found")
        org_table.add_column("EIN", style="dim")
        org_table.add_column("Name", style="white")
        org_table.add_column("State", style="cyan")
        org_table.add_column("Category", style="yellow")
        org_table.add_column("Filings", justify="right")
        org_table.add_column("Officers", justify="right")
        org_table.add_column("Grants", justify="right")
        org_table.add_column("Latest Assets", justify="right", style="green")

        for org in sorted(organizations, key=lambda o: o.filings[0].total_assets if o.filings else 0, reverse=True):
            latest_assets = org.filings[0].total_assets if org.filings else 0
            org_table.add_row(
                _display_ein(org.ein),
                org.name[:50],
                org.state,
                org.category,
                str(len(org.filings)),
                str(len(org.all_officers)),
                str(len(org.all_grants)),
                f"${latest_assets:,.0f}" if latest_assets else "—",
            )

        console.print(org_table)

    # Show person matches
    if person_matches:
        console.print()
        match_table = Table(title="Person Matches (Top 25)")
        match_table.add_column("Person", style="white")
        match_table.add_column("Officer Name", style="dim")
        match_table.add_column("Organization", style="cyan")
        match_table.add_column("Title", style="yellow")
        match_table.add_column("Compensation", justify="right", style="green")
        match_table.add_column("Score", justify="right")

        for pm in sorted(person_matches, key=lambda x: x.compensation, reverse=True)[:25]:
            match_table.add_row(
                pm.person_name,
                pm.officer_name,
                pm.org_name[:40],
                pm.title[:30],
                f"${pm.compensation:,.0f}" if pm.compensation else "—",
                f"{pm.match_score:.0%}",
            )

        console.print(match_table)
