"""SEC EDGAR downloader for Epstein-network financial filings.

SEC EDGAR has a free public API (no auth needed; just a compliant User-Agent).
Docs: https://www.sec.gov/developer

Targets filings for companies in the Epstein financial network:
- Leslie Wexner / L Brands (Bath & Body Works) / Victoria's Secret
- Hyperion Partners II (Epstein-advised fund)
- Zorro (Cypress) Inc — New Mexico property holder
- Financial Trust Company — USVI entity (if publicly registered)

Pulls 10-K, 13D, 13G, 13F, 4, 8-K filings and extracts Epstein mentions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# SEC requires a compliant UA that identifies you (email preferred)
USER_AGENT = "EpsteinExposed Research research@epsteinexposed.com"

SEC_BASE = "https://www.sec.gov"
SEC_API = "https://data.sec.gov"


@dataclass
class EdgarEntity:
    """A SEC-registered entity of interest."""
    name: str
    cik: str | None = None  # if known; else we look it up by name
    tickers: tuple[str, ...] = field(default_factory=tuple)
    relevance: str = ""  # why we care (for documentation)
    filing_types: tuple[str, ...] = ("10-K", "13D", "13G", "13F-HR", "4", "8-K", "SC 13G/A", "SC 13D/A")


# Curated entity list — Epstein-adjacent or Epstein-involved public filers
TARGETS: list[EdgarEntity] = [
    # Wexner's flagship public company. Historical CIK 701985 stayed the same
    # when "The Limited" became "Limited Brands" (2002) then "L Brands" (2013)
    # then "Bath & Body Works Inc" (2021).
    EdgarEntity(
        name="Bath & Body Works Inc (formerly L Brands / Limited Brands / The Limited)",
        cik="0000701985",
        tickers=("BBWI", "LB", "LTD"),
        relevance=(
            "Leslie Wexner's flagship public holding company across 50+ years. "
            "Wexner's finances were reportedly managed by Epstein for decades; "
            "10-Ks and proxy filings may name Epstein-linked entities or "
            "Financial Trust Company (USVI)."
        ),
    ),
    # 2021 spin-off of Victoria's Secret from L Brands.
    EdgarEntity(
        name="Victoria's Secret & Co",
        cik="0001856437",
        tickers=("VSCO",),
        relevance="2021 spin-off from L Brands; Wexner legacy entity.",
    ),
    # JPMorgan Chase itself — has Epstein-relationship disclosures in various
    # filings, especially post-2023 when USVI settlements were disclosed.
    EdgarEntity(
        name="JPMorgan Chase & Co",
        cik="0000019617",
        tickers=("JPM",),
        relevance=(
            "Primary Epstein banking relationship 1998-2013. Settled with USVI "
            "($75M, Sept 2023) and class action ($290M, June 2023); 8-K and "
            "10-Q filings from 2023 onward reference these settlements."
        ),
    ),
    # Deutsche Bank — took over Epstein from JPMorgan in 2013.
    EdgarEntity(
        name="Deutsche Bank AG",
        cik="0001159508",
        tickers=("DB",),
        relevance=(
            "Epstein's primary bank 2013-2018; settled $150M OCC penalty (2020) "
            "and $75M class action (2023). Filings disclose AML deficiencies."
        ),
    ),
]


class EdgarClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request = 0.0

    def _rate_limited_get(self, url: str, params: dict | None = None) -> requests.Response:
        # SEC asks for max 10 requests/sec
        elapsed = time.time() - self._last_request
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        r = self.session.get(url, params=params, timeout=30)
        self._last_request = time.time()
        r.raise_for_status()
        return r

    def lookup_cik_by_name(self, name: str) -> str | None:
        """Use EDGAR full-text search to find a CIK by company name."""
        r = self._rate_limited_get(
            f"{SEC_BASE}/cgi-bin/browse-edgar",
            {"action": "getcompany", "company": name, "type": "", "dateb": "",
             "owner": "include", "count": "10"},
        )
        # Parse HTML for CIK — cheap regex approach is fine here
        import re
        m = re.search(r'CIK=(\d{10})', r.text)
        if m:
            return m.group(1)
        m = re.search(r'CIK=(\d+)', r.text)
        if m:
            return m.group(1).zfill(10)
        return None

    def lookup_cik_by_ticker(self, ticker: str) -> str | None:
        """Use the company_tickers.json endpoint."""
        r = self._rate_limited_get(f"{SEC_BASE}/files/company_tickers.json")
        data = r.json()
        for _idx, info in data.items():
            if info.get("ticker", "").upper() == ticker.upper():
                return str(info["cik_str"]).zfill(10)
        return None

    def get_submissions(self, cik: str) -> dict:
        """Get all recent submissions for a CIK."""
        cik_padded = cik.zfill(10)
        r = self._rate_limited_get(f"{SEC_API}/submissions/CIK{cik_padded}.json")
        return r.json()

    def download_filing(self, cik: str, accession: str, primary_doc: str,
                        output_path: Path) -> bool:
        """Download a single filing document (primary document).

        URL pattern:
            https://www.sec.gov/Archives/edgar/data/{cik-no-leading-zeros}/{accession-no-dashes}/{primary_doc}
        """
        cik_int = int(cik)
        acc_clean = accession.replace("-", "")
        url = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_doc}"
        try:
            r = self._rate_limited_get(url)
        except requests.HTTPError as e:
            logger.warning("Filing download failed: %s (%s)", url, e)
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(r.content)
        return True


def pull_entity(client: EdgarClient, entity: EdgarEntity, output_dir: Path,
                max_filings: int = 50) -> dict:
    """Pull all relevant filings for a single entity."""
    # Resolve CIK if needed
    cik = entity.cik
    if not cik:
        for ticker in entity.tickers:
            cik = client.lookup_cik_by_ticker(ticker)
            if cik:
                break
        if not cik:
            cik = client.lookup_cik_by_name(entity.name)
    if not cik:
        logger.warning("Could not resolve CIK for %s", entity.name)
        return {"name": entity.name, "found": False}

    logger.info("Entity %s: CIK=%s", entity.name, cik)
    try:
        subs = client.get_submissions(cik)
    except Exception as e:
        logger.exception("submissions fetch failed: %s", e)
        return {"name": entity.name, "cik": cik, "found": True, "error": str(e)}

    entity_dir = output_dir / f"{cik}_{entity.name.replace(' ', '_')}"
    entity_dir.mkdir(parents=True, exist_ok=True)

    # Save raw submissions metadata
    (entity_dir / "_submissions.json").write_text(
        json.dumps(subs, indent=2), encoding="utf-8")

    # Walk recent filings
    recent = subs.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    downloaded = 0
    want_types = set(entity.filing_types)
    for acc, form, date, primary in zip(accessions, forms, dates, primary_docs):
        if downloaded >= max_filings:
            break
        # Match exact or prefix (e.g. "SC 13G" matches "SC 13G/A")
        if not (form in want_types or any(form.startswith(w) for w in want_types)):
            continue
        safe_name = f"{date}_{form.replace('/', '-').replace(' ', '_')}_{acc}.htm"
        dest = entity_dir / safe_name
        if dest.exists():
            continue
        if client.download_filing(cik, acc, primary, dest):
            downloaded += 1

    logger.info("Entity %s: %d filings downloaded", entity.name, downloaded)
    return {"name": entity.name, "cik": cik, "found": True, "downloaded": downloaded,
            "output_dir": str(entity_dir)}


def pull_all(output_dir: Path, max_per_entity: int = 50) -> list[dict]:
    client = EdgarClient()
    results = []
    for ent in TARGETS:
        try:
            r = pull_entity(client, ent, output_dir, max_filings=max_per_entity)
            results.append(r)
        except Exception as e:
            logger.exception("Entity %s failed: %s", ent.name, e)
            results.append({"name": ent.name, "error": str(e)})
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", help="Single entity name")
    parser.add_argument("--cik", help="Direct CIK lookup (bypasses name search)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("./output/sec-edgar"))
    parser.add_argument("--max", type=int, default=50, help="Max filings per entity")
    args = parser.parse_args()

    client = EdgarClient()
    results = []
    if args.all:
        results = pull_all(args.output, max_per_entity=args.max)
    elif args.cik:
        ent = EdgarEntity(name=f"CIK{args.cik}", cik=args.cik.zfill(10),
                          relevance="ad-hoc lookup")
        results = [pull_entity(client, ent, args.output, max_filings=args.max)]
    elif args.entity:
        ent = next((t for t in TARGETS if t.name == args.entity), None)
        if not ent:
            ent = EdgarEntity(name=args.entity, relevance="ad-hoc")
        results = [pull_entity(client, ent, args.output, max_filings=args.max)]
    else:
        parser.error("Specify --entity, --cik, or --all")

    import json
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Summary: %s", args.output / "_summary.json")
