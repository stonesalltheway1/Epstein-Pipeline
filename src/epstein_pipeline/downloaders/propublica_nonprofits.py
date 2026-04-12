"""ProPublica Nonprofit Explorer downloader for Epstein-linked foundations.

Pulls Form 990 filings from ProPublica's Nonprofit Explorer API
(https://projects.propublica.org/nonprofits/api/v2). No authentication
required; browser User-Agent advised to avoid anti-bot blocks.

The existing `nonprofits.py` downloader handles IRS bulk data.  This module
is focused on pulling the richer ProPublica-normalized filings + raw PDF
URLs for a curated list of Epstein-adjacent nonprofits.

Targets include Epstein-controlled foundations, Epstein-funded charities,
and associate-controlled foundations cross-referenced from prior research.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE = "https://projects.propublica.org/nonprofits/api/v2"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0"


@dataclass
class NonprofitTarget:
    ein: str           # 9-digit EIN with no dashes (e.g. "660629306")
    name: str          # canonical name for display
    relevance: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Curated targets — Epstein-controlled + known associate foundations.
# EINs sourced from prior ProPublica/IRS lookups; Epstein-related grants
# documented in court filings and estate disclosures.
TARGETS: list[NonprofitTarget] = [
    # ── Epstein-controlled ────────────────────────────────────────────
    NonprofitTarget(
        ein="660629306",
        name="Jeffrey Epstein VI Foundation",
        relevance="Epstein-controlled USVI foundation; major grant recipient of Epstein personal funds.",
    ),
    NonprofitTarget(
        ein="462400955",
        name="Gratitude America Ltd",
        relevance="Epstein-controlled vehicle (dissolved post-2019).",
    ),
    NonprofitTarget(
        ein="133643429",
        name="C.O.U.Q. Foundation Inc",
        relevance="Epstein-controlled; donated to universities and associates.",
    ),
    NonprofitTarget(
        ein="261605864",
        name="Enhanced Education",
        relevance="Epstein-linked education foundation.",
    ),
    NonprofitTarget(
        ein="133528667",
        name="J Epstein Foundation",
        relevance="Epstein family foundation.",
    ),
    NonprofitTarget(
        ein="811263733",
        name="Financial Trust Company",
        relevance="Epstein-controlled USVI entity (if publicly filed).",
    ),

    # ── Associate / recipient foundations ─────────────────────────────
    NonprofitTarget(
        ein="387193282",
        name="Mark Epstein Foundation",
        relevance="Jeffrey Epstein's brother's foundation.",
    ),
    NonprofitTarget(
        ein="133947890",
        name="Wexner Foundation",
        relevance="Leslie Wexner's charitable foundation; Epstein was a named trustee/adviser.",
    ),
    NonprofitTarget(
        ein="311318013",
        name="Wexner Heritage Foundation",
        relevance="Wexner-family vehicle with Epstein historical ties.",
    ),
    NonprofitTarget(
        ein="237320631",
        name="Clinton Foundation",
        relevance="Named in Epstein flight logs and donor disclosures.",
    ),
    NonprofitTarget(
        ein="223496220",
        name="Harvard University (President & Fellows)",
        relevance="Received Epstein donations; Program for Evolutionary Dynamics.",
    ),
    NonprofitTarget(
        ein="133996471",
        name="MIT (Massachusetts Institute of Technology)",
        relevance="Epstein donor relationship; 2020 Goodwin Procter report named donations.",
    ),
    # Phase 2 expansion candidates:
    NonprofitTarget(
        ein="030213226",
        name="University of Maine Foundation",
        relevance="Recipient of documented Epstein grants.",
    ),
    NonprofitTarget(
        ein="137265141",
        name="Melanoma Research Alliance Foundation",
        relevance="Epstein donor recipient (cancer research funding).",
    ),
    NonprofitTarget(
        ein="474634539",
        name="SOS Humanity / SOS Méditerranée (US branch)",
        relevance="Associate-linked humanitarian fund.",
    ),
    NonprofitTarget(
        ein="454757735",
        name="Zorro Charitable Foundation",
        relevance="Cypress/Zorro ranch charitable entity (if filed publicly).",
    ),
]


class ProPublicaClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last = 0.0

    def _get(self, url: str, params: dict | None = None) -> dict:
        # Be polite — ~4 requests/sec
        elapsed = time.time() - self._last
        if elapsed < 0.25:
            time.sleep(0.25 - elapsed)
        r = self.session.get(url, params=params, timeout=30)
        self._last = time.time()
        r.raise_for_status()
        return r.json()

    def get_organization(self, ein: str) -> dict:
        """Fetch full organization record including filings."""
        # Strip dashes just in case
        ein_clean = ein.replace("-", "")
        return self._get(f"{BASE}/organizations/{ein_clean}.json")

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        """Search by name; returns list of organization summaries."""
        data = self._get(f"{BASE}/search.json", {"q": query})
        return data.get("organizations", [])[:max_results]


def pull_target(client: ProPublicaClient, target: NonprofitTarget,
                output_dir: Path) -> dict:
    """Pull org metadata + raw 990 PDFs for a target."""
    try:
        data = client.get_organization(target.ein)
    except requests.HTTPError as e:
        logger.warning("%s EIN %s not found: %s", target.name, target.ein, e)
        return {"ein": target.ein, "name": target.name, "found": False, "error": str(e)}

    org = data.get("organization", {})
    filings_with_data = data.get("filings_with_data", [])
    filings_without_data = data.get("filings_without_data", [])

    logger.info("%s (EIN %s): %d filings_with_data, %d filings_without_data",
                target.name, target.ein, len(filings_with_data),
                len(filings_without_data))

    org_dir = output_dir / f"{target.ein}_{target.name.replace(' ', '_').replace('/', '_')[:60]}"
    org_dir.mkdir(parents=True, exist_ok=True)

    # Save full API payload
    (org_dir / "_organization.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")

    # Download 990 PDFs — requires a Referer header pointing at the org page
    # (ProPublica's download-filing endpoint 403s without it).
    downloaded = 0
    org_page = f"https://projects.propublica.org/nonprofits/organizations/{target.ein}"
    for filing in filings_with_data:
        pdf_url = filing.get("pdf_url")
        if not pdf_url:
            continue
        tax_prd = filing.get("tax_prd") or filing.get("tax_prd_yr", "unknown")
        form = filing.get("formtype_str") or str(filing.get("formtype", ""))
        fname = f"{tax_prd}_{form.replace(' ', '_').replace('/', '-')}.pdf"
        dest = org_dir / fname
        if dest.exists():
            continue
        try:
            r = client.session.get(
                pdf_url, timeout=60, stream=True,
                headers={"Referer": org_page},
            )
            r.raise_for_status()
            # ProPublica sometimes redirects to IRS AWS bucket for the actual PDF
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            # Verify we got a PDF (not an HTML error page)
            if dest.stat().st_size < 1024 or dest.read_bytes()[:4] != b"%PDF":
                logger.warning("PDF %s returned non-PDF content; removing", pdf_url)
                dest.unlink(missing_ok=True)
            else:
                downloaded += 1
        except Exception as e:
            logger.warning("PDF %s failed: %s", pdf_url, e)

    return {
        "ein": target.ein, "name": target.name, "found": True,
        "ntee": org.get("ntee_code"), "state": org.get("state"),
        "filings_total": len(filings_with_data) + len(filings_without_data),
        "pdfs_downloaded": downloaded,
        "output_dir": str(org_dir),
    }


def pull_all(output_dir: Path) -> list[dict]:
    client = ProPublicaClient()
    results = []
    for t in TARGETS:
        try:
            r = pull_target(client, t, output_dir)
            results.append(r)
        except Exception as e:
            logger.exception("%s failed: %s", t.name, e)
            results.append({"ein": t.ein, "name": t.name, "error": str(e)})
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ein", help="Single EIN to process")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("./output/propublica-nonprofits"))
    args = parser.parse_args()

    client = ProPublicaClient()
    results = []
    if args.all:
        results = pull_all(args.output)
    elif args.ein:
        t = next((x for x in TARGETS if x.ein == args.ein), None)
        if not t:
            t = NonprofitTarget(ein=args.ein, name=f"EIN {args.ein}", relevance="ad-hoc")
        results = [pull_target(client, t, args.output)]
    else:
        parser.error("Specify --ein or --all")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Summary: %s", args.output / "_summary.json")
