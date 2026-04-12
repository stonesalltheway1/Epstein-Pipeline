"""CourtListener / RECAP downloader for Epstein-related federal court dockets.

Pulls docket entries, filings, and PDFs from CourtListener's free REST API.
The anonymous search endpoint works without auth, but docket entries and
document downloads require a free API token (register at
https://www.courtlistener.com/help/api/rest/).

Usage:
    export COURTLISTENER_TOKEN=...
    python -m epstein_pipeline.downloaders.courtlistener \
        --docket 15-cv-07433 --court nysd --output ./output/courtlistener/

Known high-value dockets (Epstein-related):
    15-cv-07433  nysd  Giuffre v. Maxwell
    22-cv-10019  nysd  Doe 1 v. JPMorgan Chase
    22-cv-10904  nysd  Government of USVI v. JPMorgan Chase
    20-cr-00330  nysd  United States v. Ghislaine Maxwell
    19-cr-00490  nysd  United States v. Jeffrey Epstein (indictment)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CL_BASE = "https://www.courtlistener.com/api/rest/v4"
USER_AGENT = "EpsteinExposedBot/1.0 (+https://epsteinexposed.com)"


@dataclass
class DocketRef:
    """A target docket to pull."""
    docket_number: str
    court: str  # e.g. "nysd", "flsd"
    case_name: str  # for display
    priority: int = 0  # higher = pull first


# Known high-value Epstein-related dockets
EPSTEIN_DOCKETS: list[DocketRef] = [
    DocketRef("1:22-cv-10904", "nysd", "Government of USVI v. JPMorgan Chase", priority=3),
    DocketRef("1:22-cv-10019", "nysd", "Doe 1 v. JPMorgan Chase", priority=3),
    DocketRef("1:15-cv-07433", "nysd", "Giuffre v. Maxwell", priority=3),
    DocketRef("1:20-cr-00330", "nysd", "United States v. Ghislaine Maxwell", priority=2),
    DocketRef("1:19-cr-00490", "nysd", "United States v. Jeffrey Epstein", priority=2),
    DocketRef("9:08-cv-80736", "flsd", "Jane Doe v. United States (Acosta plea)", priority=2),
    DocketRef("9:09-cv-80469", "flsd", "Edwards v. Dershowitz", priority=1),
    DocketRef("1:23-cv-03003", "nysd", "Doe v. Deutsche Bank", priority=1),
]


class CourtListenerClient:
    """Thin client over CourtListener's REST API v4."""

    def __init__(self, token: str | None = None, rate_limit_sec: float = 0.5):
        self.token = token or os.environ.get("COURTLISTENER_TOKEN")
        self.rate_limit_sec = rate_limit_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if self.token:
            self.session.headers["Authorization"] = f"Token {self.token}"
        self._last_request_time = 0.0

    def _get(self, url: str, params: dict | None = None) -> dict:
        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        r = self.session.get(url, params=params, timeout=30)
        self._last_request_time = time.time()
        r.raise_for_status()
        return r.json()

    def find_docket(self, docket_number: str, court: str) -> dict | None:
        """Resolve a docket_number + court to the CourtListener docket ID."""
        if not self.token:
            raise RuntimeError(
                "COURTLISTENER_TOKEN required for docket lookup. "
                "Register at https://www.courtlistener.com/help/api/rest/"
            )
        data = self._get(
            f"{CL_BASE}/dockets/",
            {"docket_number": docket_number, "court": court, "page_size": 5},
        )
        results = data.get("results", [])
        if not results:
            return None
        # Prefer exact docket_number match
        for d in results:
            if d.get("docket_number") == docket_number:
                return d
        return results[0]

    def search_recap_by_docket(self, docket_id: int):
        """Yield RECAP documents via the search API (free-tier compatible).

        The raw docket-entries/recap-documents endpoints are 403-restricted
        on most free tokens. The search endpoint with type=r returns the
        same data in a different shape.
        """
        url = f"{CL_BASE}/search/"
        params = {"q": f"docket_id:{docket_id}", "type": "r", "page_size": 100}
        while url:
            data = self._get(url, params)
            for res in data.get("results", []):
                for rd in res.get("recap_documents", []):
                    yield rd
            url = data.get("next")
            params = None

    def search_recap_query(self, query: str, court: str | None = None,
                            page_size: int = 100):
        """Free-text search across all RECAP documents."""
        url = f"{CL_BASE}/search/"
        params = {"q": query, "type": "r", "page_size": page_size,
                  "order_by": "dateFiled desc"}
        if court:
            params["court"] = court
        while url:
            data = self._get(url, params)
            for res in data.get("results", []):
                for rd in res.get("recap_documents", []):
                    # Enrich with case metadata
                    rd = dict(rd)
                    rd["_case_name"] = res.get("caseName")
                    rd["_docket_number"] = res.get("docketNumber")
                    rd["_docket_id"] = res.get("docket_id")
                    rd["_date_filed"] = res.get("dateFiled")
                    rd["_court_id"] = res.get("court_id")
                    yield rd
            url = data.get("next")
            params = None

    def download_recap_pdf(self, recap_doc: dict, output_path: Path) -> bool:
        """Download a RECAP PDF. Returns True if downloaded, False otherwise.

        Per CourtListener docs: canonical URL is
        `https://storage.courtlistener.com/{filepath_local}`.
        Fall back to Internet Archive via filepath_ia if storage 404s.
        """
        if not recap_doc.get("is_available"):
            return False
        filepath_local = recap_doc.get("filepath_local")
        filepath_ia = recap_doc.get("filepath_ia")

        candidate_urls: list[str] = []
        if filepath_local:
            # filepath_local is already a relative S3 key like "recap/gov.uscourts.../foo.pdf"
            filepath_local = filepath_local.lstrip("/")
            candidate_urls.append(f"https://storage.courtlistener.com/{filepath_local}")
        if filepath_ia:
            # filepath_ia is an absolute archive.org URL
            candidate_urls.append(filepath_ia)

        for url in candidate_urls:
            try:
                r = self.session.get(url, timeout=60, stream=True)
                if r.status_code == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)
                    return True
            except requests.RequestException:
                continue
        return False


def pull_docket(client: CourtListenerClient, ref: DocketRef,
                output_dir: Path, max_entries: int | None = None) -> dict:
    """Pull all publicly-indexed filings for a single docket.

    Uses the free-tier search API (docket-entries/ and recap-documents/ are
    paid-tier only). Walks search results, downloads available PDFs from
    storage.courtlistener.com with IA fallback.
    """
    import json
    logger.info("Resolving docket %s in %s (%s)...",
                ref.docket_number, ref.court, ref.case_name)
    docket = client.find_docket(ref.docket_number, ref.court)
    if not docket:
        logger.warning("Docket not found: %s / %s", ref.docket_number, ref.court)
        return {"docket": ref.docket_number, "found": False}

    docket_id = docket["id"]
    case_name = docket.get("case_name", "?")
    logger.info("Docket ID: %d | %s", docket_id, case_name)

    safe_num = ref.docket_number.replace("/", "_").replace(":", "_")
    docket_dir = output_dir / f"{ref.court}-{safe_num}"
    docket_dir.mkdir(parents=True, exist_ok=True)

    (docket_dir / "_docket.json").write_text(
        json.dumps(docket, indent=2, default=str), encoding="utf-8",
    )

    total_docs = 0
    total_downloaded = 0
    seen_doc_ids: set[int] = set()

    for rd in client.search_recap_by_docket(docket_id):
        rd_id = rd.get("id")
        if rd_id in seen_doc_ids:
            continue
        seen_doc_ids.add(rd_id)
        total_docs += 1

        if max_entries and total_docs > max_entries:
            break

        # Save metadata
        entry_num = rd.get("entry_number") or rd.get("document_number") or "unknown"
        doc_num = rd.get("document_number", "?")
        att_num = rd.get("attachment_number")
        meta_name = f"entry_{entry_num}_doc_{doc_num}"
        if att_num:
            meta_name += f"_att_{att_num}"
        (docket_dir / f"{meta_name}.json").write_text(
            json.dumps(rd, indent=2, default=str), encoding="utf-8",
        )

        # Try to download PDF
        if not rd.get("is_available"):
            continue
        pdf_path = docket_dir / f"{meta_name}.pdf"
        if pdf_path.exists():
            total_downloaded += 1
            continue
        if client.download_recap_pdf(rd, pdf_path):
            total_downloaded += 1
            if total_downloaded % 10 == 0:
                logger.info("  downloaded %d PDFs...", total_downloaded)

    logger.info("Docket %s: %d recap_documents found, %d PDFs downloaded",
                ref.docket_number, total_docs, total_downloaded)
    return {
        "docket": ref.docket_number, "court": ref.court, "case_name": case_name,
        "found": True, "recap_documents": total_docs,
        "downloaded": total_downloaded, "output_dir": str(docket_dir),
    }


def pull_all_epstein_dockets(output_dir: Path, token: str | None = None) -> list[dict]:
    """Pull all known high-priority Epstein dockets."""
    client = CourtListenerClient(token=token)
    results = []
    for ref in sorted(EPSTEIN_DOCKETS, key=lambda d: -d.priority):
        try:
            r = pull_docket(client, ref, output_dir)
            results.append(r)
        except Exception as e:
            logger.exception("Failed to pull %s: %s", ref.docket_number, e)
            results.append({"docket": ref.docket_number, "found": False, "error": str(e)})
    return results


def pull_epstein_party_search(output_dir: Path, token: str | None = None,
                               max_docs: int = 500) -> dict:
    """Pull every RECAP document where Epstein is a named party.

    Uses the high-precision `party_name:"Jeffrey Epstein"` query from CL docs.
    This catches filings across any court/case — much broader than our curated
    docket list. Also pulls major associate searches (Maxwell, Wexner, etc).
    """
    import json
    client = CourtListenerClient(token=token)

    # Quoted-string queries against CourtListener v4 RECAP search.
    # The `party_name:` / `party:` field filters under-match on v4; plain quoted
    # strings are higher-recall (926 hits for "Jeffrey Epstein" vs 68 for party:).
    queries = [
        '"Jeffrey Epstein"',
        '"Ghislaine Maxwell"',
        '"Virginia Giuffre"',
        '"JPMorgan Chase" "Virgin Islands"',
        '"Deutsche Bank" "Virgin Islands"',
        '"Jeffrey Epstein" JPMorgan',
        '"Jeffrey Epstein" "Deutsche Bank"',
        '"Leslie Wexner" Epstein',
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_query = {}
    seen_doc_ids: set[int] = set()

    for q in queries:
        logger.info("Search: %s", q)
        q_dir = output_dir / ("search_" + q.replace(" ", "_").replace(":", "_")
                              .replace('"', "").replace("/", "_")[:80])
        q_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        downloaded = 0
        for rd in client.search_recap_query(q):
            rd_id = rd.get("id")
            if rd_id in seen_doc_ids:
                continue
            seen_doc_ids.add(rd_id)
            total += 1
            if total > max_docs:
                break

            docket = rd.get("_docket_number", "unknown").replace("/", "_").replace(":", "_")
            doc_num = rd.get("document_number", "?")
            att = rd.get("attachment_number")
            name = f"docket_{docket}_doc_{doc_num}"
            if att:
                name += f"_att_{att}"

            (q_dir / f"{name}.json").write_text(
                json.dumps(rd, indent=2, default=str), encoding="utf-8",
            )

            if not rd.get("is_available"):
                continue
            pdf_path = q_dir / f"{name}.pdf"
            if pdf_path.exists():
                downloaded += 1
                continue
            if client.download_recap_pdf(rd, pdf_path):
                downloaded += 1

        results_by_query[q] = {"total": total, "downloaded": downloaded,
                                "output_dir": str(q_dir)}
        logger.info("  query %r: %d docs, %d downloaded", q, total, downloaded)

    return {"queries": results_by_query, "unique_docs_seen": len(seen_doc_ids)}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket number (e.g. 22-cv-10904)")
    parser.add_argument("--court", default="nysd", help="Court slug (default: nysd)")
    parser.add_argument("--output", type=Path, default=Path("./output/courtlistener"))
    parser.add_argument("--all", action="store_true",
                        help="Pull all known Epstein dockets (see EPSTEIN_DOCKETS)")
    parser.add_argument("--party-search", action="store_true",
                        help="Run broad party_name searches for all Epstein-related filings")
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    if args.party_search:
        results = [pull_epstein_party_search(args.output)]
    elif args.all:
        results = pull_all_epstein_dockets(args.output)
    elif args.docket:
        client = CourtListenerClient()
        ref = DocketRef(args.docket, args.court, args.docket)
        results = [pull_docket(client, ref, args.output, max_entries=args.max_entries)]
    else:
        parser.error("Specify --docket <num>, --all, or --party-search")

    import json
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written: %s", summary_path)
