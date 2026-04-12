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
    DocketRef("22-cv-10904", "nysd", "Government of USVI v. JPMorgan Chase", priority=3),
    DocketRef("22-cv-10019", "nysd", "Doe 1 v. JPMorgan Chase", priority=3),
    DocketRef("15-cv-07433", "nysd", "Giuffre v. Maxwell", priority=3),
    DocketRef("20-cr-00330", "nysd", "United States v. Ghislaine Maxwell", priority=2),
    DocketRef("19-cr-00490", "nysd", "United States v. Jeffrey Epstein", priority=2),
    DocketRef("08-cv-80736", "flsd", "Jane Doe v. United States (Acosta plea)", priority=2),
    DocketRef("09-cv-80469", "flsd", "Edwards v. Dershowitz", priority=1),
    DocketRef("23-cv-03003", "nysd", "Doe v. Deutsche Bank", priority=1),
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

    def iter_docket_entries(self, docket_id: int):
        """Yield all docket entries (docket items with filings)."""
        url = f"{CL_BASE}/docket-entries/"
        params = {"docket": docket_id, "page_size": 100, "order_by": "entry_number"}
        while url:
            data = self._get(url, params)
            for entry in data.get("results", []):
                yield entry
            url = data.get("next")
            params = None  # next URL already includes params

    def iter_recap_documents(self, entry_id: int):
        """Yield RECAP documents (actual PDFs) for a docket entry."""
        url = f"{CL_BASE}/recap-documents/"
        params = {"docket_entry": entry_id, "page_size": 100}
        while url:
            data = self._get(url, params)
            for doc in data.get("results", []):
                yield doc
            url = data.get("next")
            params = None

    def download_recap_pdf(self, recap_doc: dict, output_path: Path) -> bool:
        """Download a RECAP PDF. Returns True if downloaded, False if no filepath available."""
        filepath = recap_doc.get("filepath_local") or recap_doc.get("filepath_ia")
        if not filepath:
            return False
        # filepath_local is relative; prefix with recap server
        if filepath.startswith("/"):
            url = f"https://www.courtlistener.com{filepath}"
        elif filepath.startswith("recap/"):
            url = f"https://storage.courtlistener.com/{filepath}"
        else:
            url = filepath
        r = self.session.get(url, timeout=60, stream=True)
        if r.status_code != 200:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        return True


def pull_docket(client: CourtListenerClient, ref: DocketRef,
                output_dir: Path, max_entries: int | None = None) -> dict:
    """Pull all filings for a single docket."""
    logger.info("Resolving docket %s in %s (%s)...",
                ref.docket_number, ref.court, ref.case_name)
    docket = client.find_docket(ref.docket_number, ref.court)
    if not docket:
        logger.warning("Docket not found: %s / %s", ref.docket_number, ref.court)
        return {"docket": ref.docket_number, "found": False}

    docket_id = docket["id"]
    logger.info("Docket ID: %d | %s", docket_id, docket.get("case_name"))

    docket_dir = output_dir / f"{ref.court}-{ref.docket_number.replace('/', '_')}"
    docket_dir.mkdir(parents=True, exist_ok=True)

    # Save docket metadata
    import json
    (docket_dir / "_docket.json").write_text(
        json.dumps(docket, indent=2, default=str), encoding="utf-8",
    )

    total_entries = 0
    total_pdfs = 0
    total_downloaded = 0
    for entry in client.iter_docket_entries(docket_id):
        total_entries += 1
        if max_entries and total_entries > max_entries:
            break

        entry_num = entry.get("entry_number", "?")
        entry_id = entry["id"]
        (docket_dir / f"entry_{entry_num}.json").write_text(
            json.dumps(entry, indent=2, default=str), encoding="utf-8",
        )

        for rd in client.iter_recap_documents(entry_id):
            total_pdfs += 1
            doc_num = rd.get("document_number", "?")
            att_num = rd.get("attachment_number", "")
            pdf_name = f"entry_{entry_num}_doc_{doc_num}"
            if att_num:
                pdf_name += f"_att_{att_num}"
            pdf_name += ".pdf"
            pdf_path = docket_dir / pdf_name

            if pdf_path.exists():
                continue  # resume — already downloaded

            if client.download_recap_pdf(rd, pdf_path):
                total_downloaded += 1
                if total_downloaded % 10 == 0:
                    logger.info("  downloaded %d PDFs...", total_downloaded)

    logger.info("Docket %s: %d entries, %d PDFs found, %d downloaded",
                ref.docket_number, total_entries, total_pdfs, total_downloaded)
    return {
        "docket": ref.docket_number, "court": ref.court, "case_name": ref.case_name,
        "found": True, "entries": total_entries, "pdfs": total_pdfs,
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


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket", help="Single docket number (e.g. 22-cv-10904)")
    parser.add_argument("--court", default="nysd", help="Court slug (default: nysd)")
    parser.add_argument("--output", type=Path, default=Path("./output/courtlistener"))
    parser.add_argument("--all", action="store_true",
                        help="Pull all known Epstein dockets (see EPSTEIN_DOCKETS)")
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        results = pull_all_epstein_dockets(args.output)
    elif args.docket:
        client = CourtListenerClient()
        ref = DocketRef(args.docket, args.court, args.docket)
        results = [pull_docket(client, ref, args.output, max_entries=args.max_entries)]
    else:
        parser.error("Specify --docket <num> or --all")

    import json
    summary_path = args.output / "_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written: %s", summary_path)
