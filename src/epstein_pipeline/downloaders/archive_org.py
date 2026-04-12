"""Internet Archive (archive.org) downloader for Epstein-related mirrors.

Many DOJ releases and House Oversight Epstein estate productions are mirrored
on archive.org. These mirrors are the most reliable way to ingest the data
without fighting DOJ's Akamai WAF or Oversight's Google Drive/Dropbox hosting
that doesn't play nicely with automation.

Usage:
    pip install internetarchive
    python -m epstein_pipeline.downloaders.archive_org --list
    python -m epstein_pipeline.downloaders.archive_org --item oversight-committee-additional-epstein-files
    python -m epstein_pipeline.downloaders.archive_org --all --output ./archive-org-downloads
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class IAItem:
    """An Internet Archive item we want to mirror."""
    identifier: str
    title: str
    source_type: str  # 'house-oversight', 'doj-efta', 'doj-hr4405', etc.
    date: str
    size_gb: float
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# Curated list of Archive.org mirrors for Epstein-related releases.
# Maintained manually; verified via archive.org/metadata/{identifier}.
CURATED_ITEMS: list[IAItem] = [
    # ── House Oversight Epstein Estate productions ──────────────────────
    IAItem(
        identifier="OversightCommitteeEpsteinDocs1",
        title="Epstein Estate Documents - First Production",
        source_type="house-oversight",
        date="2025-09-08",
        size_gb=0.23,
        notes="57 files; first Oversight release of Epstein estate records",
        tags=("oversight", "estate", "first-production"),
    ),
    IAItem(
        identifier="EpsteinFirstProduction",
        title="Epstein Estate Documents - First Production (mirror)",
        source_type="house-oversight",
        date="2025-09-08",
        size_gb=0.16,
        notes="Alternative mirror of first production",
        tags=("oversight", "estate", "first-production"),
    ),
    IAItem(
        identifier="epstein-production-3rd-tranch_456",
        title="Epstein Estate Documents Third Batch",
        source_type="house-oversight",
        date="2025-09-26",
        size_gb=0.003,
        tags=("oversight", "estate", "third-batch"),
    ),
    IAItem(
        identifier="oversight-committee-additional-epstein-files",
        title="Oversight Committee Additional Epstein Files (Nov 12, 2025)",
        source_type="house-oversight",
        date="2025-11-12",
        size_gb=25.03,
        notes="26,042 files; THE major Nov 2025 estate batch",
        tags=("oversight", "estate", "nov-2025", "primary-release"),
    ),
    IAItem(
        identifier="Epstein_Estate_Documents_-_Seventh_Production",
        title="Epstein Estate Documents - Seventh Production",
        source_type="house-oversight",
        date="2025-11-25",
        size_gb=39.33,
        notes="26,053 files; largest estate production batch",
        tags=("oversight", "estate", "seventh-production"),
    ),
    IAItem(
        identifier="12.11.25-estate-production_202512",
        title="Epstein Estate Documents December 11 Batch",
        source_type="house-oversight",
        date="2025-12-12",
        size_gb=0.05,
        tags=("oversight", "estate", "dec-2025"),
    ),
    IAItem(
        identifier="20000-epstein-estate-documents",
        title="20,000 Epstein Estate Documents",
        source_type="house-oversight",
        date="2025-11-13",
        size_gb=0.05,
        tags=("oversight", "estate"),
    ),

    # ── DOJ Epstein Files Transparency Act (H.R. 4405) ──────────────────
    IAItem(
        identifier="epstein_library_transparency_act_hr_4405_dataset1_20260204",
        title="DOJ Disclosures - Epstein Files Transparency Act - DataSet 01",
        source_type="doj-hr4405",
        date="2026-02-04",
        size_gb=1.32,
        notes="First HR 4405 compliance batch",
        tags=("doj", "hr4405", "dataset-1"),
    ),
    IAItem(
        identifier="epstein_library_transparency_act_hr_4405_dataset8",
        title="DOJ Disclosures - DataSet 8",
        source_type="doj-hr4405",
        date="2025-12-19",
        size_gb=10.69,
        tags=("doj", "hr4405", "dataset-8"),
    ),
    IAItem(
        identifier="epstein_library_transparency_act_hr_4405_dataset11_202602",
        title="DOJ Disclosures - DataSet 11",
        source_type="doj-hr4405",
        date="2026-02",
        size_gb=27.44,
        tags=("doj", "hr4405", "dataset-11"),
    ),
    IAItem(
        identifier="epstein_library_transparency_act_hr_4405_dataset9_202602",
        title="DOJ Disclosures - DataSet 9",
        source_type="doj-hr4405",
        date="2026-02",
        size_gb=107.13,
        notes="LARGE - only pull if disk space allows",
        tags=("doj", "hr4405", "dataset-9"),
    ),
    IAItem(
        identifier="epstein_library_transparency_act_hr_4405_dataset10_202605",
        title="DOJ Disclosures - DataSet 10",
        source_type="doj-hr4405",
        date="2026-05",
        size_gb=84.44,
        notes="LARGE",
        tags=("doj", "hr4405", "dataset-10"),
    ),

    # ── Other DOJ/Federal releases ──────────────────────────────────────
    IAItem(
        identifier="data-set-12_20260131",
        title="DOJ EFTA Data Set 12",
        source_type="doj-efta",
        date="2026-01-31",
        size_gb=0.12,
        tags=("doj", "efta", "dataset-12"),
    ),
    IAItem(
        identifier="ds-9-efta-gap-repair",
        title="Dataset 9 EFTA Gap Repair",
        source_type="doj-efta",
        date="2026-02",
        size_gb=0.07,
        notes="Fills gaps in DS9 where files were missing from initial release",
        tags=("doj", "efta", "dataset-9", "gap-repair"),
    ),
    IAItem(
        identifier="jeffrey-epstein-files-released-2025-02-27",
        title="DOJ Jeffrey Epstein Files Released 2025-02-27",
        source_type="doj-efta",
        date="2025-02-27",
        size_gb=0.12,
        notes="Feb 2025 'Phase 1' binder release",
        tags=("doj", "feb-2025", "phase-1"),
    ),
    IAItem(
        identifier="efta-19-dec-2025",
        title="EFTA - 19 December 2025",
        source_type="doj-efta",
        date="2025-12-19",
        size_gb=5.10,
        notes="Dec 19, 2025 HR 4405 first compliance batch (deadline release)",
        tags=("doj", "dec-2025", "hr4405"),
    ),
    IAItem(
        identifier="data-set-1",
        title="Jeffrey Epstein DOJ Disclosures: Data Set 1-7",
        source_type="doj-efta",
        date="2025",
        size_gb=3.22,
        notes="Combined DS1-7",
        tags=("doj", "efta", "ds1-7"),
    ),
    IAItem(
        identifier="JEpstein_H_R_4405_251223",
        title="DOJ Disclosures HR 4405 (Dec 23, 2025)",
        source_type="doj-hr4405",
        date="2025-12-23",
        size_gb=13.88,
        tags=("doj", "hr4405", "dec-2025"),
    ),
]


def list_items() -> None:
    """Print known items sorted by size."""
    items = sorted(CURATED_ITEMS, key=lambda x: x.size_gb)
    print(f"{'SIZE':>10}  {'DATE':>10}  {'TYPE':<16}  {'IDENTIFIER':<60}  TITLE")
    print("-" * 140)
    total = 0
    for it in items:
        total += it.size_gb
        print(f"{it.size_gb:>8.2f}GB  {it.date:<10}  {it.source_type:<16}  "
              f"{it.identifier:<60}  {it.title[:50]}")
    print(f"\nTOTAL: {total:.2f} GB across {len(items)} items")


def download_item(item: IAItem, output_dir: Path) -> dict:
    """Download an IA item via the `ia` CLI (most reliable, handles resumption)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (%.2f GB)...", item.identifier, item.size_gb)
    result = subprocess.run(
        ["ia", "download", item.identifier],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=24 * 3600,
    )
    success = result.returncode == 0
    item_dir = output_dir / item.identifier
    files_found = list(item_dir.rglob("*")) if item_dir.exists() else []
    pdf_count = sum(1 for f in files_found if f.suffix.lower() == ".pdf")

    return {
        "identifier": item.identifier,
        "success": success,
        "files": len(files_found),
        "pdfs": pdf_count,
        "output_dir": str(item_dir),
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def download_all(output_dir: Path, max_size_gb: float | None = None) -> list[dict]:
    """Download all curated items, optionally filtered by max size."""
    targets = [x for x in CURATED_ITEMS
               if max_size_gb is None or x.size_gb <= max_size_gb]
    logger.info("Downloading %d items (%.2f GB total)...",
                len(targets), sum(x.size_gb for x in targets))
    results = []
    for it in targets:
        try:
            results.append(download_item(it, output_dir))
        except Exception as e:
            logger.exception("Failed %s: %s", it.identifier, e)
            results.append({"identifier": it.identifier, "success": False, "error": str(e)})
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List known items")
    parser.add_argument("--item", help="Download a specific identifier")
    parser.add_argument("--all", action="store_true", help="Download all curated items")
    parser.add_argument("--max-size-gb", type=float, default=None,
                        help="Only download items up to this size")
    parser.add_argument("--output", type=Path, default=Path("./archive-org-downloads"))
    args = parser.parse_args()

    if args.list:
        list_items()
    elif args.item:
        it = next((x for x in CURATED_ITEMS if x.identifier == args.item), None)
        if not it:
            it = IAItem(identifier=args.item, title="ad-hoc", source_type="unknown",
                        date="?", size_gb=0)
        print(download_item(it, args.output))
    elif args.all:
        results = download_all(args.output, max_size_gb=args.max_size_gb)
        import json
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "_ia_summary.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
    else:
        parser.error("Specify --list, --item <id>, or --all")
