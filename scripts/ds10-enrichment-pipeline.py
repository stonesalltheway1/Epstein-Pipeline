"""DS10 Enrichment Pipeline — Master Orchestrator.

Runs the full DS10 enrichment pipeline in order:
  1. Ingest documents + OCR text into Neon
  2. NER person linking from OCR text
  3. Link media items to persons
  4. (Optional) Vision enrichment for key items

Usage:
    # Run full pipeline (steps 1-3)
    python scripts/ds10-enrichment-pipeline.py

    # Run specific steps
    python scripts/ds10-enrichment-pipeline.py --steps 1,2,3

    # Include vision enrichment (step 4, costs API credits)
    python scripts/ds10-enrichment-pipeline.py --steps 1,2,3,4 --vision-limit 500

    # Dry run (no DB writes)
    python scripts/ds10-enrichment-pipeline.py --dry-run

Prerequisites:
    pip install pymupdf psycopg2-binary python-dotenv anthropic httpx
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_step(name: str, script: str, args: list[str], dry_run: bool = False):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'='*60}", flush=True)
    print(f"  STEP: {name}", flush=True)
    print(f"  Script: {script}", flush=True)
    print(f"{'='*60}\n", flush=True)

    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + args
    if dry_run:
        cmd.append("--dry-run")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent))
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} completed in {elapsed/60:.1f} minutes\n", flush=True)

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DS10 Enrichment Pipeline")
    parser.add_argument("--steps", default="1,2,3", help="Comma-separated step numbers (1-4)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for ingestion")
    parser.add_argument("--vision-limit", type=int, default=500, help="Max items for vision enrichment")
    parser.add_argument("--vision-model", default="claude-haiku-4-5-20251001", help="Vision model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit docs for testing")
    args = parser.parse_args()

    steps = {int(s.strip()) for s in args.steps.split(",")}

    print("DS10 Enrichment Pipeline", flush=True)
    print(f"  Steps: {sorted(steps)}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    if args.dry_run:
        print("  MODE: DRY RUN", flush=True)

    start = time.time()
    results = {}

    # Step 1: Ingest documents + OCR into Neon
    if 1 in steps:
        step_args = ["--workers", str(args.workers)]
        if args.limit:
            step_args += ["--limit", str(args.limit)]
        results[1] = run_step(
            "Ingest DS10 Documents + OCR to Neon",
            "ingest-ds10-to-neon.py",
            step_args,
            args.dry_run,
        )

    # Step 2: NER person linking
    if 2 in steps:
        step_args = ["--batch-size", "500"]
        if args.limit:
            step_args += ["--limit", str(args.limit)]
        results[2] = run_step(
            "NER Person Linking",
            "ner-ds10-persons.py",
            step_args,
            args.dry_run,
        )

    # Step 3: Link media items to persons
    if 3 in steps:
        results[3] = run_step(
            "Link Media to Persons",
            "link-ds10-media.py",
            [],
            args.dry_run,
        )

    # Step 4: Vision enrichment (optional, costs API credits)
    if 4 in steps:
        step_args = [
            "--limit", str(args.vision_limit),
            "--model", args.vision_model,
        ]
        results[4] = run_step(
            "Vision Enrichment",
            "enrich-ds10-vision-batch.py",
            step_args,
            args.dry_run,
        )

    # Summary
    elapsed = time.time() - start
    print(f"\n{'='*60}", flush=True)
    print(f"  PIPELINE COMPLETE — {elapsed/60:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)
    for step, success in sorted(results.items()):
        status = "OK" if success else "FAILED"
        print(f"  Step {step}: [{status}]", flush=True)


if __name__ == "__main__":
    main()
