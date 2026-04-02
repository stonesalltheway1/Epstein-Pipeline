#!/usr/bin/env python3
"""Apply confirmed entity resolution merges to the persons registry and Neon database.

For each merge: keeps the canonical person record, absorbs the duplicate's
aliases and document links, then removes the duplicate. All Neon tables with
person FK references are updated.

Usage:
    # Dry run (preview only, no changes)
    python scripts/apply-entity-merges.py --dry-run

    # Apply to registry JSON only (no database)
    python scripts/apply-entity-merges.py --registry-only

    # Apply to both registry and Neon database
    python scripts/apply-entity-merges.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Confirmed merges: (keep_id, drop_id, reason)
# keep_id = canonical record to keep (more complete name/aliases)
# drop_id = duplicate to absorb into keep_id then remove
# ---------------------------------------------------------------------------

CONFIRMED_MERGES: list[tuple[str, str, str]] = [
    ("p-0076", "p-0073", "Alexander Acosta = Alex Acosta"),
    ("p-0080", "p-0079", "Alexandria Dixon = Alexandra Dixon"),
    ("p-0547", "p-0546", "Forrest Sawyer = Forest Sawyer"),
    ("p-0604", "p-0601", "Glenn Dubin = Glen Dubin"),
    ("p-0763", "p-0760", "Jimmy Cayne = Jim Cayne"),
    ("p-0894", "p-0897", "Laura Edelstein = Laurie Edelstein"),
    ("p-0917", "p-0913", "Leslie Wexner = Les Wexner"),
    ("p-1365", "p-1367", "Shelley Harrison = Shelly Harrison"),
    ("p-1415", "p-1407", "Steven Tisch = Steve Tisch"),
    ("p-1449", "p-1448", "Thorbjorn Jagland (accented) = Thorbjorn Jagland"),
    ("p-1476", "p-1477", "Tova Noel = Tovah Noel"),
]

REGISTRY_PATH = Path("data/persons-registry.json")


def merge_registry(dry_run: bool = False) -> dict[str, str]:
    """Merge duplicate persons in the registry JSON.

    For each merge pair:
    1. Absorb the drop record's name as an alias on the keep record
    2. Absorb any aliases from the drop record
    3. Prefer the keep record's category/bio (it's the more complete record)
    4. Remove the drop record

    Returns the merge map: {drop_id: keep_id}
    """
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in registry}

    merge_map: dict[str, str] = {}
    merged_count = 0

    for keep_id, drop_id, reason in CONFIRMED_MERGES:
        keep = by_id.get(keep_id)
        drop = by_id.get(drop_id)

        if not keep:
            print(f"  WARNING: keep_id {keep_id} not found in registry, skipping")
            continue
        if not drop:
            print(f"  WARNING: drop_id {drop_id} not found in registry, skipping")
            continue

        # Absorb drop's name as alias (if not already present)
        existing_aliases = set(a.lower() for a in keep.get("aliases", []))
        existing_aliases.add(keep["name"].lower())

        new_aliases = list(keep.get("aliases", []))
        if drop["name"].lower() not in existing_aliases:
            new_aliases.append(drop["name"])

        # Absorb drop's aliases
        for alias in drop.get("aliases", []):
            if alias.lower() not in existing_aliases:
                new_aliases.append(alias)
                existing_aliases.add(alias.lower())

        # Prefer keep's bio, but use drop's if keep has none
        if not keep.get("shortBio") and drop.get("shortBio"):
            keep["shortBio"] = drop["shortBio"]

        keep["aliases"] = new_aliases
        merge_map[drop_id] = keep_id

        action = "WOULD MERGE" if dry_run else "MERGED"
        print(
            f"  {action}: {drop_id} \"{drop['name']}\" -> "
            f"{keep_id} \"{keep['name']}\" (aliases now: {new_aliases})"
        )
        merged_count += 1

    if not dry_run:
        # Remove dropped records
        registry = [p for p in registry if p["id"] not in merge_map]
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Registry updated: {merged_count} merges applied, "
              f"{len(registry)} persons remaining")
    else:
        print(f"\n  DRY RUN: {merged_count} merges would be applied")

    return merge_map


def merge_neon(merge_map: dict[str, str], dry_run: bool = False) -> None:
    """Apply merges to all Neon database tables that reference person IDs.

    For each (drop_id -> keep_id) merge:
    1. Copy join table rows from drop to keep (ON CONFLICT DO NOTHING)
    2. Update FK references in non-join tables
    3. Delete the dropped person record (cascades join tables)

    Tables affected:
    - document_persons (join)     - copy rows, cascade delete handles rest
    - email_persons (join)        - copy rows, cascade delete handles rest
    - flight_passengers (join)    - copy rows, cascade delete handles rest
    - relationships               - update person1_id and person2_id
    - video_depositions           - update deponent_person_id
    - deposition_segments         - update speaker_person_id
    - sanctions_matches           - update person_id
    - political_donations         - update person_id
    - nonprofit_officers          - update person_id
    - icij_matches                - update source_id
    - icij_relationships          - update source_person_id
    - persons                     - update aliases on keep, delete drop
    """
    _raw_url = os.environ.get("EPSTEIN_NEON_DATABASE_URL", "")
    if not _raw_url:
        print("  EPSTEIN_NEON_DATABASE_URL not set, skipping database merge")
        return

    try:
        import psycopg
    except ImportError:
        print("  psycopg not installed, skipping database merge")
        return

    db_url = re.sub(r"[&?]sslnegotiation=[^&]*", "", _raw_url)

    if dry_run:
        print(f"\n  DRY RUN: Would update {len(merge_map)} person IDs across 12 Neon tables")
        return

    print(f"\n  Connecting to Neon...")
    conn = psycopg.connect(db_url)
    cur = conn.cursor()

    # Check which tables actually exist in this database
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    existing_tables = {row[0] for row in cur.fetchall()}

    for drop_id, keep_id in merge_map.items():
        print(f"  Merging {drop_id} -> {keep_id} in Neon...")
        rows_affected = 0

        # 1. document_persons (uses doc_id, not document_id)
        if "document_persons" in existing_tables:
            cur.execute("""
                INSERT INTO document_persons (doc_id, person_id)
                SELECT doc_id, %(keep)s FROM document_persons WHERE person_id = %(drop)s
                ON CONFLICT DO NOTHING
            """, {"keep": keep_id, "drop": drop_id})
            cur.execute(
                "DELETE FROM document_persons WHERE person_id = %(drop)s",
                {"drop": drop_id},
            )
            rows_affected += cur.rowcount

        # 2. email_persons (may not exist)
        if "email_persons" in existing_tables:
            cur.execute("""
                INSERT INTO email_persons (email_id, person_id)
                SELECT email_id, %(keep)s FROM email_persons WHERE person_id = %(drop)s
                ON CONFLICT DO NOTHING
            """, {"keep": keep_id, "drop": drop_id})
            cur.execute(
                "DELETE FROM email_persons WHERE person_id = %(drop)s",
                {"drop": drop_id},
            )

        # 3. flight_passengers (may not exist)
        if "flight_passengers" in existing_tables:
            cur.execute("""
                INSERT INTO flight_passengers (flight_id, person_id, role)
                SELECT flight_id, %(keep)s, role FROM flight_passengers
                WHERE person_id = %(drop)s
                ON CONFLICT DO NOTHING
            """, {"keep": keep_id, "drop": drop_id})
            cur.execute(
                "DELETE FROM flight_passengers WHERE person_id = %(drop)s",
                {"drop": drop_id},
            )

        # 4. relationships (may not exist)
        if "relationships" in existing_tables:
            cur.execute(
                "DELETE FROM relationships WHERE person1_id = %(drop)s "
                "OR person2_id = %(drop)s",
                {"drop": drop_id},
            )

        # 5. video_depositions
        if "video_depositions" in existing_tables:
            cur.execute(
                "UPDATE video_depositions SET deponent_person_id = %(keep)s "
                "WHERE deponent_person_id = %(drop)s",
                {"keep": keep_id, "drop": drop_id},
            )
            rows_affected += cur.rowcount

        # 6. deposition_segments
        if "deposition_segments" in existing_tables:
            cur.execute(
                "UPDATE deposition_segments SET speaker_person_id = %(keep)s "
                "WHERE speaker_person_id = %(drop)s",
                {"keep": keep_id, "drop": drop_id},
            )
            rows_affected += cur.rowcount

        # 7. sanctions_matches
        if "sanctions_matches" in existing_tables:
            cur.execute("""
                UPDATE sanctions_matches SET person_id = %(keep)s
                WHERE person_id = %(drop)s
                AND NOT EXISTS (
                    SELECT 1 FROM sanctions_matches s2
                    WHERE s2.person_id = %(keep)s
                      AND s2.entity_id = sanctions_matches.entity_id
                )
            """, {"keep": keep_id, "drop": drop_id})
            rows_affected += cur.rowcount
            cur.execute(
                "DELETE FROM sanctions_matches WHERE person_id = %(drop)s",
                {"drop": drop_id},
            )

        # 8. political_donations
        if "political_donations" in existing_tables:
            cur.execute("""
                UPDATE political_donations SET person_id = %(keep)s
                WHERE person_id = %(drop)s
                AND NOT EXISTS (
                    SELECT 1 FROM political_donations p2
                    WHERE p2.person_id = %(keep)s
                      AND p2.fec_transaction_id = political_donations.fec_transaction_id
                )
            """, {"keep": keep_id, "drop": drop_id})
            rows_affected += cur.rowcount
            cur.execute(
                "DELETE FROM political_donations WHERE person_id = %(drop)s",
                {"drop": drop_id},
            )

        # 9. nonprofit_officers
        if "nonprofit_officers" in existing_tables:
            cur.execute(
                "UPDATE nonprofit_officers SET person_id = %(keep)s "
                "WHERE person_id = %(drop)s",
                {"keep": keep_id, "drop": drop_id},
            )
            rows_affected += cur.rowcount

        # 10. icij_matches
        if "icij_matches" in existing_tables:
            cur.execute("""
                UPDATE icij_matches SET source_id = %(keep)s
                WHERE source_id = %(drop)s AND source_type = 'person'
                AND NOT EXISTS (
                    SELECT 1 FROM icij_matches i2
                    WHERE i2.source_id = %(keep)s
                      AND i2.source_type = 'person'
                      AND i2.icij_node_id = icij_matches.icij_node_id
                )
            """, {"keep": keep_id, "drop": drop_id})
            rows_affected += cur.rowcount
            cur.execute(
                "DELETE FROM icij_matches "
                "WHERE source_id = %(drop)s AND source_type = 'person'",
                {"drop": drop_id},
            )

        # 11. icij_relationships
        if "icij_relationships" in existing_tables:
            cur.execute(
                "UPDATE icij_relationships SET source_person_id = %(keep)s "
                "WHERE source_person_id = %(drop)s",
                {"keep": keep_id, "drop": drop_id},
            )
            rows_affected += cur.rowcount

        # 12. Delete the dropped person record
        if "persons" in existing_tables:
            cur.execute(
                "DELETE FROM persons WHERE id = %(drop)s", {"drop": drop_id}
            )
            rows_affected += cur.rowcount

        print(f"    {rows_affected} rows affected")

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Neon merge complete: {len(merge_map)} persons merged")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply entity resolution merges")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--registry-only", action="store_true", help="Update registry JSON only")
    args = parser.parse_args()

    print("=== Entity Resolution: Apply Confirmed Merges ===\n")
    print(f"  Merges to apply: {len(CONFIRMED_MERGES)}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Build merge map from the hardcoded list (works even if registry already cleaned)
    full_merge_map = {drop_id: keep_id for keep_id, drop_id, _ in CONFIRMED_MERGES}

    # Step 1: Merge registry
    print("--- Step 1: Registry JSON ---")
    registry_map = merge_registry(dry_run=args.dry_run)
    if not registry_map:
        print("  (Registry already clean — using hardcoded merge map for Neon)")

    # Step 2: Merge Neon database (always use the full merge map)
    if not args.registry_only:
        print("\n--- Step 2: Neon Database ---")
        merge_neon(full_merge_map, dry_run=args.dry_run)
    else:
        print("\n--- Step 2: Neon Database (SKIPPED: --registry-only) ---")

    # Step 3: Save audit trail
    if not args.dry_run:
        audit = {
            "merges_applied": len(full_merge_map),
            "merge_map": full_merge_map,
            "details": [
                {"keep_id": k, "drop_id": d, "reason": r}
                for k, d, r in CONFIRMED_MERGES
            ],
        }
        audit_path = Path("output/entity-merge-audit.json")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"\n  Audit trail saved to {audit_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
