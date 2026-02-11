"""Data integrity checks for the Epstein Pipeline.

Validates cross-references, date ranges, uniqueness constraints, and
formatting rules that go beyond schema validation.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from epstein_pipeline.models.document import Document
from epstein_pipeline.models.registry import PersonRegistry

# Bates range patterns used by DOJ/EFTA releases
_BATES_PATTERNS = [
    re.compile(r"^EFTA\d{8}$"),                          # EFTA00039025
    re.compile(r"^EFTA\d{8}-EFTA\d{8}$"),                # EFTA00039025-EFTA00039030
    re.compile(r"^[A-Z]{2,10}\d{4,10}$"),                # DEF00001234
    re.compile(r"^[A-Z]{2,10}\d{4,10}-[A-Z]{2,10}\d{4,10}$"),  # DEF00001234-DEF00001240
    re.compile(r"^[A-Z]+-[A-Z]+-\d{4,10}$"),             # US-GOV-00001234
]

# Reasonable date range for Epstein case documents
_MIN_YEAR = 1950
_MAX_YEAR = 2026

# ISO date pattern (YYYY-MM-DD, with optional month/day)
_DATE_PATTERN = re.compile(
    r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$"
)


class IntegrityChecker:
    """Run integrity checks on a collection of documents.

    Checks include:
    1. All personIds reference real persons in the registry
    2. Dates are valid and within a reasonable range (1950-2026)
    3. No duplicate document IDs
    4. Bates ranges are properly formatted
    5. Required fields are non-empty
    """

    def __init__(self, registry: PersonRegistry) -> None:
        self._registry = registry
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, documents: list[Document]) -> list[str]:
        """Run all integrity checks on a list of documents.

        Parameters
        ----------
        documents:
            The documents to validate.

        Returns
        -------
        list[str]
            A list of human-readable error/warning messages.
            Empty if all checks pass.
        """
        errors: list[str] = []

        errors.extend(self._check_duplicate_ids(documents))
        errors.extend(self._check_required_fields(documents))
        errors.extend(self._check_dates(documents))
        errors.extend(self._check_person_ids(documents))
        errors.extend(self._check_bates_ranges(documents))

        # Print summary
        self._print_summary(documents, errors)

        return errors

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_duplicate_ids(self, documents: list[Document]) -> list[str]:
        """Check for duplicate document IDs."""
        errors: list[str] = []
        id_counts = Counter(doc.id for doc in documents)

        for doc_id, count in id_counts.items():
            if count > 1:
                errors.append(
                    f"DUPLICATE_ID: Document ID '{doc_id}' appears {count} times"
                )

        return errors

    def _check_required_fields(self, documents: list[Document]) -> list[str]:
        """Check that required fields are non-empty."""
        errors: list[str] = []

        for doc in documents:
            if not doc.id or not doc.id.strip():
                errors.append("EMPTY_FIELD: Document has empty 'id'")

            if not doc.title or not doc.title.strip():
                errors.append(
                    f"EMPTY_FIELD: Document '{doc.id}' has empty 'title'"
                )

            if not doc.source or not doc.source.strip():
                errors.append(
                    f"EMPTY_FIELD: Document '{doc.id}' has empty 'source'"
                )

            if not doc.category or not doc.category.strip():
                errors.append(
                    f"EMPTY_FIELD: Document '{doc.id}' has empty 'category'"
                )

        return errors

    def _check_dates(self, documents: list[Document]) -> list[str]:
        """Check that dates are valid ISO format and within range."""
        errors: list[str] = []

        for doc in documents:
            if doc.date is None:
                continue

            date_str = doc.date.strip()
            if not date_str:
                continue

            match = _DATE_PATTERN.match(date_str)
            if not match:
                errors.append(
                    f"INVALID_DATE: Document '{doc.id}' has invalid date "
                    f"format '{doc.date}' (expected YYYY-MM-DD)"
                )
                continue

            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else None
            day = int(match.group(3)) if match.group(3) else None

            if year < _MIN_YEAR or year > _MAX_YEAR:
                errors.append(
                    f"DATE_RANGE: Document '{doc.id}' has date year {year} "
                    f"outside range {_MIN_YEAR}-{_MAX_YEAR}"
                )

            if month is not None and (month < 1 or month > 12):
                errors.append(
                    f"INVALID_DATE: Document '{doc.id}' has invalid month "
                    f"{month} in date '{doc.date}'"
                )

            if day is not None and (day < 1 or day > 31):
                errors.append(
                    f"INVALID_DATE: Document '{doc.id}' has invalid day "
                    f"{day} in date '{doc.date}'"
                )

            # More precise day validation for known months
            if month is not None and day is not None:
                days_in_month = {
                    1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
                }
                max_days = days_in_month.get(month, 31)
                if day > max_days:
                    errors.append(
                        f"INVALID_DATE: Document '{doc.id}' has day {day} "
                        f"which exceeds maximum {max_days} for month {month}"
                    )

        return errors

    def _check_person_ids(self, documents: list[Document]) -> list[str]:
        """Check that all personIds reference persons in the registry."""
        errors: list[str] = []
        missing_ids: Counter[str] = Counter()

        for doc in documents:
            for person_id in doc.personIds:
                if person_id not in self._registry:
                    missing_ids[person_id] += 1
                    errors.append(
                        f"UNKNOWN_PERSON: Document '{doc.id}' references "
                        f"unknown person ID '{person_id}'"
                    )

        # Add a summary if there are many missing persons
        if len(missing_ids) > 10:
            top_missing = missing_ids.most_common(10)
            summary_lines = [
                f"  {pid}: {count} references" for pid, count in top_missing
            ]
            errors.append(
                f"PERSON_SUMMARY: {len(missing_ids)} unique unknown person IDs. "
                f"Top 10:\n" + "\n".join(summary_lines)
            )

        return errors

    def _check_bates_ranges(self, documents: list[Document]) -> list[str]:
        """Check that Bates ranges match expected patterns."""
        errors: list[str] = []

        for doc in documents:
            if doc.batesRange is None:
                continue

            bates = doc.batesRange.strip()
            if not bates:
                continue

            if not any(pattern.match(bates) for pattern in _BATES_PATTERNS):
                errors.append(
                    f"INVALID_BATES: Document '{doc.id}' has unrecognised "
                    f"Bates range format '{bates}'"
                )
                continue

            # For EFTA ranges, check that start <= end
            efta_range = re.match(
                r"^(EFTA)(\d{8})-(EFTA)(\d{8})$", bates
            )
            if efta_range:
                start_num = int(efta_range.group(2))
                end_num = int(efta_range.group(4))
                if start_num > end_num:
                    errors.append(
                        f"INVALID_BATES: Document '{doc.id}' has inverted "
                        f"Bates range (start {start_num} > end {end_num})"
                    )

        return errors

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_summary(
        self, documents: list[Document], errors: list[str]
    ) -> None:
        """Print a formatted summary of integrity check results."""
        self._console.print()
        self._console.rule("[bold cyan]Integrity Check Results[/bold cyan]")
        self._console.print()

        # Categorise errors
        categories: dict[str, int] = Counter()
        for err in errors:
            prefix = err.split(":")[0] if ":" in err else "OTHER"
            categories[prefix] += 1

        table = Table(show_lines=False)
        table.add_column("Check", style="bold")
        table.add_column("Errors", justify="right")
        table.add_column("Status")

        check_names = {
            "DUPLICATE_ID": "Duplicate IDs",
            "EMPTY_FIELD": "Required fields",
            "INVALID_DATE": "Date format",
            "DATE_RANGE": "Date range",
            "UNKNOWN_PERSON": "Person references",
            "PERSON_SUMMARY": "Person summary",
            "INVALID_BATES": "Bates format",
        }

        all_checks = [
            "DUPLICATE_ID", "EMPTY_FIELD", "INVALID_DATE",
            "DATE_RANGE", "UNKNOWN_PERSON", "INVALID_BATES",
        ]

        for check_key in all_checks:
            count = categories.get(check_key, 0)
            label = check_names.get(check_key, check_key)
            status = "[green]PASS[/green]" if count == 0 else f"[red]{count} error(s)[/red]"
            table.add_row(label, str(count), status)

        self._console.print(table)
        self._console.print()

        total_with_dates = sum(1 for d in documents if d.date)
        total_with_persons = sum(1 for d in documents if d.personIds)
        total_with_bates = sum(1 for d in documents if d.batesRange)

        self._console.print(f"Documents checked: {len(documents):,}")
        self._console.print(f"  With dates:      {total_with_dates:,}")
        self._console.print(f"  With person IDs: {total_with_persons:,}")
        self._console.print(f"  With Bates range: {total_with_bates:,}")
        self._console.print()

        if errors:
            # Filter out PERSON_SUMMARY from the count since it's a meta-error
            real_errors = [e for e in errors if not e.startswith("PERSON_SUMMARY")]
            self._console.print(
                f"[bold red]FAILED: {len(real_errors):,} error(s) found.[/bold red]"
            )
        else:
            self._console.print(
                "[bold green]PASSED: All integrity checks passed.[/bold green]"
            )
