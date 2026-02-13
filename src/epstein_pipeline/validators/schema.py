"""Schema validation for Epstein Pipeline data files.

Validates raw dicts and JSON files against the Document Pydantic model,
reporting structured error messages for any fields that fail validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from epstein_pipeline.models.document import Document


class SchemaValidator:
    """Validate document data against the Pydantic Document schema."""

    def __init__(self) -> None:
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_document(self, data: dict) -> list[str]:
        """Validate a single document dict against the Document model.

        Parameters
        ----------
        data:
            A dict representing a single document (e.g. from parsed JSON).

        Returns
        -------
        list[str]
            A list of human-readable error messages.  Empty if valid.
        """
        try:
            Document.model_validate(data)
            return []
        except ValidationError as exc:
            return self._format_validation_errors(exc, data.get("id", "<unknown>"))

    def validate_file(self, path: Path) -> list[str]:
        """Load a JSON file and validate all documents in it.

        The file must contain either a JSON array of document objects or a
        single document object.

        Parameters
        ----------
        path:
            Path to a JSON file.

        Returns
        -------
        list[str]
            A list of all validation error messages across all documents.
        """
        path = Path(path)
        errors: list[str] = []

        if not path.exists():
            return [f"File not found: {path}"]

        if not path.suffix.lower() == ".json":
            return [f"Expected a .json file, got: {path.suffix}"]

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [f"Invalid JSON in {path.name}: {exc}"]

        # Handle single object or array
        if isinstance(raw, dict):
            documents = [raw]
        elif isinstance(raw, list):
            documents = raw
        else:
            return [f"Expected a JSON object or array in {path.name}, got {type(raw).__name__}"]

        for idx, doc_data in enumerate(documents):
            if not isinstance(doc_data, dict):
                errors.append(f"{path.name}[{idx}]: Expected object, got {type(doc_data).__name__}")
                continue

            doc_errors = self.validate_document(doc_data)
            for err in doc_errors:
                errors.append(f"{path.name}[{idx}]: {err}")

        # Print summary
        valid_count = len(documents) - sum(
            1 for doc in documents if isinstance(doc, dict) and self.validate_document(doc)
        )

        if errors:
            self._console.print(
                f"[yellow]{path.name}:[/yellow] {valid_count}/{len(documents)} valid, "
                f"{len(errors)} error(s)"
            )
        else:
            self._console.print(
                f"[green]{path.name}:[/green] {len(documents)} documents, all valid"
            )

        return errors

    def validate_directory(self, dir_path: Path) -> dict[str, list[str]]:
        """Validate all JSON files in a directory.

        Parameters
        ----------
        dir_path:
            Path to a directory containing JSON files.

        Returns
        -------
        dict[str, list[str]]
            Mapping of filename to list of error messages.
            Files with no errors are included with an empty list.
        """
        dir_path = Path(dir_path)
        results: dict[str, list[str]] = {}

        if not dir_path.is_dir():
            self._console.print(f"[red]Not a directory: {dir_path}[/red]")
            return {str(dir_path): [f"Not a directory: {dir_path}"]}

        json_files = sorted(dir_path.glob("*.json"))

        if not json_files:
            self._console.print(f"[yellow]No JSON files found in {dir_path}[/yellow]")
            return {}

        self._console.print(f"[cyan]Validating {len(json_files)} JSON file(s) in {dir_path}[/cyan]")
        self._console.print()

        total_docs = 0
        total_errors = 0

        for json_file in json_files:
            errors = self.validate_file(json_file)
            results[json_file.name] = errors
            total_errors += len(errors)

            # Count docs for summary
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    total_docs += len(raw)
                elif isinstance(raw, dict):
                    total_docs += 1
            except (json.JSONDecodeError, OSError):
                pass

        # Print summary table
        self._console.print()
        table = Table(title="Validation Summary", title_style="bold cyan")
        table.add_column("File", style="bold")
        table.add_column("Errors", justify="right")
        table.add_column("Status")

        for filename, errors in results.items():
            if errors:
                table.add_row(
                    filename,
                    str(len(errors)),
                    "[red]FAIL[/red]",
                )
            else:
                table.add_row(
                    filename,
                    "0",
                    "[green]PASS[/green]",
                )

        self._console.print(table)
        self._console.print()
        self._console.print(
            f"Total: {total_docs:,} documents, {total_errors:,} errors "
            f"across {len(json_files)} files"
        )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_validation_errors(exc: ValidationError, doc_id: str) -> list[str]:
        """Convert a Pydantic ValidationError into readable strings."""
        errors: list[str] = []

        for err in exc.errors():
            location = " -> ".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            err_type = err["type"]
            errors.append(f"[{doc_id}] {location}: {msg} (type={err_type})")

        return errors
