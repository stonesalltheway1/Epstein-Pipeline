"""Tests for entity extraction."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from epstein_pipeline.models.registry import PersonRegistry


def test_registry_exact_match(tmp_path: Path):
    registry_file = tmp_path / "persons.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "id": "p-0001",
                    "slug": "jeffrey-epstein",
                    "name": "Jeffrey Epstein",
                    "aliases": ["Jeff Epstein", "JE"],
                    "category": "convicted",
                },
                {
                    "id": "p-0002",
                    "slug": "ghislaine-maxwell",
                    "name": "Ghislaine Maxwell",
                    "aliases": ["G. Maxwell"],
                    "category": "convicted",
                },
            ]
        )
    )

    registry = PersonRegistry(registry_file)
    assert registry.match("Jeffrey Epstein") == "p-0001"
    assert registry.match("Jeff Epstein") == "p-0001"
    assert registry.match("Ghislaine Maxwell") == "p-0002"
    assert registry.match("G. Maxwell") == "p-0002"


def test_registry_fuzzy_match(tmp_path: Path):
    registry_file = tmp_path / "persons.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "id": "p-0001",
                    "slug": "jeffrey-epstein",
                    "name": "Jeffrey Epstein",
                    "aliases": [],
                    "category": "convicted",
                },
            ]
        )
    )

    registry = PersonRegistry(registry_file)
    # Fuzzy match should catch close variants
    result = registry.match("Jeffery Epstein")  # Common misspelling
    assert result == "p-0001"


def test_registry_no_match(tmp_path: Path):
    registry_file = tmp_path / "persons.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "id": "p-0001",
                    "slug": "jeffrey-epstein",
                    "name": "Jeffrey Epstein",
                    "aliases": [],
                    "category": "convicted",
                },
            ]
        )
    )

    registry = PersonRegistry(registry_file)
    assert registry.match("John Smith") is None
    assert registry.match("") is None
