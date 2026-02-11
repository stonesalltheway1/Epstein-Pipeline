"""Person registry with exact and fuzzy name-to-ID matching."""

from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz, process

from epstein_pipeline.models.document import Person


class PersonRegistry:
    """In-memory person lookup loaded from persons-registry.json.

    The registry supports exact name matching (including aliases) and
    falls back to fuzzy matching via rapidfuzz when no exact hit is found.

    Expected JSON format::

        [
            {
                "id": "p-0001",
                "slug": "jeffrey-epstein",
                "name": "Jeffrey Epstein",
                "aliases": ["Jeff Epstein", "JE"],
                "category": "key-figure",
                "shortBio": "..."
            },
            ...
        ]
    """

    def __init__(self, persons: list[Person]) -> None:
        self._persons_by_id: dict[str, Person] = {p.id: p for p in persons}

        # Build a normalised-name -> person ID lookup for exact matching.
        # Includes both canonical names and all aliases.
        self._name_to_id: dict[str, str] = {}
        self._all_names: list[str] = []

        for p in persons:
            canonical = self._normalise(p.name)
            self._name_to_id[canonical] = p.id
            self._all_names.append(canonical)
            for alias in p.aliases:
                norm_alias = self._normalise(alias)
                self._name_to_id[norm_alias] = p.id
                self._all_names.append(norm_alias)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> PersonRegistry:
        """Load the registry from a persons-registry.json file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        persons = [Person.model_validate(entry) for entry in raw]
        return cls(persons)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, name: str, threshold: int = 85) -> str | None:
        """Return the person ID for *name*, or ``None`` if no match.

        1. Try an exact normalised lookup first (O(1)).
        2. Fall back to fuzzy token-sort-ratio matching via rapidfuzz.
           Only returns a match if the best score meets *threshold* (0-100).
        """
        norm = self._normalise(name)

        # Exact hit
        if norm in self._name_to_id:
            return self._name_to_id[norm]

        # Fuzzy fallback
        if not self._all_names:
            return None

        result = process.extractOne(
            norm,
            self._all_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            return None

        matched_name, _score, _index = result
        return self._name_to_id.get(matched_name)

    def get(self, person_id: str) -> Person | None:
        """Return the Person for *person_id*, or ``None``."""
        return self._persons_by_id.get(person_id)

    def __len__(self) -> int:
        return len(self._persons_by_id)

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._persons_by_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(name: str) -> str:
        """Lowercase, strip, and collapse whitespace."""
        return " ".join(name.lower().split())
