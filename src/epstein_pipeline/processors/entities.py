"""Entity extraction using spaCy NER with person registry matching."""

from __future__ import annotations

import spacy
from spacy.language import Language

from epstein_pipeline.config import Settings
from epstein_pipeline.models.registry import PersonRegistry


class EntityExtractor:
    """Extract person IDs from text using spaCy NER and a PersonRegistry.

    The extraction pipeline works in two passes:
    1. **NER pass** -- Run the spaCy model to detect PERSON entities, then
       attempt to match each entity against the registry (exact + fuzzy).
    2. **Direct scan** -- Walk every name in the registry and check whether it
       appears verbatim in the text (catches names that spaCy missed).

    Results are deduplicated before being returned.
    """

    def __init__(self, config: Settings, registry: PersonRegistry) -> None:
        self.config = config
        self.registry = registry
        self._nlp: Language = spacy.load(config.spacy_model)

        # Pre-build a mapping of lowered canonical names to person IDs so the
        # direct scan can operate efficiently.
        self._name_to_id: dict[str, str] = {}
        for person_id, person in registry._persons_by_id.items():
            self._name_to_id[person.name.lower()] = person_id
            for alias in person.aliases:
                self._name_to_id[alias.lower()] = person_id

    def extract(self, text: str) -> list[str]:
        """Return a deduplicated list of person IDs found in *text*.

        Parameters
        ----------
        text:
            Free-form text (OCR output, email body, document summary, etc.).

        Returns
        -------
        list[str]
            Sorted list of unique person IDs (e.g. ``["p-0001", "p-0042"]``).
        """
        if not text or not text.strip():
            return []

        matched_ids: set[str] = set()

        # ----- Pass 1: spaCy NER -----
        # Process in chunks if text is very long to avoid memory issues.
        max_chars = 1_000_000
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        for chunk in chunks:
            doc = self._nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    person_id = self.registry.match(ent.text)
                    if person_id is not None:
                        matched_ids.add(person_id)

        # ----- Pass 2: Direct name scan -----
        text_lower = text.lower()
        for name_lower, person_id in self._name_to_id.items():
            # Skip very short names (1-2 chars) to avoid false positives.
            if len(name_lower) < 3:
                continue
            if name_lower in text_lower:
                matched_ids.add(person_id)

        return sorted(matched_ids)
