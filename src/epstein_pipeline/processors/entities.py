"""Entity extraction using spaCy NER with person registry matching.

Supports extraction of PERSON entities (matched against registry) and
optionally non-person entity types (ORG, GPE, DATE, MONEY, etc.)
via regex patterns for types spaCy doesn't natively handle well.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import spacy
from spacy.language import Language

from epstein_pipeline.config import Settings
from epstein_pipeline.models.registry import PersonRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom regex patterns for entities spaCy misses
# ---------------------------------------------------------------------------

_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_ACCOUNT_PATTERN = re.compile(r"\b(?:account|acct|a/c)[\s#:]*\d{4,}\b", re.IGNORECASE)
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)\b"
)

REGEX_EXTRACTORS: dict[str, re.Pattern[str]] = {
    "PHONE": _PHONE_PATTERN,
    "EMAIL_ADDR": _EMAIL_PATTERN,
    "ACCOUNT": _ACCOUNT_PATTERN,
    "ADDRESS": _ADDRESS_PATTERN,
}


@dataclass
class ExtractedEntity:
    """A single extracted entity mention."""

    text: str
    label: str  # PERSON, ORG, GPE, DATE, MONEY, PHONE, EMAIL_ADDR, etc.
    person_id: str | None = None  # Only for PERSON entities matched to registry
    start: int | None = None
    end: int | None = None


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from a document."""

    person_ids: list[str] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)


class EntityExtractor:
    """Extract entities from text using spaCy NER and a PersonRegistry.

    The extraction pipeline works in two passes:
    1. **NER pass** -- Run the spaCy model to detect PERSON entities, then
       attempt to match each entity against the registry (exact + fuzzy).
    2. **Direct scan** -- Walk every name in the registry and check whether it
       appears verbatim in the text (catches names that spaCy missed).

    When ``entity_types`` includes non-PERSON types, those are also extracted
    from the spaCy output plus custom regex patterns.
    """

    # spaCy entity types we care about
    SPACY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "LOC", "NORP", "EVENT"}

    def __init__(
        self,
        config: Settings,
        registry: PersonRegistry,
        entity_types: set[str] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.entity_types = entity_types or {"PERSON"}
        self._nlp: Language = spacy.load(config.spacy_model)

        # Pre-build a mapping of lowered canonical names to person IDs
        self._name_to_id: dict[str, str] = {}
        for person_id, person in registry._persons_by_id.items():
            self._name_to_id[person.name.lower()] = person_id
            for alias in person.aliases:
                self._name_to_id[alias.lower()] = person_id

    def extract(self, text: str) -> list[str]:
        """Return a deduplicated list of person IDs found in *text*.

        This is the backward-compatible API that returns only person IDs.
        """
        result = self.extract_all(text)
        return result.person_ids

    def extract_all(self, text: str) -> EntityExtractionResult:
        """Return full extraction results including all entity types."""
        if not text or not text.strip():
            return EntityExtractionResult()

        matched_ids: set[str] = set()
        entities: list[ExtractedEntity] = []

        # Process in chunks if text is very long
        max_chars = 1_000_000
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        for chunk in chunks:
            doc = self._nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    person_id = self.registry.match(ent.text)
                    if person_id is not None:
                        matched_ids.add(person_id)
                    entities.append(
                        ExtractedEntity(
                            text=ent.text,
                            label="PERSON",
                            person_id=person_id,
                            start=ent.start_char,
                            end=ent.end_char,
                        )
                    )
                elif ent.label_ in self.entity_types and ent.label_ in self.SPACY_TYPES:
                    entities.append(
                        ExtractedEntity(
                            text=ent.text,
                            label=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                        )
                    )

        # Direct name scan for person IDs
        text_lower = text.lower()
        for name_lower, person_id in self._name_to_id.items():
            if len(name_lower) < 3:
                continue
            if name_lower in text_lower:
                matched_ids.add(person_id)

        # Regex-based entity extraction
        for entity_type, pattern in REGEX_EXTRACTORS.items():
            if entity_type in self.entity_types or "all" in self.entity_types:
                for match in pattern.finditer(text):
                    entities.append(
                        ExtractedEntity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        return EntityExtractionResult(
            person_ids=sorted(matched_ids),
            entities=entities,
        )

    def extract_batch(
        self,
        texts: list[tuple[str, str]],
        max_workers: int = 4,
    ) -> dict[str, EntityExtractionResult]:
        """Extract entities from multiple texts using ThreadPoolExecutor.

        Parameters
        ----------
        texts:
            List of (doc_id, text) tuples.
        max_workers:
            Number of threads for concurrent extraction.

        Returns
        -------
        dict[str, EntityExtractionResult]
            Mapping of doc_id to extraction results.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[str, EntityExtractionResult] = {}

        if max_workers <= 1:
            for doc_id, text in texts:
                results[doc_id] = self.extract_all(text)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.extract_all, text): doc_id for doc_id, text in texts
            }
            for future in as_completed(future_to_id):
                doc_id = future_to_id[future]
                try:
                    results[doc_id] = future.result()
                except Exception as exc:
                    logger.error("Entity extraction failed for %s: %s", doc_id, exc)
                    results[doc_id] = EntityExtractionResult()

        return results
