"""Entity extraction using spaCy NER, GLiNER zero-shot, and regex patterns.

Supports three extraction backends (controlled via ``Settings.ner_backend``):
- **spacy** — spaCy NER with en_core_web_trf (transformer-based, 3x more accurate)
- **gliner** — GLiNER zero-shot NER with custom legal entity types
- **both** — Union merge from both extractors (recommended)

Custom entity types for the Epstein case files:
PERSON, ORGANIZATION, LOCATION, DATE, CASE_NUMBER, FLIGHT_ID,
PROPERTY_ADDRESS, FINANCIAL_AMOUNT, PHONE_NUMBER, EMAIL_ADDRESS
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from epstein_pipeline.config import NerBackend, Settings
from epstein_pipeline.models.document import EntityResult
from epstein_pipeline.models.registry import PersonRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom regex patterns for entities NER models often miss
# ---------------------------------------------------------------------------

_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_ACCOUNT_PATTERN = re.compile(r"\b(?:account|acct|a/c)[\s#:]*\d{4,}\b", re.IGNORECASE)
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)\b"
)
_CASE_NUMBER_PATTERN = re.compile(
    r"\b(?:Case|No\.|Docket|Cause)\s*(?:#|No\.?)?\s*\d[\d\-A-Z:/ ]{3,20}\b",
    re.IGNORECASE,
)
_FLIGHT_ID_PATTERN = re.compile(
    r"\b(?:N\d{1,5}[A-Z]{1,2}|(?:Flight|Flt)\s*#?\s*\d{1,6})\b",
    re.IGNORECASE,
)
_FINANCIAL_PATTERN = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b",
    re.IGNORECASE,
)

REGEX_EXTRACTORS: dict[str, re.Pattern[str]] = {
    "PHONE": _PHONE_PATTERN,
    "EMAIL_ADDR": _EMAIL_PATTERN,
    "ACCOUNT": _ACCOUNT_PATTERN,
    "ADDRESS": _ADDRESS_PATTERN,
    "CASE_NUMBER": _CASE_NUMBER_PATTERN,
    "FLIGHT_ID": _FLIGHT_ID_PATTERN,
    "FINANCIAL_AMOUNT": _FINANCIAL_PATTERN,
}

# GLiNER entity labels for zero-shot extraction
GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "case number",
    "flight identifier",
    "property address",
    "financial amount",
    "phone number",
    "email address",
]

# Map GLiNER labels to our standard entity types
_GLINER_LABEL_MAP: dict[str, str] = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "date": "DATE",
    "case number": "CASE_NUMBER",
    "flight identifier": "FLIGHT_ID",
    "property address": "PROPERTY_ADDRESS",
    "financial amount": "FINANCIAL_AMOUNT",
    "phone number": "PHONE",
    "email address": "EMAIL_ADDR",
}


@dataclass
class ExtractedEntity:
    """A single extracted entity mention."""

    text: str
    label: str  # PERSON, ORG, GPE, DATE, MONEY, PHONE, EMAIL_ADDR, etc.
    person_id: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    source: str = "spacy"  # 'spacy', 'gliner', 'regex'


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from a document."""

    person_ids: list[str] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)
    entity_results: list[EntityResult] = field(default_factory=list)


class EntityExtractor:
    """Extract entities from text using spaCy NER, GLiNER, and regex.

    The extraction pipeline:
    1. **spaCy NER** — detect standard NER types with en_core_web_trf
    2. **GLiNER** — zero-shot extraction for custom legal entity types
    3. **Regex** — pattern matching for structured entities (phone, email, etc.)
    4. **Registry scan** — direct name matching against person registry

    Results from all backends are merged with dedup.
    """

    SPACY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "LOC", "NORP", "EVENT"}

    def __init__(
        self,
        config: Settings,
        registry: PersonRegistry,
        entity_types: set[str] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.entity_types = entity_types or {"PERSON", "all"}
        self.backend = config.ner_backend
        self.confidence_threshold = config.ner_confidence_threshold

        # Lazy-loaded models
        self._nlp = None
        self._gliner_model = None

        # Pre-build name lookup
        self._name_to_id: dict[str, str] = {}
        for person_id, person in registry._persons_by_id.items():
            self._name_to_id[person.name.lower()] = person_id
            for alias in person.aliases:
                self._name_to_id[alias.lower()] = person_id

    def _load_spacy(self):
        """Lazy-load the spaCy model."""
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy

            model_name = self.config.spacy_model
            logger.info("Loading spaCy model: %s", model_name)

            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                logger.warning("%s not found, falling back to en_core_web_sm", model_name)
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    raise ImportError(
                        f"No spaCy model found. Install with: python -m spacy download {model_name}"
                    )
        except ImportError:
            raise ImportError(
                "spaCy is required for NER. Install with: pip install 'epstein-pipeline[nlp]'"
            )

        return self._nlp

    def _load_gliner(self):
        """Lazy-load the GLiNER model."""
        if self._gliner_model is not None:
            return self._gliner_model

        try:
            from gliner import GLiNER

            model_name = self.config.gliner_model
            logger.info("Loading GLiNER model: %s", model_name)
            self._gliner_model = GLiNER.from_pretrained(model_name)
            return self._gliner_model
        except ImportError:
            raise ImportError(
                "GLiNER is required for zero-shot NER. "
                "Install with: pip install 'epstein-pipeline[nlp-gliner]'"
            )

    def extract(self, text: str) -> list[str]:
        """Return a deduplicated list of person IDs found in *text*.

        Backward-compatible API that returns only person IDs.
        """
        result = self.extract_all(text)
        return result.person_ids

    def extract_all(self, text: str) -> EntityExtractionResult:
        """Return full extraction results from all configured backends."""
        if not text or not text.strip():
            return EntityExtractionResult()

        matched_ids: set[str] = set()
        entities: list[ExtractedEntity] = []
        seen_entities: set[tuple[str, str, int | None]] = set()

        # --- spaCy pass ---
        if self.backend in (NerBackend.SPACY, NerBackend.BOTH):
            try:
                spacy_entities = self._extract_spacy(text)
                for ent in spacy_entities:
                    key = (ent.label, ent.text.lower().strip(), ent.start)
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append(ent)
                        if ent.person_id:
                            matched_ids.add(ent.person_id)
            except ImportError as e:
                logger.warning("spaCy extraction skipped: %s", e)

        # --- GLiNER pass ---
        if self.backend in (NerBackend.GLINER, NerBackend.BOTH):
            try:
                gliner_entities = self._extract_gliner(text)
                for ent in gliner_entities:
                    key = (ent.label, ent.text.lower().strip(), ent.start)
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append(ent)
                        if ent.label == "PERSON":
                            person_id = self.registry.match(ent.text)
                            if person_id:
                                ent.person_id = person_id
                                matched_ids.add(person_id)
            except ImportError as e:
                logger.warning("GLiNER extraction skipped: %s", e)

        # --- Regex pass ---
        for entity_type, pattern in REGEX_EXTRACTORS.items():
            if entity_type in self.entity_types or "all" in self.entity_types:
                for match in pattern.finditer(text):
                    key = (entity_type, match.group().lower().strip(), match.start())
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append(
                            ExtractedEntity(
                                text=match.group(),
                                label=entity_type,
                                start=match.start(),
                                end=match.end(),
                                confidence=1.0,
                                source="regex",
                            )
                        )

        # --- Direct registry scan ---
        text_lower = text.lower()
        for name_lower, person_id in self._name_to_id.items():
            if len(name_lower) < 3:
                continue
            if name_lower in text_lower:
                matched_ids.add(person_id)

        # --- Build EntityResult list for ProcessingResult ---
        entity_results = [
            EntityResult(
                entity_type=e.label,
                value=e.text,
                confidence=e.confidence,
                source=e.source,
                span=text[max(0, (e.start or 0) - 30) : (e.end or 0) + 30]
                if e.start is not None
                else None,
            )
            for e in entities
        ]

        return EntityExtractionResult(
            person_ids=sorted(matched_ids),
            entities=entities,
            entity_results=entity_results,
        )

    def _extract_spacy(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        nlp = self._load_spacy()
        entities: list[ExtractedEntity] = []

        max_chars = 1_000_000
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    person_id = self.registry.match(ent.text)
                    entities.append(
                        ExtractedEntity(
                            text=ent.text,
                            label="PERSON",
                            person_id=person_id,
                            start=ent.start_char,
                            end=ent.end_char,
                            source="spacy",
                        )
                    )
                elif ent.label_ in self.SPACY_TYPES:
                    entities.append(
                        ExtractedEntity(
                            text=ent.text,
                            label=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                            source="spacy",
                        )
                    )

        return entities

    def _extract_gliner(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using GLiNER zero-shot NER."""
        model = self._load_gliner()
        entities: list[ExtractedEntity] = []

        max_chars = 4096
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        for chunk_idx, chunk in enumerate(chunks):
            char_offset = chunk_idx * max_chars
            try:
                predictions = model.predict_entities(chunk, GLINER_LABELS, threshold=0.3)
            except Exception as exc:
                logger.warning("GLiNER prediction failed on chunk: %s", exc)
                continue

            for pred in predictions:
                label = _GLINER_LABEL_MAP.get(pred.get("label", ""), pred.get("label", "UNKNOWN"))
                confidence = pred.get("score", 0.0)

                if confidence < self.confidence_threshold:
                    continue

                entities.append(
                    ExtractedEntity(
                        text=pred.get("text", ""),
                        label=label,
                        start=(pred.get("start", 0) or 0) + char_offset,
                        end=(pred.get("end", 0) or 0) + char_offset,
                        confidence=round(confidence, 4),
                        source="gliner",
                    )
                )

        return entities

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
