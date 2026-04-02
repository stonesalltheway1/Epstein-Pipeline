"""Entity extraction using spaCy NER, GLiNER zero-shot, and regex patterns.

Supports four extraction backends (controlled via ``Settings.ner_backend``):
- **spacy** — spaCy NER with en_core_web_trf (transformer-based, 3x more accurate)
- **gliner** — GLiNER v1 zero-shot NER with custom legal entity types
- **gliner2** — GLiNER2 unified NER with richer extraction API
- **both** — Union merge from spaCy + GLiNER (recommended)

Optional coreference resolution (``Settings.enable_coref``):
- Pre-NER pronoun resolution using fastcoref
- Resolves "he", "she", "they" to named entities for 30-50% more mentions

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
    0. **Coreference** (optional) — resolve pronouns to named entities
    1. **spaCy NER** — detect standard NER types with en_core_web_trf
    2. **GLiNER** — zero-shot extraction for custom legal entity types
    3. **GLiNER2** — unified NER with richer extraction API
    4. **Regex** — pattern matching for structured entities (phone, email, etc.)
    5. **Registry scan** — direct name matching against person registry

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
        self._gliner2_model = None
        self._coref_model = None

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

    def _load_gliner2(self):
        """Lazy-load the GLiNER2 model."""
        if self._gliner2_model is not None:
            return self._gliner2_model

        try:
            from gliner2 import GLiNER2

            model_name = self.config.gliner2_model
            logger.info("Loading GLiNER2 model: %s", model_name)
            self._gliner2_model = GLiNER2.from_pretrained(model_name)
            return self._gliner2_model
        except ImportError:
            raise ImportError(
                "gliner2 is required for GLiNER2 NER. "
                "Install with: pip install 'epstein-pipeline[nlp-gliner2]'"
            )

    def _load_coref(self):
        """Lazy-load the fastcoref model for coreference resolution."""
        if self._coref_model is not None:
            return self._coref_model

        try:
            from fastcoref import FCoref, LingMessCoref

            model_cls = self.config.coref_model
            if model_cls == "LingMessCoref":
                logger.info("Loading LingMessCoref (accurate, slower)")
                self._coref_model = LingMessCoref(device="cpu")
            else:
                logger.info("Loading FCoref (fast)")
                self._coref_model = FCoref(device="cpu")
            return self._coref_model
        except ImportError:
            raise ImportError(
                "fastcoref is required for coreference resolution. "
                "Install with: pip install 'epstein-pipeline[nlp-coref]'"
            )

    def resolve_coreferences(self, text: str) -> str:
        """Resolve pronoun coreferences in text using fastcoref.

        Replaces pronouns ("he", "she", "they") with their antecedent
        entity names, increasing downstream NER recall by 30-50%.

        Returns the resolved text with pronouns replaced.
        """
        if not text or not text.strip():
            return text

        model = self._load_coref()

        try:
            preds = model.predict(texts=[text])
            if preds and hasattr(preds[0], "get_clusters"):
                clusters = preds[0].get_clusters(as_strings=False)
                if not clusters:
                    return text

                # Build replacement map: for each cluster, replace later
                # mentions with the first (longest) mention
                resolved = text
                replacements: list[tuple[int, int, str]] = []

                for cluster in clusters:
                    if len(cluster) < 2:
                        continue
                    # Pick the longest mention as the canonical name
                    mention_texts = [text[start:end] for start, end in cluster]
                    canonical = max(mention_texts, key=len)

                    # Replace later mentions (skip the canonical itself)
                    for start, end in cluster:
                        mention = text[start:end]
                        if mention != canonical and len(mention) < len(canonical):
                            replacements.append((start, end, canonical))

                # Apply replacements in reverse order to preserve offsets
                for start, end, replacement in sorted(replacements, reverse=True):
                    resolved = resolved[:start] + replacement + resolved[end:]

                logger.info(
                    "Coref: resolved %d pronoun mentions across %d clusters",
                    len(replacements),
                    len(clusters),
                )
                return resolved
        except Exception as exc:
            logger.warning("Coreference resolution failed, using original text: %s", exc)

        return text

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

        # --- Coreference resolution (optional, pre-NER) ---
        if self.config.enable_coref:
            try:
                text = self.resolve_coreferences(text)
            except ImportError as e:
                logger.warning("Coreference resolution skipped: %s", e)

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

        # --- GLiNER2 pass ---
        if self.backend == NerBackend.GLINER2:
            try:
                gliner2_entities = self._extract_gliner2(text)
                for ent in gliner2_entities:
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
                logger.warning("GLiNER2 extraction skipped: %s", e)

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

    def _extract_gliner2(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using GLiNER2 unified NER.

        GLiNER2 uses a different API than GLiNER v1:
        - ``from gliner2 import GLiNER2``
        - ``model.extract_entities(text, labels)`` returns grouped dict
        """
        model = self._load_gliner2()
        entities: list[ExtractedEntity] = []

        # GLiNER2 supports entity type descriptions for better accuracy
        gliner2_labels = {
            "person": "Names of people, including first and last names",
            "organization": "Company, institution, or government body names",
            "location": "Place names including cities, countries, addresses, properties",
            "date": "Dates, time periods, or temporal references",
            "case number": "Legal case numbers and docket references",
            "flight identifier": "Aircraft tail numbers or flight numbers",
            "financial amount": "Monetary amounts with currency",
        }

        max_chars = 4096
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        for chunk_idx, chunk in enumerate(chunks):
            char_offset = chunk_idx * max_chars
            try:
                result = model.extract_entities(
                    chunk,
                    gliner2_labels,
                    include_confidence=True,
                    include_spans=True,
                )
            except Exception as exc:
                logger.warning("GLiNER2 prediction failed on chunk: %s", exc)
                continue

            # GLiNER2 returns: {"entities": {"person": [...], "organization": [...]}}
            extracted = result.get("entities", {})
            for label_key, mentions in extracted.items():
                mapped_label = _GLINER_LABEL_MAP.get(label_key, label_key.upper())

                if isinstance(mentions, list):
                    for mention in mentions:
                        # mention can be a string or dict with text/confidence/start/end
                        if isinstance(mention, str):
                            entities.append(
                                ExtractedEntity(
                                    text=mention,
                                    label=mapped_label,
                                    start=char_offset,
                                    confidence=None,
                                    source="gliner2",
                                )
                            )
                        elif isinstance(mention, dict):
                            confidence = mention.get("confidence", 0.0)
                            if confidence and confidence < self.confidence_threshold:
                                continue
                            entities.append(
                                ExtractedEntity(
                                    text=mention.get("text", ""),
                                    label=mapped_label,
                                    start=(mention.get("start", 0) or 0) + char_offset,
                                    end=(mention.get("end", 0) or 0) + char_offset,
                                    confidence=(
                                        round(confidence, 4) if confidence else None
                                    ),
                                    source="gliner2",
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
