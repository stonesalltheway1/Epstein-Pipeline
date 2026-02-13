"""Confidence scoring for entity matches and document links.

Provides numeric confidence values for:
- Entity-to-person matches (exact, alias, fuzzy, substring)
- Document-to-person links (based on independent signal count)
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from epstein_pipeline.models.registry import PersonRegistry


@dataclass
class MatchScore:
    """A scored match between a text mention and a person."""

    person_id: str
    confidence: float  # 0.0 - 1.0
    match_type: str  # "exact", "alias", "fuzzy", "substring"
    matched_name: str  # The name that matched


class ConfidenceScorer:
    """Score entity matches and document links with confidence values."""

    # Confidence tiers
    EXACT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    FUZZY_95_CONFIDENCE = 0.85
    FUZZY_90_CONFIDENCE = 0.75
    SUBSTRING_CONFIDENCE = 0.60

    def __init__(self, registry: PersonRegistry) -> None:
        self.registry = registry

        # Build lookup tables
        self._exact_names: dict[str, str] = {}  # normalized name -> person_id
        self._alias_names: dict[str, str] = {}  # normalized alias -> person_id
        self._all_names: list[tuple[str, str]] = []  # [(normalized, person_id)]

        for person_id, person in registry._persons_by_id.items():
            norm = self._normalize(person.name)
            self._exact_names[norm] = person_id
            self._all_names.append((norm, person_id))
            for alias in person.aliases:
                norm_alias = self._normalize(alias)
                self._alias_names[norm_alias] = person_id
                self._all_names.append((norm_alias, person_id))

    def score_entity_match(self, mention: str) -> MatchScore | None:
        """Score a text mention against the person registry.

        Returns the best match with a confidence score, or None if no
        match meets the minimum threshold.

        Scoring tiers:
        - Exact canonical name match: 1.0
        - Exact alias match: 0.95
        - Fuzzy match > 95%: 0.85
        - Fuzzy match > 90%: 0.75
        - Substring match (name in text): 0.60
        """
        norm = self._normalize(mention)
        if not norm or len(norm) < 3:
            return None

        # Tier 1: Exact canonical match
        if norm in self._exact_names:
            pid = self._exact_names[norm]
            person = self.registry.get(pid)
            return MatchScore(
                person_id=pid,
                confidence=self.EXACT_CONFIDENCE,
                match_type="exact",
                matched_name=person.name if person else norm,
            )

        # Tier 2: Exact alias match
        if norm in self._alias_names:
            pid = self._alias_names[norm]
            return MatchScore(
                person_id=pid,
                confidence=self.ALIAS_CONFIDENCE,
                match_type="alias",
                matched_name=norm,
            )

        # Tier 3-4: Fuzzy matching
        best_score = 0.0
        best_name = ""
        best_pid = ""

        for name, pid in self._all_names:
            score = fuzz.token_sort_ratio(norm, name) / 100.0
            if score > best_score:
                best_score = score
                best_name = name
                best_pid = pid

        if best_score >= 0.95:
            return MatchScore(
                person_id=best_pid,
                confidence=self.FUZZY_95_CONFIDENCE,
                match_type="fuzzy",
                matched_name=best_name,
            )
        elif best_score >= 0.90:
            return MatchScore(
                person_id=best_pid,
                confidence=self.FUZZY_90_CONFIDENCE,
                match_type="fuzzy",
                matched_name=best_name,
            )

        return None

    def score_document_link(
        self,
        person_id: str,
        signals: dict[str, bool],
    ) -> float:
        """Score a document-person link based on independent signals.

        Parameters
        ----------
        person_id:
            The person being linked.
        signals:
            A dict of signal names to presence booleans. Expected keys:
            - "ner_match": spaCy NER detected the person
            - "direct_scan": Name found via substring scan
            - "title_mention": Name appears in document title
            - "bates_match": Bates range associated with person
            - "metadata_match": Person in document metadata

        Returns
        -------
        float
            Confidence score from 0.0 to 1.0.
        """
        weights = {
            "ner_match": 0.25,
            "direct_scan": 0.20,
            "title_mention": 0.30,
            "bates_match": 0.15,
            "metadata_match": 0.10,
        }

        total = 0.0
        for signal_name, present in signals.items():
            if present and signal_name in weights:
                total += weights[signal_name]

        return min(total, 1.0)

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase and collapse whitespace."""
        return " ".join(name.lower().split())
