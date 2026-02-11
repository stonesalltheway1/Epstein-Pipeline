"""Person linking -- attach person IDs to documents based on text content."""

from __future__ import annotations

from epstein_pipeline.models.document import Document
from epstein_pipeline.models.registry import PersonRegistry


class PersonLinker:
    """Link documents to persons by scanning title, summary, and OCR text.

    For each document the linker concatenates all available text fields
    (title, summary, ocrText) and searches for every known person name and
    alias.  Matched person IDs are written to ``doc.personIds``.

    This is intentionally a simple substring-based linker rather than a
    full NER pipeline -- it is fast enough to run over 100K+ documents in
    seconds and avoids the spaCy dependency for the common case.  For
    higher-precision extraction, use :class:`EntityExtractor` instead.
    """

    def __init__(self, registry: PersonRegistry) -> None:
        self.registry = registry

        # Build (lowered_name -> person_id) lookup covering canonical names
        # and all aliases.  Skip very short names to avoid false positives.
        self._name_to_id: dict[str, str] = {}
        for person_id, person in registry._persons_by_id.items():
            if len(person.name) >= 3:
                self._name_to_id[person.name.lower()] = person_id
            for alias in person.aliases:
                if len(alias) >= 3:
                    self._name_to_id[alias.lower()] = person_id

    def link_document(self, doc: Document) -> Document:
        """Set ``doc.personIds`` by scanning all text fields.

        Returns the same Document instance (mutated in place) for
        convenience, so callers can chain: ``linker.link_document(doc)``.
        """
        parts: list[str] = []
        if doc.title:
            parts.append(doc.title)
        if doc.summary:
            parts.append(doc.summary)
        if doc.ocrText:
            parts.append(doc.ocrText)

        if not parts:
            return doc

        combined_lower = "\n".join(parts).lower()
        matched: set[str] = set()

        for name_lower, person_id in self._name_to_id.items():
            if name_lower in combined_lower:
                matched.add(person_id)

        doc.personIds = sorted(matched)
        return doc

    def link_batch(self, docs: list[Document]) -> list[Document]:
        """Link person IDs for a list of documents.

        Each document is modified in place.  Returns the same list for
        convenience.
        """
        for doc in docs:
            self.link_document(doc)
        return docs
