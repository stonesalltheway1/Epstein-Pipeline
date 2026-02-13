"""Tests for person linking."""

from epstein_pipeline.models.document import Document
from epstein_pipeline.processors.person_linker import PersonLinker


def test_link_document_by_title(person_registry):
    linker = PersonLinker(person_registry)
    doc = Document(
        id="doc-test",
        title="Jeffrey Epstein Financial Records",
        source="efta",
        category="financial",
    )
    linker.link_document(doc)
    assert "p-0001" in doc.personIds


def test_link_document_by_ocr_text(person_registry):
    linker = PersonLinker(person_registry)
    doc = Document(
        id="doc-test",
        title="Unknown Document",
        source="other",
        category="other",
        ocrText="Bill Clinton was mentioned in the document alongside Ghislaine Maxwell.",
    )
    linker.link_document(doc)
    assert "p-0002" in doc.personIds
    assert "p-0003" in doc.personIds


def test_link_document_no_match(person_registry):
    linker = PersonLinker(person_registry)
    doc = Document(
        id="doc-test",
        title="Random Document About Weather",
        source="other",
        category="other",
        ocrText="The weather forecast shows sunny skies.",
    )
    linker.link_document(doc)
    assert len(doc.personIds) == 0


def test_link_batch(person_registry):
    linker = PersonLinker(person_registry)
    docs = [
        Document(id="d1", title="Jeffrey Epstein records", source="efta", category="other"),
        Document(
            id="d2", title="Ghislaine Maxwell testimony", source="court-filing", category="legal"
        ),
        Document(id="d3", title="Random unrelated file", source="other", category="other"),
    ]
    linker.link_batch(docs)
    assert "p-0001" in docs[0].personIds
    assert "p-0002" in docs[1].personIds
    assert len(docs[2].personIds) == 0
