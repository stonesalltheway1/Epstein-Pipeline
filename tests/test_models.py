"""Tests for Pydantic data models."""

from epstein_pipeline.models import Document, Email, EmailContact, Flight, Person


def test_document_minimal():
    doc = Document(id="doc-001", title="Test Document", source="other", category="other")
    assert doc.id == "doc-001"
    assert doc.personIds == []
    assert doc.tags == []


def test_document_full():
    doc = Document(
        id="doc-002",
        title="EFTA Filing",
        date="2024-01-15",
        source="efta",
        category="legal",
        summary="A test filing",
        personIds=["p-0001", "p-0002"],
        tags=["financial", "estate"],
        pdfUrl="https://example.com/doc.pdf",
        pageCount=5,
        batesRange="EFTA00039025-EFTA00039030",
    )
    assert doc.source == "efta"
    assert len(doc.personIds) == 2


def test_person():
    person = Person(
        id="p-0001",
        slug="jeffrey-epstein",
        name="Jeffrey Epstein",
        aliases=["Jeff Epstein", "JE"],
        category="convicted",
        shortBio="Convicted sex trafficker",
    )
    assert person.slug == "jeffrey-epstein"
    assert len(person.aliases) == 2


def test_email():
    email = Email(
        id="email-001",
        subject="Meeting tomorrow",
        **{"from": EmailContact(name="John Doe", email="john@example.com")},
        to=[EmailContact(name="Jane Doe")],
        date="2005-03-15",
        body="Let's meet at 3pm.",
    )
    assert email.subject == "Meeting tomorrow"


def test_flight():
    flight = Flight(
        id="flight-001",
        date="2002-06-15",
        aircraft="Gulfstream II",
        tailNumber="N908JE",
        origin="Teterboro, NJ",
        destination="St. Thomas, USVI",
        passengerIds=["p-0001", "p-0002"],
    )
    assert flight.tailNumber == "N908JE"
    assert len(flight.passengerIds) == 2


def test_document_json_roundtrip():
    doc = Document(
        id="doc-003",
        title="Test",
        source="court-filing",
        category="legal",
        personIds=["p-0001"],
    )
    data = doc.model_dump()
    doc2 = Document(**data)
    assert doc2.id == doc.id
    assert doc2.personIds == doc.personIds
