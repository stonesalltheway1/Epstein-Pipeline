"""Dataclass models for IRS Form 990 nonprofit cross-referencing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Nonprofit990Officer:
    """Officer/trustee from a 990 filing."""

    name: str
    title: str = ""
    hours_per_week: float = 0.0
    compensation: int = 0  # dollars
    compensation_related: int = 0
    other_compensation: int = 0
    is_former: bool = False
    filing_year: int = 0
    match_person_id: str | None = None
    match_score: float = 0.0


@dataclass
class Nonprofit990Grant:
    """Grant paid from a 990/990-PF filing."""

    recipient_name: str
    recipient_ein: str = ""
    recipient_city: str = ""
    recipient_state: str = ""
    amount: int = 0  # dollars
    purpose: str = ""
    filing_year: int = 0


@dataclass
class Nonprofit990RelatedOrg:
    """Related organization from Schedule R."""

    name: str
    ein: str = ""
    relationship_type: str = ""
    direct_controlling_entity: str = ""


@dataclass
class Nonprofit990Filing:
    """Single filing year financial summary."""

    tax_period: str = ""  # YYYYMM
    tax_year: int = 0
    form_type: str = ""  # 990, 990-EZ, 990-PF
    total_revenue: int = 0
    total_expenses: int = 0
    total_assets: int = 0
    total_liabilities: int = 0
    grants_paid: int = 0
    contributions_received: int = 0
    officer_comp_total: int = 0
    pdf_url: str = ""
    irs_object_id: str = ""
    officers: list[Nonprofit990Officer] = field(default_factory=list)
    grants: list[Nonprofit990Grant] = field(default_factory=list)
    related_orgs: list[Nonprofit990RelatedOrg] = field(default_factory=list)


@dataclass
class Nonprofit990Org:
    """Organization record with all filings."""

    ein: str
    name: str
    city: str = ""
    state: str = ""
    ntee_code: str = ""
    subsection_code: str = ""  # 501(c)(3), etc.
    ruling_date: str = ""
    category: str = ""  # education, health, foundation, charity, other
    propublica_url: str = ""
    filings: list[Nonprofit990Filing] = field(default_factory=list)
    all_officers: list[Nonprofit990Officer] = field(default_factory=list)
    all_grants: list[Nonprofit990Grant] = field(default_factory=list)
    all_related_orgs: list[Nonprofit990RelatedOrg] = field(default_factory=list)


@dataclass
class NonprofitPersonMatch:
    """A person matched to a nonprofit officer role."""

    person_id: str
    person_name: str
    officer_name: str
    org_ein: str
    org_name: str
    title: str = ""
    filing_year: int = 0
    compensation: int = 0
    match_score: float = 0.0


@dataclass
class NonprofitSearchResult:
    """Top-level output from a nonprofit cross-reference run."""

    total_orgs_found: int = 0
    total_filings: int = 0
    total_officers: int = 0
    total_grants: int = 0
    total_person_matches: int = 0
    checked_at: str = ""
    organizations: list[Nonprofit990Org] = field(default_factory=list)
    person_matches: list[NonprofitPersonMatch] = field(default_factory=list)
