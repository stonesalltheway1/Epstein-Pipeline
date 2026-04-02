"""Tests for Splink entity resolution processor."""

from __future__ import annotations

import pytest

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Person
from epstein_pipeline.processors.entity_resolution import (
    EntityCluster,
    EntityResolver,
    PersonRecord,
    ResolutionResult,
)

# ── PersonRecord tests ──────────────────────────────────────────────────


class TestPersonRecord:
    def test_fields(self):
        r = PersonRecord(
            unique_id="p-0001",
            source="registry",
            name="Jeffrey Epstein",
            name_lower="jeffrey epstein",
            first_name="Jeffrey",
            last_name="Epstein",
        )
        assert r.unique_id == "p-0001"
        assert r.last_name == "Epstein"

    def test_defaults(self):
        r = PersonRecord(
            unique_id="p-0001",
            source="registry",
            name="Test",
            name_lower="test",
        )
        assert r.first_name is None
        assert r.aliases is None
        assert r.category is None
        assert r.document_ids is None


# ── PersonsToRecords tests ──────────────────────────────────────────────


class TestPersonsToRecords:
    def test_converts_persons(self):
        persons = [
            Person(
                id="p-0001",
                slug="jeffrey-epstein",
                name="Jeffrey Epstein",
                aliases=["Jeff Epstein", "JE"],
                category="convicted",
            ),
        ]
        records = EntityResolver.persons_to_records(persons)
        # Should create 1 main record + 2 alias records = 3
        assert len(records) == 3
        names = {r.name for r in records}
        assert "Jeffrey Epstein" in names
        assert "Jeff Epstein" in names
        assert "JE" in names

    def test_splits_names(self):
        persons = [
            Person(
                id="p-0001",
                slug="jeffrey-epstein",
                name="Jeffrey Epstein",
                category="convicted",
            ),
        ]
        records = EntityResolver.persons_to_records(persons)
        main = records[0]
        assert main.first_name == "Jeffrey"
        assert main.last_name == "Epstein"

    def test_single_name(self):
        persons = [
            Person(id="p-0001", slug="madonna", name="Madonna", category="other"),
        ]
        records = EntityResolver.persons_to_records(persons)
        assert records[0].first_name == "Madonna"
        assert records[0].last_name is None

    def test_no_aliases(self):
        persons = [
            Person(
                id="p-0001",
                slug="test",
                name="Test Person",
                category="other",
            ),
        ]
        records = EntityResolver.persons_to_records(persons)
        assert len(records) == 1
        assert records[0].aliases is None

    def test_alias_source_tagged(self):
        persons = [
            Person(
                id="p-0001",
                slug="test",
                name="Test Person",
                aliases=["TP"],
                category="other",
            ),
        ]
        records = EntityResolver.persons_to_records(persons)
        alias_record = [r for r in records if r.source == "registry_alias"]
        assert len(alias_record) == 1
        assert alias_record[0].name == "TP"

    def test_alias_unique_id_format(self):
        persons = [
            Person(
                id="p-0001",
                slug="test",
                name="Test Person",
                aliases=["TP"],
                category="other",
            ),
        ]
        records = EntityResolver.persons_to_records(persons)
        alias_record = [r for r in records if "::alias::" in r.unique_id]
        assert len(alias_record) == 1
        assert alias_record[0].unique_id == "p-0001::alias::TP"

    def test_empty_persons(self):
        records = EntityResolver.persons_to_records([])
        assert records == []


# ── ResolutionResult tests ──────────────────────────────────────────────


class TestResolutionResult:
    def test_empty_result(self):
        r = ResolutionResult()
        assert r.total_clusters == 0
        assert r.merge_map == {}
        assert r.clusters == []

    def test_merge_map(self):
        r = ResolutionResult(
            merge_map={"p-0002::alias::G. Maxwell": "p-0002"},
            total_input_records=5,
            total_clusters=3,
        )
        assert r.merge_map["p-0002::alias::G. Maxwell"] == "p-0002"


class TestEntityCluster:
    def test_defaults(self):
        c = EntityCluster(cluster_id=1)
        assert c.canonical_person_id is None
        assert c.canonical_name == ""
        assert c.records == []
        assert c.avg_match_probability == 0.0


# ── EntityResolver tests ────────────────────────────────────────────────


class TestEntityResolver:
    @pytest.fixture()
    def settings(self, tmp_path):
        return Settings(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / ".cache",
        )

    def test_init(self, settings):
        resolver = EntityResolver(settings)
        assert resolver.threshold == 0.85

    def test_init_custom_threshold(self, settings):
        settings.splink_match_probability_threshold = 0.95
        resolver = EntityResolver(settings)
        assert resolver.threshold == 0.95

    def test_resolve_too_few_records(self, settings):
        resolver = EntityResolver(settings)
        result = resolver.resolve(
            [
                PersonRecord(
                    unique_id="p-0001",
                    source="reg",
                    name="A",
                    name_lower="a",
                ),
            ]
        )
        assert result.total_clusters == 1
        assert len(result.merge_map) == 0

    def test_resolve_zero_records(self, settings):
        resolver = EntityResolver(settings)
        result = resolver.resolve([])
        assert result.total_clusters == 0
        assert result.total_input_records == 0
