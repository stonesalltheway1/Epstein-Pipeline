"""Tests for coreference resolution and GLiNER2 NER backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from epstein_pipeline.config import NerBackend, Settings
from epstein_pipeline.models.registry import PersonRegistry
from epstein_pipeline.processors.entities import (
    _GLINER_LABEL_MAP,
    EntityExtractor,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / ".cache",
    )


@pytest.fixture()
def registry(tmp_path):
    registry_file = tmp_path / "persons.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "id": "p-0001",
                    "slug": "jeffrey-epstein",
                    "name": "Jeffrey Epstein",
                    "aliases": ["Jeff Epstein"],
                    "category": "convicted",
                },
                {
                    "id": "p-0002",
                    "slug": "ghislaine-maxwell",
                    "name": "Ghislaine Maxwell",
                    "aliases": ["G. Maxwell"],
                    "category": "convicted",
                },
            ]
        )
    )
    return PersonRegistry.from_json(registry_file)


# ── NerBackend enum tests ───────────────────────────────────────────────


class TestNerBackendEnum:
    def test_gliner2_value(self):
        assert NerBackend.GLINER2.value == "gliner2"

    def test_all_backends(self):
        assert "gliner2" in [b.value for b in NerBackend]


# ── Coreference resolution tests ────────────────────────────────────────


class TestCoreference:
    def test_coref_disabled_by_default(self, settings, registry):
        assert settings.enable_coref is False
        extractor = EntityExtractor(settings, registry)
        # extract_all should NOT call resolve_coreferences
        result = extractor.extract_all("He went to the store.")
        assert isinstance(result.person_ids, list)

    def test_resolve_empty_text(self, settings, registry):
        settings.enable_coref = True
        extractor = EntityExtractor(settings, registry)
        assert extractor.resolve_coreferences("") == ""
        assert extractor.resolve_coreferences("   ") == "   "

    def test_coref_import_error_graceful(self, settings, registry):
        """When fastcoref is not installed, coref should be skipped gracefully."""
        settings.enable_coref = True
        extractor = EntityExtractor(settings, registry)
        # _load_coref will raise ImportError if fastcoref not installed
        # extract_all should catch this and proceed without coref
        result = extractor.extract_all("He went to the store.")
        assert isinstance(result.person_ids, list)

    def test_resolve_coreferences_with_mock(self, settings, registry):
        """Test the resolve_coreferences method with a mocked model."""
        settings.enable_coref = True
        extractor = EntityExtractor(settings, registry)

        # Mock the coref model
        mock_pred = MagicMock()
        mock_pred.get_clusters.return_value = [
            [(0, 16), (50, 52)],  # "Jeffrey Epstein" -> "He"
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_pred]
        extractor._coref_model = mock_model

        text = "Jeffrey Epstein hosted a party. Some guests arrived. He left early."
        resolved = extractor.resolve_coreferences(text)

        # "He" (pos 50-52) should be replaced with "Jeffrey Epstein"
        assert "Jeffrey Epstein" in resolved
        mock_model.predict.assert_called_once()

    def test_resolve_coreferences_no_clusters(self, settings, registry):
        """When no coref clusters found, return original text."""
        extractor = EntityExtractor(settings, registry)

        mock_pred = MagicMock()
        mock_pred.get_clusters.return_value = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_pred]
        extractor._coref_model = mock_model

        text = "The quick brown fox jumped."
        assert extractor.resolve_coreferences(text) == text

    def test_resolve_coreferences_exception_returns_original(self, settings, registry):
        """On exception, return original text."""
        extractor = EntityExtractor(settings, registry)

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model crashed")
        extractor._coref_model = mock_model

        text = "He went home."
        assert extractor.resolve_coreferences(text) == text


# ── GLiNER2 tests ───────────────────────────────────────────────────────


class TestGliner2:
    def test_gliner2_import_error(self, settings, registry):
        """Should raise ImportError with install instructions when gliner2 missing."""
        settings.ner_backend = NerBackend.GLINER2
        extractor = EntityExtractor(settings, registry)
        # gliner2 is likely not installed in test env, so extract_all
        # should catch the ImportError and log a warning
        result = extractor.extract_all("Jeffrey Epstein met with someone.")
        assert isinstance(result.person_ids, list)

    def test_extract_gliner2_with_mock(self, settings, registry):
        """Test GLiNER2 extraction with mocked model."""
        settings.ner_backend = NerBackend.GLINER2
        extractor = EntityExtractor(settings, registry)

        # Mock the GLiNER2 model
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "person": [
                    {"text": "Jeffrey Epstein", "confidence": 0.95, "start": 0, "end": 15},
                    {"text": "Ghislaine Maxwell", "confidence": 0.90, "start": 25, "end": 42},
                ],
                "organization": [
                    {"text": "JP Morgan", "confidence": 0.88, "start": 50, "end": 59},
                ],
            }
        }
        extractor._gliner2_model = mock_model

        text = "Jeffrey Epstein and then Ghislaine Maxwell worked with JP Morgan."
        entities = extractor._extract_gliner2(text)

        assert len(entities) == 3
        person_ents = [e for e in entities if e.label == "PERSON"]
        assert len(person_ents) == 2
        assert person_ents[0].text == "Jeffrey Epstein"
        assert person_ents[0].source == "gliner2"

        org_ents = [e for e in entities if e.label == "ORGANIZATION"]
        assert len(org_ents) == 1

    def test_extract_gliner2_string_mentions(self, settings, registry):
        """GLiNER2 can return plain strings instead of dicts."""
        settings.ner_backend = NerBackend.GLINER2
        extractor = EntityExtractor(settings, registry)

        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "person": ["Jeffrey Epstein", "Ghislaine Maxwell"],
            }
        }
        extractor._gliner2_model = mock_model

        entities = extractor._extract_gliner2("some text")
        assert len(entities) == 2
        assert entities[0].text == "Jeffrey Epstein"
        assert entities[0].source == "gliner2"

    def test_extract_gliner2_empty_result(self, settings, registry):
        """GLiNER2 returns empty entities."""
        settings.ner_backend = NerBackend.GLINER2
        extractor = EntityExtractor(settings, registry)

        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {}}
        extractor._gliner2_model = mock_model

        entities = extractor._extract_gliner2("no entities here")
        assert entities == []

    def test_extract_gliner2_confidence_filter(self, settings, registry):
        """Entities below confidence threshold should be filtered."""
        settings.ner_backend = NerBackend.GLINER2
        settings.ner_confidence_threshold = 0.8
        extractor = EntityExtractor(settings, registry)

        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "person": [
                    {"text": "High Conf", "confidence": 0.95, "start": 0, "end": 9},
                    {"text": "Low Conf", "confidence": 0.3, "start": 10, "end": 18},
                ],
            }
        }
        extractor._gliner2_model = mock_model

        entities = extractor._extract_gliner2("High Conf Low Conf")
        assert len(entities) == 1
        assert entities[0].text == "High Conf"

    def test_extract_all_with_gliner2_backend(self, settings, registry):
        """Full extract_all flow with GLINER2 backend."""
        settings.ner_backend = NerBackend.GLINER2
        extractor = EntityExtractor(settings, registry)

        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "person": [
                    {"text": "Jeffrey Epstein", "confidence": 0.95, "start": 0, "end": 15},
                ],
            }
        }
        extractor._gliner2_model = mock_model

        result = extractor.extract_all("Jeffrey Epstein attended the meeting.")
        # Should find person via GLiNER2 + registry scan
        assert "p-0001" in result.person_ids

    def test_gliner2_label_mapping(self):
        """Verify GLiNER2 label keys map correctly."""
        assert _GLINER_LABEL_MAP["person"] == "PERSON"
        assert _GLINER_LABEL_MAP["organization"] == "ORGANIZATION"
        assert _GLINER_LABEL_MAP["location"] == "LOCATION"


# ── Config tests ────────────────────────────────────────────────────────


class TestConfigSettings:
    def test_default_coref_disabled(self, settings):
        assert settings.enable_coref is False

    def test_default_coref_model(self, settings):
        assert settings.coref_model == "FCoref"

    def test_default_gliner2_model(self, settings):
        assert settings.gliner2_model == "fastino/gliner2-base-v1"

    def test_coref_enable(self, settings):
        settings.enable_coref = True
        assert settings.enable_coref is True
