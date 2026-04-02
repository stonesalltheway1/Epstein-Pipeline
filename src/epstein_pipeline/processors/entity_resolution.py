"""Probabilistic entity resolution using Splink 4 with DuckDB backend.

Resolves the "John Smith problem" by comparing person records across
multiple fields (name, aliases, category, document co-occurrence) using
Fellegi-Sunter probabilistic record linkage.

Augments (does not replace) the existing PersonRegistry with high-confidence
match clusters.

Usage:
    resolver = EntityResolver(settings)
    result = resolver.resolve(person_records)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from epstein_pipeline.config import Settings
from epstein_pipeline.models.document import Person

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class PersonRecord:
    """A person record prepared for Splink comparison."""

    unique_id: str
    source: str
    name: str
    name_lower: str
    first_name: str | None = None
    last_name: str | None = None
    aliases: str | None = None
    category: str | None = None
    document_ids: str | None = None


@dataclass
class EntityCluster:
    """A cluster of person records resolved to the same entity."""

    cluster_id: str | int
    canonical_person_id: str | None = None
    canonical_name: str = ""
    records: list[PersonRecord] = field(default_factory=list)
    avg_match_probability: float = 0.0


@dataclass
class ResolutionResult:
    """Complete output of entity resolution."""

    clusters: list[EntityCluster] = field(default_factory=list)
    total_input_records: int = 0
    total_clusters: int = 0
    merge_map: dict[str, str] = field(default_factory=dict)


# ── Resolver class ───────────────────────────────────────────────────────


class EntityResolver:
    """Probabilistic entity resolution using Splink 4 + DuckDB.

    Parameters
    ----------
    settings : Settings
        Pipeline settings with Splink configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.threshold = settings.splink_match_probability_threshold

    @staticmethod
    def persons_to_records(
        persons: list[Person],
        source: str = "registry",
    ) -> list[PersonRecord]:
        """Convert Person models to PersonRecord for Splink input."""
        records: list[PersonRecord] = []
        for p in persons:
            name_parts = p.name.split(maxsplit=1)
            first_name = name_parts[0] if name_parts else None
            last_name = name_parts[1] if len(name_parts) > 1 else None

            records.append(
                PersonRecord(
                    unique_id=p.id,
                    source=source,
                    name=p.name,
                    name_lower=p.name.lower().strip(),
                    first_name=first_name,
                    last_name=last_name,
                    aliases="|".join(p.aliases) if p.aliases else None,
                    category=p.category,
                    document_ids=None,
                )
            )

            # Also create records for each alias to enable cross-alias matching
            for alias in p.aliases:
                alias_parts = alias.split(maxsplit=1)
                records.append(
                    PersonRecord(
                        unique_id=f"{p.id}::alias::{alias}",
                        source=f"{source}_alias",
                        name=alias,
                        name_lower=alias.lower().strip(),
                        first_name=alias_parts[0] if alias_parts else None,
                        last_name=alias_parts[1] if len(alias_parts) > 1 else None,
                        aliases=None,
                        category=p.category,
                        document_ids=None,
                    )
                )

        return records

    def resolve(self, records: list[PersonRecord]) -> ResolutionResult:
        """Run Splink entity resolution on a list of PersonRecords.

        Parameters
        ----------
        records : list[PersonRecord]
            All person records from all sources.

        Returns
        -------
        ResolutionResult
            Clusters, merge map, and statistics.
        """
        if len(records) < 2:
            logger.warning("Need at least 2 records for entity resolution")
            return ResolutionResult(
                total_input_records=len(records),
                total_clusters=len(records),
            )

        try:
            import splink  # noqa: F401
        except ImportError:
            raise ImportError(
                "splink is required for entity resolution. "
                "Install with: pip install 'epstein-pipeline[splink]'"
            )

        import pandas as pd
        import splink.comparison_library as cl
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on

        logger.info("Running Splink entity resolution on %d records...", len(records))

        # Convert records to a pandas DataFrame for Splink 4
        input_data = pd.DataFrame(
            [
                {
                    "unique_id": r.unique_id,
                    "source": r.source,
                    "name": r.name,
                    "name_lower": r.name_lower,
                    "first_name": r.first_name or "",
                    "last_name": r.last_name or "",
                    "aliases": r.aliases or "",
                    "category": r.category or "",
                }
                for r in records
            ]
        )

        # Configure Splink model
        db_api = DuckDBAPI()

        splink_settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=[
                cl.JaroWinklerAtThresholds("name", [0.95, 0.88, 0.7]),
                cl.JaroWinklerAtThresholds("first_name", [0.95, 0.88]),
                cl.JaroWinklerAtThresholds("last_name", [0.95, 0.88]),
                cl.JaroWinklerAtThresholds("aliases", [0.88]),
                cl.ExactMatch("category"),
            ],
            blocking_rules_to_generate_predictions=[
                block_on("name_lower"),
                block_on("last_name"),
            ],
            retain_intermediate_calculation_columns=False,
            max_iterations=10,
            em_convergence=0.001,
        )

        linker = Linker(input_data, splink_settings, db_api)

        # Step 1: Estimate prior probability two random records match
        try:
            linker.training.estimate_probability_two_random_records_match(
                [block_on("name_lower")], recall=0.7,
            )
        except Exception as exc:
            logger.warning("Prior estimation failed: %s", exc)

        # Step 2: Estimate u probabilities from random sampling
        linker.training.estimate_u_using_random_sampling(
            max_pairs=min(self.settings.splink_max_pairs, len(records) * 100)
        )

        # Step 3: Train m probabilities using blocking rules
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on("name_lower"),
            )
        except Exception as exc:
            logger.warning(
                "EM training on name_lower failed (may be too few pairs): %s", exc
            )

        try:
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on("last_name"),
            )
        except Exception as exc:
            logger.warning("EM training on last_name failed: %s", exc)

        # Step 4: Predict matches
        predictions = linker.inference.predict(
            threshold_match_probability=self.threshold
        )

        # Step 5: Cluster pairwise predictions
        clusters_df = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, self.threshold
        )

        # Convert clusters to our output format
        clusters_data = clusters_df.as_pandas_dataframe()
        record_map = {r.unique_id: r for r in records}

        cluster_groups: dict[str | int, list[PersonRecord]] = {}
        for _, row in clusters_data.iterrows():
            cid = row["cluster_id"]
            uid = row["unique_id"]
            if uid in record_map:
                cluster_groups.setdefault(cid, []).append(record_map[uid])

        # Build EntityClusters
        entity_clusters: list[EntityCluster] = []
        merge_map: dict[str, str] = {}

        for cid, cluster_records in cluster_groups.items():
            # Pick canonical: prefer "registry" source, then longest name
            registry_records = [r for r in cluster_records if r.source == "registry"]
            if registry_records:
                canonical = registry_records[0]
            else:
                canonical = max(cluster_records, key=lambda r: len(r.name))

            ec = EntityCluster(
                cluster_id=cid,
                canonical_person_id=(
                    canonical.unique_id
                    if "::" not in canonical.unique_id
                    else None
                ),
                canonical_name=canonical.name,
                records=cluster_records,
            )
            entity_clusters.append(ec)

            # Build merge map
            for r in cluster_records:
                if r.unique_id != canonical.unique_id:
                    merge_map[r.unique_id] = canonical.unique_id

        result = ResolutionResult(
            clusters=entity_clusters,
            total_input_records=len(records),
            total_clusters=len(entity_clusters),
            merge_map=merge_map,
        )

        logger.info(
            "Entity resolution: %d records -> %d clusters (%d merges)",
            result.total_input_records,
            result.total_clusters,
            len(result.merge_map),
        )

        return result

    def resolve_persons(self, persons: list[Person]) -> ResolutionResult:
        """Convenience: resolve a list of Person models directly."""
        records = self.persons_to_records(persons)
        return self.resolve(records)
