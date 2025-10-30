# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from datus.configuration.agent_config import AgentConfig, SemanticModelConfig
from datus.schemas.node_models import (
    SemanticModelBaseTable,
    SemanticModelDimension,
    SemanticModelFact,
    SemanticModelLogicalTable,
    SemanticModelMetricSpec,
    SemanticModelPayload,
    SemanticModelRelationship,
    SemanticModelRelationshipColumn,
    SemanticModelTimeDimension,
    SemanticModelVerifiedQuery,
    SemanticModelFilter,
    SqlTask,
)
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SemanticModelRepositoryError(RuntimeError):
    """Raised when the semantic model cannot be loaded according to user preference."""


@dataclass
class _SemanticModelLocation:
    kind: str
    path: Optional[Path] = None
    volume: str = ""
    directory: str = ""
    filename: str = ""

    @property
    def display_path(self) -> str:
        if self.kind == "local" and self.path:
            return str(self.path)
        components = [self.volume.rstrip("/")]
        relative_path = "/".join(part for part in (self.directory.strip("/"), self.filename) if part)
        if relative_path:
            components.append(relative_path)
        return "/".join(filter(None, components))


class SemanticModelRepository:
    """Centralised loader for semantic model YAML assets."""

    def __init__(self, agent_config: AgentConfig):
        self._config = agent_config

    def load(
        self,
        task: SqlTask,
        connector: Optional[BaseSqlConnector] = None,
    ) -> Optional[SemanticModelPayload]:
        defaults = self._config.semantic_model_defaults()
        strategy = (task.context_strategy or defaults.default_strategy).lower()
        location = self._resolve_location(task, defaults)
        if not location:
            if strategy == "semantic_model":
                raise SemanticModelRepositoryError(
                    "Semantic model strategy selected but no semantic model file was specified."
                )
            logger.debug("No semantic model details supplied; skipping semantic model load.")
            return None

        try:
            raw_yaml, source = self._load_raw(location, connector)
        except FileNotFoundError as exc:
            if strategy == "semantic_model":
                raise SemanticModelRepositoryError(str(exc)) from exc
            logger.warning("Semantic model not found (%s). Falling back to schema linking.", exc)
            return None
        except Exception as exc:
            if strategy == "semantic_model":
                raise SemanticModelRepositoryError(f"Failed to load semantic model: {exc}") from exc
            logger.warning("Error while loading semantic model: %s", exc)
            return None

        payload = self._build_payload(raw_yaml, source)
        logger.info("Loaded semantic model '%s' from %s", payload.name or "Unnamed model", source)
        return payload

    def list_models(
        self,
        *,
        volume: Optional[str] = None,
        directory: Optional[str] = None,
        local_dir: Optional[str] = None,
        connector: Optional[BaseSqlConnector] = None,
    ) -> List[str]:
        """Enumerate semantic model filenames from the configured storage backends."""
        defaults = self._config.semantic_model_defaults()
        results: List[str] = []

        resolved_volume = (volume or "").strip() or defaults.default_volume.strip()
        resolved_directory = (directory or "").strip() or defaults.default_directory.strip()

        if local_dir:
            base_path = Path(local_dir).expanduser()
            if base_path.is_file():
                if base_path.suffix.lower() in (".yaml", ".yml"):
                    results.append(base_path.name)
            elif base_path.is_dir():
                for path in sorted(base_path.glob("*.yaml")):
                    results.append(path.name)
                for path in sorted(base_path.glob("*.yml")):
                    if path.name not in results:
                        results.append(path.name)
            else:
                logger.warning("Local semantic model path '%s' does not exist.", base_path)

        if resolved_volume:
            if isinstance(connector, ClickzettaConnector):
                try:
                    files = connector.list_volume_files(resolved_volume, resolved_directory)
                    for file_name in files:
                        if file_name not in results:
                            results.append(file_name)
                except Exception as exc:
                    logger.warning("Failed to list semantic models from %s/%s: %s", resolved_volume, resolved_directory, exc)
            else:
                logger.debug(
                    "Volume '%s' provided but active connector %s does not support volume listing.",
                    resolved_volume,
                    type(connector).__name__ if connector else "None",
                )

        return sorted(results)

    def _resolve_location(self, task: SqlTask, defaults: SemanticModelConfig) -> Optional[_SemanticModelLocation]:
        local_path = task.semantic_model_local_path.strip()
        if local_path:
            if not defaults.allow_local_path:
                logger.warning(
                    "Semantic model local path specified (%s) but allow_local_path is disabled in configuration.",
                    local_path,
                )
            else:
                path = Path(local_path).expanduser()
                return _SemanticModelLocation(kind="local", path=path)

        filename = task.semantic_model_filename.strip()
        if not filename:
            return None

        volume = task.semantic_model_volume.strip() or defaults.default_volume.strip()
        directory = task.semantic_model_directory.strip() or defaults.default_directory.strip()

        if not volume:
            logger.warning(
                "Semantic model filename '%s' provided without a volume or local path; unable to resolve location.",
                filename,
            )
            return None

        return _SemanticModelLocation(kind="volume", volume=volume, directory=directory, filename=filename)

    def _load_raw(
        self, location: _SemanticModelLocation, connector: Optional[BaseSqlConnector]
    ) -> tuple[str, str]:
        if location.kind == "local":
            assert location.path is not None  # For type-checkers
            if not location.path.exists():
                raise FileNotFoundError(f"Semantic model file not found at {location.path}")
            raw_yaml = location.path.read_text(encoding="utf-8")
            return raw_yaml, str(location.path.resolve())

        if location.kind == "volume":
            if not isinstance(connector, ClickzettaConnector):
                raise SemanticModelRepositoryError(
                    "Semantic model volume loading requires an active ClickZetta connector."
                )
            relative_parts = [part for part in (location.directory, location.filename) if part]
            relative_path = "/".join(relative_parts)
            raw_yaml = connector.read_volume_file(location.volume, relative_path)
            if not raw_yaml:
                raise FileNotFoundError(
                    f"Semantic model '{relative_path}' not found in volume '{location.volume}'."
                )
            return raw_yaml, f"{location.volume.rstrip('/')}/{relative_path}"

        raise SemanticModelRepositoryError(f"Unsupported location type '{location.kind}' for semantic model.")

    def _build_payload(self, raw_yaml: str, source: str) -> SemanticModelPayload:
        try:
            documents = list(yaml.safe_load_all(raw_yaml))
        except Exception as exc:
            logger.warning("Failed to parse semantic model YAML (%s). Using raw YAML for prompts.", exc)
            documents = []

        payload = SemanticModelPayload(
            name="",
            source=source,
            raw_yaml=raw_yaml,
        )

        table_names: set[str] = set()
        measure_names: set[str] = set()
        dimension_names: set[str] = set()
        logical_tables: list[SemanticModelLogicalTable] = []
        relationships: list[SemanticModelRelationship] = []
        model_metrics: list[SemanticModelMetricSpec] = []
        verified_queries: list[SemanticModelVerifiedQuery] = []

        def _collect_synonyms(values) -> List[str]:
            if not values:
                return []
            if isinstance(values, (list, tuple, set)):
                return [str(item) for item in values if item]
            return [str(values)]

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            if not payload.name and doc.get("name"):
                payload.name = str(doc["name"])
            if doc.get("description") and not payload.description:
                payload.description = str(doc["description"])
            if doc.get("comments") and not payload.comments:
                payload.comments = str(doc["comments"])

            if "tables" in doc and isinstance(doc["tables"], list):
                for table in doc["tables"]:
                    if not isinstance(table, dict):
                        continue
                    table_name = str(table.get("name", "") or "")
                    table_desc = str(table.get("description", "") or "")
                    base_table_data = table.get("base_table", {})
                    base_table = None
                    if isinstance(base_table_data, dict) and any(base_table_data.values()):
                        base_table = SemanticModelBaseTable(
                            catalog=str(base_table_data.get("catalog", "") or ""),
                            database=str(base_table_data.get("database", "") or ""),
                            schema=str(base_table_data.get("schema", "") or ""),
                            table=str(base_table_data.get("table", "") or ""),
                        )
                        if base_table.table:
                            table_names.add(base_table.to_fqn())

                    dimensions = []
                    for dimension in table.get("dimensions", []) or []:
                        if not isinstance(dimension, dict):
                            continue
                        dim = SemanticModelDimension(
                            name=str(dimension.get("name", "") or ""),
                            description=str(dimension.get("description", "") or ""),
                            synonyms=_collect_synonyms(dimension.get("synonyms")),
                            expr=str(dimension.get("expr", "") or ""),
                            data_type=str(dimension.get("data_type", "") or ""),
                            unique=dimension.get("unique"),
                            is_enum=dimension.get("is_enum"),
                        )
                        dimensions.append(dim)
                        if dim.name:
                            dimension_names.add(f"{table_name}.{dim.name}" if table_name else dim.name)

                    time_dimensions = []
                    for dimension in table.get("time_dimensions", []) or []:
                        if not isinstance(dimension, dict):
                            continue
                        time_dim = SemanticModelTimeDimension(
                            name=str(dimension.get("name", "") or ""),
                            description=str(dimension.get("description", "") or ""),
                            synonyms=_collect_synonyms(dimension.get("synonyms")),
                            expr=str(dimension.get("expr", "") or ""),
                            data_type=str(dimension.get("data_type", "") or ""),
                            unique=dimension.get("unique"),
                        )
                        time_dimensions.append(time_dim)
                        if time_dim.name:
                            dimension_names.add(f"{table_name}.{time_dim.name}" if table_name else time_dim.name)

                    facts = []
                    for fact_data in table.get("facts", []) or []:
                        if not isinstance(fact_data, dict):
                            continue
                        fact = SemanticModelFact(
                            name=str(fact_data.get("name", "") or ""),
                            description=str(fact_data.get("description", "") or ""),
                            synonyms=_collect_synonyms(fact_data.get("synonyms")),
                            expr=str(fact_data.get("expr", "") or ""),
                            data_type=str(fact_data.get("data_type", "") or ""),
                            access_modifier=str(fact_data.get("access_modifier", "") or "public_access"),
                        )
                        facts.append(fact)
                        if fact.name:
                            measure_names.add(f"{table_name}.{fact.name}" if table_name else fact.name)

                    table_metrics = []
                    for metric in table.get("metrics", []) or []:
                        if not isinstance(metric, dict):
                            continue
                        metric_obj = SemanticModelMetricSpec(
                            name=str(metric.get("name", "") or ""),
                            description=str(metric.get("description", "") or ""),
                            synonyms=_collect_synonyms(metric.get("synonyms")),
                            expr=str(metric.get("expr", "") or ""),
                            access_modifier=str(metric.get("access_modifier", "") or "public_access"),
                            metric_type=str(metric.get("type", "") or ""),
                        )
                        table_metrics.append(metric_obj)
                        if metric_obj.name:
                            measure_names.add(f"{table_name}.{metric_obj.name}" if table_name else metric_obj.name)

                    filters = []
                    for filter_item in table.get("filters", []) or []:
                        if not isinstance(filter_item, dict):
                            continue
                        filter_obj = SemanticModelFilter(
                            name=str(filter_item.get("name", "") or ""),
                            description=str(filter_item.get("description", "") or ""),
                            synonyms=_collect_synonyms(filter_item.get("synonyms")),
                            expr=str(filter_item.get("expr", "") or ""),
                        )
                        filters.append(filter_obj)

                    logical_tables.append(
                        SemanticModelLogicalTable(
                            name=table_name,
                            description=table_desc,
                            base_table=base_table,
                            dimensions=dimensions,
                            time_dimensions=time_dimensions,
                            facts=facts,
                            metrics=table_metrics,
                            filters=filters,
                        )
                    )

                    if table_name:
                        table_names.add(table_name)

            if "relationships" in doc and isinstance(doc["relationships"], list):
                for rel in doc["relationships"]:
                    if not isinstance(rel, dict):
                        continue
                    cols = []
                    for column in rel.get("relationship_columns", []) or []:
                        if isinstance(column, dict):
                            cols.append(
                                SemanticModelRelationshipColumn(
                                    left_column=str(column.get("left_column", "") or ""),
                                    right_column=str(column.get("right_column", "") or ""),
                                )
                            )
                    relationships.append(
                        SemanticModelRelationship(
                            name=str(rel.get("name", "") or ""),
                            left_table=str(rel.get("left_table", "") or ""),
                            right_table=str(rel.get("right_table", "") or ""),
                            relationship_columns=cols,
                            join_type=str(rel.get("join_type", "") or ""),
                            relationship_type=str(rel.get("relationship_type", "") or ""),
                        )
                    )

            if "metrics" in doc and isinstance(doc["metrics"], list):
                for metric in doc["metrics"]:
                    if not isinstance(metric, dict):
                        continue
                    metric_obj = SemanticModelMetricSpec(
                        name=str(metric.get("name", "") or ""),
                        description=str(metric.get("description", "") or ""),
                        synonyms=_collect_synonyms(metric.get("synonyms")),
                        expr=str(metric.get("expr", "") or ""),
                        access_modifier=str(metric.get("access_modifier", "") or "public_access"),
                        metric_type=str(metric.get("type", "") or ""),
                    )
                    if metric_obj.name and all(existing.name != metric_obj.name for existing in model_metrics):
                        model_metrics.append(metric_obj)
                        measure_names.add(metric_obj.name)

            if "verified_queries" in doc and isinstance(doc["verified_queries"], list):
                for query in doc["verified_queries"]:
                    if not isinstance(query, dict):
                        continue
                    verified_queries.append(
                        SemanticModelVerifiedQuery(
                            name=str(query.get("name", "") or ""),
                            question=str(query.get("question", "") or ""),
                            sql=str(query.get("sql", "") or ""),
                            verified_at=str(query.get("verified_at", "") or "") if query.get("verified_at") else None,
                            verified_by=str(query.get("verified_by", "") or "") if query.get("verified_by") else None,
                            use_as_onboarding_question=query.get("use_as_onboarding_question"),
                        )
                    )

            semantic_models = doc.get("semantic_models") or []
            if isinstance(semantic_models, list):
                for entry in semantic_models:
                    if not isinstance(entry, dict):
                        continue
                    entry_name = str(entry.get("name", "") or "")
                    entry_description = str(entry.get("description", "") or "")
                    logical_table = SemanticModelLogicalTable(
                        name=entry_name,
                        description=entry_description,
                        dimensions=[
                            SemanticModelDimension(
                                name=str(d.get("name", "") or ""),
                                description=str(d.get("description", "") or ""),
                            )
                            for d in entry.get("dimensions", []) or []
                            if isinstance(d, dict)
                        ],
                        metrics=[
                            SemanticModelMetricSpec(
                                name=str(m.get("name", "") or ""),
                                description=str(m.get("description", "") or ""),
                                expr=str(m.get("expr", "") or ""),
                            )
                            for m in entry.get("measures", []) or []
                            if isinstance(m, dict)
                        ],
                    )
                    if entry.get("model"):
                        table_names.add(str(entry["model"]))
                    if entry_name:
                        table_names.add(entry_name)
                    for dim in logical_table.dimensions:
                        if dim.name:
                            dimension_names.add(f"{entry_name}.{dim.name}" if entry_name else dim.name)
                    for metric in logical_table.metrics:
                        if metric.name:
                            measure_names.add(f"{entry_name}.{metric.name}" if entry_name else metric.name)
                    logical_tables.append(logical_table)

            metrics_block = doc.get("metrics")
            if isinstance(metrics_block, list) and semantic_models:
                for metric in metrics_block:
                    if not isinstance(metric, dict):
                        continue
                    metric_obj = SemanticModelMetricSpec(
                        name=str(metric.get("name", "") or ""),
                        description=str(metric.get("description", "") or ""),
                        expr=str(metric.get("expr", "") or ""),
                        metric_type=str(metric.get("type", "") or ""),
                    )
                    if metric_obj.name and all(existing.name != metric_obj.name for existing in model_metrics):
                        model_metrics.append(metric_obj)
                        measure_names.add(metric_obj.name)

        payload.logical_tables = logical_tables
        payload.relationships = relationships

        unique_model_metrics: list[SemanticModelMetricSpec] = []
        seen_metric_names: set[str] = set()
        for metric in model_metrics:
            key = metric.name or metric.expr
            if key in seen_metric_names:
                continue
            seen_metric_names.add(key)
            unique_model_metrics.append(metric)
        payload.model_metrics = unique_model_metrics

        payload.verified_queries = verified_queries
        payload.tables = sorted(table_names)
        payload.measures = sorted(measure_names)
        payload.dimensions = sorted(dimension_names)

        prompt_summary = payload.build_prompt()
        payload.prompt_text = prompt_summary or raw_yaml

        if not payload.name:
            payload.name = payload.tables[0] if payload.tables else ""

        return payload
