# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""ClickZetta format converter for semantic models."""

import yaml
from typing import Any, Dict, List, Tuple

from ..core.format_converter import FormatConverter
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ClickZettaFormatConverter(FormatConverter):
    """Converter for ClickZetta semantic model format to MetricFlow format."""

    def detect_format(self, content: str) -> str:
        """Detect if content is in ClickZetta format."""
        try:
            data = yaml.safe_load(content)
            if self._is_clickzetta_format(data):
                return "clickzetta"
            return "unknown"
        except Exception:
            return "unknown"

    def _is_clickzetta_format(self, data: Dict[str, Any]) -> bool:
        """Check if data structure matches ClickZetta format."""
        if not isinstance(data, dict) or "tables" not in data:
            return False

        tables = data.get("tables", [])
        if not isinstance(tables, list) or not tables:
            return False

        # Check for ClickZetta-specific structure
        for table in tables:
            if isinstance(table, dict) and "base_table" in table:
                base_table = table["base_table"]
                if isinstance(base_table, dict) and any(
                    key in base_table for key in ["workspace", "database", "schema", "table"]
                ):
                    return True
        return False

    def convert_to_metricflow(self, source_content: str, source_format: str = None) -> Dict[str, Any]:
        """Convert ClickZetta format to MetricFlow format.

        Args:
            source_content: Raw ClickZetta YAML content
            source_format: Source format (ignored, assumed to be ClickZetta)

        Returns:
            MetricFlow format semantic model
        """
        try:
            clickzetta_model = yaml.safe_load(source_content)
            logger.info(f"Converting ClickZetta model: {clickzetta_model.get('name', 'unnamed')}")

            metricflow_model = {
                "model": clickzetta_model.get("name", "unknown"),
                "description": clickzetta_model.get("description", ""),
                "data_sources": [],
                "metrics": []
            }

            # Convert tables to data_sources
            for table in clickzetta_model.get("tables", []):
                data_source = self._convert_table_to_data_source(table)
                metricflow_model["data_sources"].append(data_source)

            # Convert model-level metrics
            for metric in clickzetta_model.get("metrics", []):
                converted_metric = self._convert_model_metric(metric)
                metricflow_model["metrics"].append(converted_metric)

            # Preserve ClickZetta-specific information in extensions
            metricflow_model["clickzetta_extensions"] = {
                "relationships": clickzetta_model.get("relationships", []),
                "verified_queries": clickzetta_model.get("verified_queries", []),
                "comments": clickzetta_model.get("comments", "")
            }

            logger.info(f"Successfully converted ClickZetta model to MetricFlow format")
            return metricflow_model

        except Exception as exc:
            logger.error(f"Failed to convert ClickZetta model: {exc}")
            raise ValueError(f"Invalid ClickZetta semantic model: {exc}") from exc

    def _convert_table_to_data_source(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ClickZetta table to MetricFlow data_source.

        Args:
            table: ClickZetta table definition

        Returns:
            MetricFlow data_source definition
        """
        base_table = table.get("base_table", {})

        # Build SQL table reference
        workspace = base_table.get("workspace", base_table.get("database", ""))
        schema = base_table.get("schema", "")
        table_name = base_table.get("table", "")

        sql_table = f"{workspace}.{schema}.{table_name}" if all([workspace, schema, table_name]) else ""

        data_source = {
            "name": table.get("name", "unknown"),
            "description": table.get("description", ""),
            "sql_table": sql_table,
            "dimensions": [],
            "measures": []
        }

        # Convert dimensions (regular + time)
        dimensions = self._convert_dimensions(
            table.get("dimensions", []),
            table.get("time_dimensions", [])
        )
        data_source["dimensions"] = dimensions

        # Convert facts to measures
        measures = self._convert_facts_to_measures(table.get("facts", []))
        data_source["measures"] = measures

        # Convert table-level metrics
        table_metrics = []
        for metric in table.get("metrics", []):
            converted_metric = self._convert_table_metric(metric, table.get("name"))
            table_metrics.append(converted_metric)

        if table_metrics:
            data_source["table_metrics"] = table_metrics

        # Preserve ClickZetta-specific table information
        data_source["clickzetta_extensions"] = {
            "vcluster": table.get("vcluster"),
            "partitioning": table.get("partitioning", {}),
            "volume_references": table.get("volume_files", []),
            "base_table": base_table,
            "filters": table.get("filters", [])
        }

        return data_source

    def _convert_dimensions(self, regular_dims: List[Dict], time_dims: List[Dict]) -> List[Dict]:
        """Convert ClickZetta dimensions to MetricFlow dimensions.

        Args:
            regular_dims: List of regular dimensions
            time_dims: List of time dimensions

        Returns:
            List of MetricFlow dimensions
        """
        dimensions = []

        # Convert regular dimensions
        for dim in regular_dims:
            converted_dim = {
                "name": dim.get("name", "unknown"),
                "type": "CATEGORICAL",
                "expr": dim.get("expr", dim.get("name", "unknown")),
                "description": dim.get("description", "")
            }

            # ClickZetta-specific dimension features
            if dim.get("synonyms"):
                converted_dim["synonyms"] = dim["synonyms"]

            if dim.get("is_enum"):
                converted_dim["enum_values"] = dim.get("enum_values", [])

            if dim.get("cortex_search_service"):
                converted_dim["clickzetta_cortex"] = {
                    "search_service": dim["cortex_search_service"],
                    "searchable": True
                }

            # Map data type
            if dim.get("data_type"):
                converted_dim["data_type"] = self._map_data_type(dim["data_type"])

            dimensions.append(converted_dim)

        # Convert time dimensions
        for time_dim in time_dims:
            converted_dim = {
                "name": time_dim.get("name", "unknown"),
                "type": "TIME",
                "expr": time_dim.get("expr", time_dim.get("name", "unknown")),
                "description": time_dim.get("description", "")
            }

            if time_dim.get("data_type"):
                converted_dim["data_type"] = self._map_data_type(time_dim["data_type"])

            if time_dim.get("synonyms"):
                converted_dim["synonyms"] = time_dim["synonyms"]

            dimensions.append(converted_dim)

        return dimensions

    def _convert_facts_to_measures(self, facts: List[Dict]) -> List[Dict]:
        """Convert ClickZetta facts to MetricFlow measures.

        Args:
            facts: List of ClickZetta facts

        Returns:
            List of MetricFlow measures
        """
        measures = []

        for fact in facts:
            measure = {
                "name": fact.get("name", "unknown"),
                "agg": self._map_aggregation_type(fact.get("agg", "SUM")),
                "expr": fact.get("expr", fact.get("name", "unknown")),
                "description": fact.get("description", "")
            }

            # Map data type
            if fact.get("data_type"):
                measure["data_type"] = self._map_data_type(fact["data_type"])

            # Handle access modifier
            if fact.get("access_modifier") == "private_access":
                measure["access_modifier"] = "private"
            else:
                measure["access_modifier"] = "public"

            # Preserve synonyms
            if fact.get("synonyms"):
                measure["synonyms"] = fact["synonyms"]

            measures.append(measure)

        return measures

    def _convert_table_metric(self, metric: Dict[str, Any], table_name: str = None) -> Dict[str, Any]:
        """Convert ClickZetta table-level metric to MetricFlow metric.

        Args:
            metric: ClickZetta metric definition
            table_name: Name of the parent table

        Returns:
            MetricFlow metric definition
        """
        converted_metric = {
            "name": metric.get("name", "unknown"),
            "type": "MEASURE_PROXY",  # Default type for table metrics
            "description": metric.get("description", ""),
            "expr": metric.get("expr", "")
        }

        # Handle access modifier
        if metric.get("access_modifier") == "private_access":
            converted_metric["access_modifier"] = "private"
        else:
            converted_metric["access_modifier"] = "public"

        # Preserve synonyms
        if metric.get("synonyms"):
            converted_metric["synonyms"] = metric["synonyms"]

        # Add table context if available
        if table_name:
            converted_metric["table_scope"] = table_name

        return converted_metric

    def _convert_model_metric(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ClickZetta model-level metric to MetricFlow metric.

        Args:
            metric: ClickZetta metric definition

        Returns:
            MetricFlow metric definition
        """
        return {
            "name": metric.get("name", "unknown"),
            "type": "DERIVED",  # Model-level metrics are typically derived
            "description": metric.get("description", ""),
            "expr": metric.get("expr", ""),
            "synonyms": metric.get("synonyms", [])
        }

    def _map_data_type(self, clickzetta_type: str) -> str:
        """Map ClickZetta data type to MetricFlow data type.

        Args:
            clickzetta_type: ClickZetta data type

        Returns:
            MetricFlow data type
        """
        type_mapping = {
            "INTEGER": "INT",
            "BIGINT": "BIGINT",
            "FLOAT": "FLOAT",
            "DOUBLE": "DOUBLE",
            "STRING": "STRING",
            "VARCHAR": "STRING",
            "TIMESTAMP": "TIMESTAMP",
            "DATE": "DATE",
            "BOOLEAN": "BOOLEAN"
        }

        return type_mapping.get(clickzetta_type.upper(), "STRING")

    def _map_aggregation_type(self, clickzetta_agg: str) -> str:
        """Map ClickZetta aggregation type to MetricFlow aggregation type.

        Args:
            clickzetta_agg: ClickZetta aggregation type

        Returns:
            MetricFlow aggregation type
        """
        agg_mapping = {
            "SUM": "SUM",
            "COUNT": "COUNT",
            "COUNT_DISTINCT": "COUNT_DISTINCT",
            "AVG": "AVERAGE",
            "AVERAGE": "AVERAGE",
            "MIN": "MIN",
            "MAX": "MAX",
            "MEDIAN": "MEDIAN"
        }

        return agg_mapping.get(clickzetta_agg.upper(), "SUM")

    def convert_from_metricflow(self, metricflow_model: Dict[str, Any], target_format: str) -> str:
        """Convert MetricFlow format back to ClickZetta format.

        Args:
            metricflow_model: MetricFlow format model
            target_format: Target format (should be 'clickzetta')

        Returns:
            ClickZetta format YAML string
        """
        if target_format != "clickzetta":
            raise NotImplementedError(f"Conversion to {target_format} not supported")

        try:
            clickzetta_model = {
                "name": metricflow_model.get("model", "unknown"),
                "description": metricflow_model.get("description", ""),
                "tables": [],
                "metrics": []
            }

            # Convert data_sources back to tables
            for data_source in metricflow_model.get("data_sources", []):
                table = self._convert_data_source_to_table(data_source)
                clickzetta_model["tables"].append(table)

            # Convert model-level metrics
            for metric in metricflow_model.get("metrics", []):
                clickzetta_metric = self._convert_metricflow_metric(metric)
                clickzetta_model["metrics"].append(clickzetta_metric)

            # Restore ClickZetta-specific information
            cz_extensions = metricflow_model.get("clickzetta_extensions", {})
            if cz_extensions.get("relationships"):
                clickzetta_model["relationships"] = cz_extensions["relationships"]
            if cz_extensions.get("verified_queries"):
                clickzetta_model["verified_queries"] = cz_extensions["verified_queries"]
            if cz_extensions.get("comments"):
                clickzetta_model["comments"] = cz_extensions["comments"]

            return yaml.dump(clickzetta_model, default_flow_style=False, allow_unicode=True)

        except Exception as exc:
            logger.error(f"Failed to convert MetricFlow to ClickZetta: {exc}")
            raise ValueError(f"Conversion failed: {exc}") from exc

    def _convert_data_source_to_table(self, data_source: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MetricFlow data_source back to ClickZetta table.

        Args:
            data_source: MetricFlow data_source definition

        Returns:
            ClickZetta table definition
        """
        # Restore base_table from extensions or parse from sql_table
        cz_extensions = data_source.get("clickzetta_extensions", {})
        base_table = cz_extensions.get("base_table", {})

        if not base_table and data_source.get("sql_table"):
            # Try to parse workspace.schema.table format
            parts = data_source["sql_table"].split(".")
            if len(parts) >= 3:
                base_table = {
                    "workspace": parts[0],
                    "schema": parts[1],
                    "table": parts[2]
                }

        table = {
            "name": data_source.get("name", "unknown"),
            "description": data_source.get("description", ""),
            "base_table": base_table,
            "dimensions": [],
            "time_dimensions": [],
            "facts": [],
            "metrics": []
        }

        # Restore ClickZetta-specific fields
        if cz_extensions.get("vcluster"):
            table["vcluster"] = cz_extensions["vcluster"]
        if cz_extensions.get("partitioning"):
            table["partitioning"] = cz_extensions["partitioning"]
        if cz_extensions.get("volume_references"):
            table["volume_files"] = cz_extensions["volume_references"]
        if cz_extensions.get("filters"):
            table["filters"] = cz_extensions["filters"]

        # Convert dimensions back
        for dim in data_source.get("dimensions", []):
            if dim.get("type") == "TIME":
                table["time_dimensions"].append(self._convert_metricflow_dimension(dim))
            else:
                table["dimensions"].append(self._convert_metricflow_dimension(dim))

        # Convert measures back to facts
        for measure in data_source.get("measures", []):
            table["facts"].append(self._convert_metricflow_measure(measure))

        # Convert table metrics
        for metric in data_source.get("table_metrics", []):
            table["metrics"].append(self._convert_metricflow_metric(metric))

        return table

    def _convert_metricflow_dimension(self, dimension: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MetricFlow dimension back to ClickZetta dimension."""
        dim = {
            "name": dimension.get("name", "unknown"),
            "expr": dimension.get("expr", ""),
            "description": dimension.get("description", "")
        }

        if dimension.get("data_type"):
            dim["data_type"] = dimension["data_type"]

        if dimension.get("synonyms"):
            dim["synonyms"] = dimension["synonyms"]

        if dimension.get("enum_values"):
            dim["is_enum"] = True
            dim["enum_values"] = dimension["enum_values"]

        # Restore ClickZetta Cortex search integration
        cortex = dimension.get("clickzetta_cortex", {})
        if cortex.get("search_service"):
            dim["cortex_search_service"] = cortex["search_service"]

        return dim

    def _convert_metricflow_measure(self, measure: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MetricFlow measure back to ClickZetta fact."""
        fact = {
            "name": measure.get("name", "unknown"),
            "expr": measure.get("expr", ""),
            "description": measure.get("description", "")
        }

        if measure.get("agg"):
            fact["agg"] = measure["agg"]

        if measure.get("data_type"):
            fact["data_type"] = measure["data_type"]

        if measure.get("synonyms"):
            fact["synonyms"] = measure["synonyms"]

        # Convert access modifier
        if measure.get("access_modifier") == "private":
            fact["access_modifier"] = "private_access"
        else:
            fact["access_modifier"] = "public_access"

        return fact

    def _convert_metricflow_metric(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MetricFlow metric back to ClickZetta metric."""
        return {
            "name": metric.get("name", "unknown"),
            "expr": metric.get("expr", ""),
            "description": metric.get("description", ""),
            "synonyms": metric.get("synonyms", [])
        }