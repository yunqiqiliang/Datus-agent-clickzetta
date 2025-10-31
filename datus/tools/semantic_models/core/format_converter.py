# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Format converters for semantic models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import yaml
import re

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FormatConverter(ABC):
    """Abstract base class for semantic model format converters."""

    @abstractmethod
    def detect_format(self, content: str) -> str:
        """Detect the format of semantic model content.

        Args:
            content: Raw content of semantic model

        Returns:
            Format identifier (e.g., 'clickzetta', 'dbt', 'metricflow')
        """
        pass

    @abstractmethod
    def convert_to_metricflow(self, source_content: str, source_format: str = None) -> Dict[str, Any]:
        """Convert source format to MetricFlow format.

        Args:
            source_content: Raw content in source format
            source_format: Source format identifier (auto-detected if None)

        Returns:
            Semantic model in MetricFlow format
        """
        pass

    def convert_from_metricflow(self, metricflow_model: Dict[str, Any], target_format: str) -> str:
        """Convert MetricFlow format to target format.

        This is an optional operation that may not be supported by all converters.

        Args:
            metricflow_model: Semantic model in MetricFlow format
            target_format: Target format identifier

        Returns:
            Semantic model content in target format

        Raises:
            NotImplementedError: If conversion is not supported
        """
        raise NotImplementedError(f"Conversion to {target_format} not supported")


class UniversalFormatConverter(FormatConverter):
    """Universal format converter supporting multiple semantic model formats."""

    def __init__(self):
        """Initialize converter with format-specific converters."""
        self.converters = {}
        self._register_converters()

    def _register_converters(self):
        """Register format-specific converters."""
        try:
            from ..adapters.clickzetta_converter import ClickZettaFormatConverter
            self.converters["clickzetta"] = ClickZettaFormatConverter()
        except ImportError:
            logger.debug("ClickZetta converter not available")

        # Register MetricFlow passthrough converter
        self.converters["metricflow"] = MetricFlowPassthroughConverter()

    def detect_format(self, content: str) -> str:
        """Auto-detect semantic model format.

        Args:
            content: Raw content of semantic model

        Returns:
            Format identifier
        """
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return "unknown"

            # ClickZetta format detection
            if self._is_clickzetta_format(data):
                return "clickzetta"

            # dbt format detection
            if self._is_dbt_format(data):
                return "dbt"

            # MetricFlow format detection
            if self._is_metricflow_format(data):
                return "metricflow"

            # Looker format detection
            if self._is_looker_format(data):
                return "looker"

            return "unknown"

        except Exception as e:
            logger.warning(f"Failed to detect format: {e}")
            return "unknown"

    def _is_clickzetta_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in ClickZetta format."""
        # ClickZetta format has 'tables' with 'base_table' structure
        if "tables" not in data:
            return False

        tables = data.get("tables", [])
        if not isinstance(tables, list) or not tables:
            return False

        # Check if any table has ClickZetta-specific structure
        for table in tables:
            if isinstance(table, dict) and "base_table" in table:
                base_table = table["base_table"]
                if isinstance(base_table, dict) and any(
                    key in base_table for key in ["workspace", "database", "schema", "table"]
                ):
                    return True

        return False

    def _is_dbt_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in dbt format."""
        return "version" in data and ("models" in data or "metrics" in data)

    def _is_metricflow_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in MetricFlow format."""
        return "data_sources" in data or ("semantic_models" in data and "metrics" in data)

    def _is_looker_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in Looker format."""
        content_str = str(data)
        return "view" in data and ("dimension" in content_str or "measure" in content_str)

    def convert_to_metricflow(self, source_content: str, source_format: str = None) -> Dict[str, Any]:
        """Convert any supported format to MetricFlow.

        Args:
            source_content: Raw content in source format
            source_format: Source format identifier (auto-detected if None)

        Returns:
            Semantic model in MetricFlow format

        Raises:
            ValueError: If format is not supported
        """
        if source_format is None:
            source_format = self.detect_format(source_content)

        if source_format not in self.converters:
            raise ValueError(f"Unsupported source format: {source_format}")

        converter = self.converters[source_format]
        return converter.convert_to_metricflow(source_content, source_format)

    def convert_from_metricflow(self, metricflow_model: Dict[str, Any], target_format: str) -> str:
        """Convert MetricFlow format to target format.

        Args:
            metricflow_model: Semantic model in MetricFlow format
            target_format: Target format identifier

        Returns:
            Semantic model content in target format
        """
        if target_format not in self.converters:
            raise ValueError(f"Unsupported target format: {target_format}")

        converter = self.converters[target_format]
        return converter.convert_from_metricflow(metricflow_model, target_format)


class MetricFlowPassthroughConverter(FormatConverter):
    """Passthrough converter for MetricFlow format (no conversion needed)."""

    def detect_format(self, content: str) -> str:
        """Always returns metricflow for this converter."""
        return "metricflow"

    def convert_to_metricflow(self, source_content: str, source_format: str = None) -> Dict[str, Any]:
        """Return the content as-is since it's already MetricFlow format."""
        try:
            return yaml.safe_load(source_content)
        except Exception as e:
            raise ValueError(f"Invalid MetricFlow YAML: {e}")

    def convert_from_metricflow(self, metricflow_model: Dict[str, Any], target_format: str) -> str:
        """Convert MetricFlow model back to YAML."""
        if target_format != "metricflow":
            raise NotImplementedError(f"Conversion to {target_format} not supported")

        return yaml.dump(metricflow_model, default_flow_style=False, allow_unicode=True)