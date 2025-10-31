# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for external semantic model integration framework."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import yaml

from datus.tools.semantic_models.core.format_converter import UniversalFormatConverter, MetricFlowPassthroughConverter
from datus.tools.semantic_models.adapters.clickzetta_adapter import ClickZettaVolumeAdapter
from datus.tools.semantic_models.adapters.clickzetta_converter import ClickZettaFormatConverter
from datus.tools.semantic_models.core.integration_service import UniversalSemanticModelIntegration


class TestUniversalFormatConverter:
    """Test universal format converter."""

    def test_detect_clickzetta_format(self):
        """Test ClickZetta format detection."""
        converter = UniversalFormatConverter()

        clickzetta_content = """
        name: customer_analytics
        description: Customer analytics model
        tables:
          - name: customers
            base_table:
              workspace: production
              schema: public
              table: customers
            dimensions:
              - name: customer_id
                data_type: INTEGER
        """

        format_type = converter.detect_format(clickzetta_content)
        assert format_type == "clickzetta"

    def test_detect_metricflow_format(self):
        """Test MetricFlow format detection."""
        converter = UniversalFormatConverter()

        metricflow_content = """
        model: customer_analytics
        data_sources:
          - name: customers
            sql_table: production.public.customers
            dimensions:
              - name: customer_id
                type: CATEGORICAL
        """

        format_type = converter.detect_format(metricflow_content)
        assert format_type == "metricflow"

    def test_detect_unknown_format(self):
        """Test unknown format detection."""
        converter = UniversalFormatConverter()

        unknown_content = """
        some_random_field: value
        another_field: 123
        """

        format_type = converter.detect_format(unknown_content)
        assert format_type == "unknown"


class TestClickZettaFormatConverter:
    """Test ClickZetta format converter."""

    def test_convert_simple_model(self):
        """Test converting a simple ClickZetta model."""
        converter = ClickZettaFormatConverter()

        clickzetta_model = {
            "name": "test_model",
            "description": "Test model",
            "tables": [
                {
                    "name": "test_table",
                    "description": "Test table",
                    "base_table": {
                        "workspace": "test_workspace",
                        "schema": "test_schema",
                        "table": "test_table"
                    },
                    "dimensions": [
                        {
                            "name": "id",
                            "data_type": "INTEGER",
                            "description": "ID field"
                        }
                    ],
                    "facts": [
                        {
                            "name": "amount",
                            "data_type": "FLOAT",
                            "agg": "SUM",
                            "description": "Amount field"
                        }
                    ]
                }
            ]
        }

        content = yaml.dump(clickzetta_model)
        result = converter.convert_to_metricflow(content)

        assert result["model"] == "test_model"
        assert result["description"] == "Test model"
        assert len(result["data_sources"]) == 1

        data_source = result["data_sources"][0]
        assert data_source["name"] == "test_table"
        assert data_source["sql_table"] == "test_workspace.test_schema.test_table"
        assert len(data_source["dimensions"]) == 1
        assert len(data_source["measures"]) == 1

        dimension = data_source["dimensions"][0]
        assert dimension["name"] == "id"
        assert dimension["type"] == "CATEGORICAL"

        measure = data_source["measures"][0]
        assert measure["name"] == "amount"
        assert measure["agg"] == "SUM"


class TestClickZettaVolumeAdapter:
    """Test ClickZetta volume adapter."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock ClickZetta connector."""
        connector = Mock()
        connector.read_volume_file = Mock()
        connector._run_command = Mock()
        return connector

    @pytest.fixture
    def adapter_config(self):
        """Create adapter configuration."""
        return {
            "provider_config": {
                "volume_type": "user",
                "volume_path": "/semantic_models"
            },
            "file_patterns": ["*.yml", "*.yaml"]
        }

    def test_volume_adapter_init(self, mock_connector, adapter_config):
        """Test volume adapter initialization."""
        adapter = ClickZettaVolumeAdapter(mock_connector, adapter_config)

        assert adapter.volume_type == "user"
        assert adapter.volume_path == "/semantic_models"

    def test_volume_adapter_invalid_type(self, mock_connector):
        """Test volume adapter with invalid type."""
        config = {
            "provider_config": {
                "volume_type": "invalid"
            }
        }

        with pytest.raises(ValueError, match="Unsupported volume type"):
            ClickZettaVolumeAdapter(mock_connector, config)

    def test_read_model(self, mock_connector, adapter_config):
        """Test reading model from volume."""
        mock_connector.read_volume_file.return_value = "test: content"

        adapter = ClickZettaVolumeAdapter(mock_connector, adapter_config)
        content = adapter.read_model("test_model.yml")

        assert content == "test: content"
        mock_connector.read_volume_file.assert_called_once()

    def test_list_models(self, mock_connector, adapter_config):
        """Test listing models from volume."""
        # Mock the result from _run_command
        mock_result = Mock()
        mock_result.to_pandas.return_value = Mock()
        mock_result.to_pandas.return_value.columns = ['relative_path']
        mock_result.to_pandas.return_value.__getitem__ = Mock(return_value=Mock())
        mock_result.to_pandas.return_value.__getitem__.return_value.tolist.return_value = [
            "model1.yml", "model2.yaml", "not_a_model.txt"
        ]

        mock_connector._run_command.return_value = mock_result

        adapter = ClickZettaVolumeAdapter(mock_connector, adapter_config)
        models = adapter.list_models()

        # Should filter out non-yaml files
        assert "model1.yml" in models
        assert "model2.yaml" in models
        assert "not_a_model.txt" not in models


class TestUniversalSemanticModelIntegration:
    """Test universal semantic model integration service."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        connector = Mock()
        connector.read_volume_file = Mock()
        return connector

    @pytest.fixture
    def integration_config(self):
        """Create integration configuration."""
        return {
            "enabled": True,
            "storage_provider": "volume",
            "auto_import": True,
            "sync_on_startup": False,
            "file_patterns": ["*.yml", "*.yaml"],
            "provider_config": {
                "volume_type": "user",
                "volume_path": "/semantic_models"
            }
        }

    @patch('datus.storage.metric.store.SemanticModelStorage')
    @patch('datus.storage.embedding_models.get_embedding_model')
    def test_integration_service_init(self, mock_get_embedding, mock_storage_class, mock_connector, integration_config):
        """Test integration service initialization."""
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        mock_embedding = Mock()
        mock_get_embedding.return_value = mock_embedding

        service = UniversalSemanticModelIntegration(mock_connector, integration_config)

        assert service.is_enabled() is True
        assert service.storage_adapter is not None
        assert service.semantic_storage is not None
        mock_storage_class.assert_called_once()
        mock_get_embedding.assert_called_once_with("metric")

    @patch('datus.storage.metric.store.SemanticModelStorage')
    @patch('datus.storage.embedding_models.get_embedding_model')
    def test_import_model_success(self, mock_get_embedding, mock_storage_class, mock_connector, integration_config):
        """Test successful model import."""
        mock_storage = Mock()
        mock_storage.exists.return_value = False
        mock_storage.save = Mock()
        mock_storage_class.return_value = mock_storage
        mock_embedding = Mock()
        mock_get_embedding.return_value = mock_embedding

        # Mock the storage adapter
        service = UniversalSemanticModelIntegration(mock_connector, integration_config)
        service.storage_adapter.read_model = Mock(return_value="""
        name: test_model
        tables:
          - name: test_table
            base_table:
              workspace: test
              schema: public
              table: test
        """)

        result = service.import_model("test_model.yml")

        assert result["status"] == "imported"
        assert result["model_name"] == "test_model"
        assert result["source_format"] == "clickzetta"
        mock_storage.save.assert_called_once()

    @patch('datus.storage.metric.store.SemanticModelStorage')
    @patch('datus.storage.embedding_models.get_embedding_model')
    def test_import_model_skip_existing(self, mock_get_embedding, mock_storage_class, mock_connector, integration_config):
        """Test skipping existing model."""
        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_storage_class.return_value = mock_storage
        mock_embedding = Mock()
        mock_get_embedding.return_value = mock_embedding

        service = UniversalSemanticModelIntegration(mock_connector, integration_config)
        service.storage_adapter.read_model = Mock(return_value="""
        name: test_model
        tables:
          - name: test_table
            base_table:
              workspace: test
              schema: public
              table: test
        """)

        result = service.import_model("test_model.yml", force_update=False)

        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"

    def test_disabled_integration(self, mock_connector):
        """Test disabled integration."""
        config = {
            "enabled": False,
            "storage_provider": "volume"
        }

        service = UniversalSemanticModelIntegration(mock_connector, config)
        assert service.is_enabled() is False

        result = service.auto_import_models()
        assert result["status"] == "disabled"


class TestMetricFlowPassthroughConverter:
    """Test MetricFlow passthrough converter."""

    def test_passthrough_conversion(self):
        """Test MetricFlow passthrough conversion."""
        converter = MetricFlowPassthroughConverter()

        metricflow_content = """
        model: test
        description: Test model
        data_sources:
          - name: test_source
        """

        result = converter.convert_to_metricflow(metricflow_content)

        assert result["model"] == "test"
        assert result["description"] == "Test model"
        assert "data_sources" in result

    def test_invalid_yaml(self):
        """Test invalid YAML handling."""
        converter = MetricFlowPassthroughConverter()

        invalid_content = """
        invalid: yaml: content:
        """

        with pytest.raises(ValueError, match="Invalid MetricFlow YAML"):
            converter.convert_to_metricflow(invalid_content)