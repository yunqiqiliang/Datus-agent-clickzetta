# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""ClickZetta volume adapter for semantic models."""

import os
import tempfile
from typing import Any, Dict, List
from datetime import datetime

from ..core.storage_adapter import SemanticModelStorageAdapter
from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ClickZettaVolumeAdapter(SemanticModelStorageAdapter):
    """ClickZetta Volume storage adapter for semantic models.

    Supports both User Volume and Named Volume access patterns.
    """

    def __init__(self, connector: ClickzettaConnector, config: Dict[str, Any]):
        """Initialize ClickZetta volume adapter.

        Args:
            connector: ClickZetta connector instance
            config: Configuration containing volume settings
        """
        super().__init__(connector, config)
        self.provider_config = config.get("provider_config", {})
        self.volume_type = self.provider_config.get("volume_type", "user")
        self.volume_name = self.provider_config.get("volume_name", "semantic_models")
        self.volume_path = self.provider_config.get("volume_path", "/semantic_models")

        # Validate volume type
        if self.volume_type not in ["user", "named"]:
            raise ValueError(f"Unsupported volume type: {self.volume_type}. Must be 'user' or 'named'")

        logger.info(f"Initialized ClickZetta adapter: type={self.volume_type}, "
                   f"volume={self.volume_name}, path={self.volume_path}")

    def _get_volume_prefix(self) -> str:
        """Get the SQL prefix for volume operations.

        Returns:
            Volume prefix string for SQL commands
        """
        if self.volume_type == "user":
            return "USER VOLUME"
        elif self.volume_type == "named":
            return f"VOLUME {self.volume_name}"
        else:
            raise ValueError(f"Unsupported volume type: {self.volume_type}")

    def list_models(self) -> List[str]:
        """List all semantic model files in the volume.

        Returns:
            List of semantic model filenames

        Raises:
            DatusException: If listing fails
        """
        try:
            volume_prefix = self._get_volume_prefix()

            if self.volume_path.strip("/"):
                # List files in subdirectory
                sql = f"LIST {volume_prefix} SUBDIRECTORY '{self.volume_path.strip('/')}'"
            else:
                # List files in root, filter by extension
                sql = f"LIST {volume_prefix}"

            logger.info(f"[DEBUG] Listing models with SQL: {sql}")
            result = self.connector._run_command(sql)

            logger.info(f"[DEBUG] LIST result type: {type(result)}")
            if hasattr(result, 'shape'):
                logger.info(f"[DEBUG] LIST result shape: {result.shape}")
            if hasattr(result, 'columns'):
                logger.info(f"[DEBUG] LIST result columns: {list(result.columns)}")

            # Parse result to extract filenames
            files = self._parse_list_result(result)
            logger.info(f"[DEBUG] Parsed files: {files}")

            # Filter by configured patterns
            patterns = self.config.get("file_patterns", ["*.yml", "*.yaml"])
            filtered_files = self._filter_files_by_patterns(files, patterns)

            logger.info(f"Found {len(filtered_files)} semantic model files: {filtered_files}")
            return filtered_files

        except Exception as exc:
            logger.error(f"Failed to list semantic models: {exc}")
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": f"Failed to list semantic models from volume: {exc}",
                    "sql": sql if 'sql' in locals() else "LIST volume command"
                }
            ) from exc

    def read_model(self, model_name: str) -> str:
        """Read semantic model file content from volume.

        Args:
            model_name: Name of the model file to read

        Returns:
            Raw content of the semantic model file

        Raises:
            DatusException: If reading fails
        """
        try:
            # Use the existing volume file reading capability
            if self.volume_type == "user":
                # For user volume, construct the correct path
                # The LIST command shows paths like "semantic_models/file.yaml"
                # So we need to use this exact path without duplication
                volume_path = self.volume_path.strip("/")  # "semantic_models"
                file_path = f"{volume_path}/{model_name}"  # "semantic_models/Test006.yaml"
                volume_uri = "volume:user://~/"

                logger.info(f"[DEBUG] Reading model '{model_name}':")
                logger.info(f"[DEBUG]   volume_uri: '{volume_uri}'")
                logger.info(f"[DEBUG]   file_path: '{file_path}'")
                logger.info(f"[DEBUG]   volume_type: '{self.volume_type}'")
                logger.info(f"[DEBUG]   volume_path config: '{self.volume_path}'")

                return self.connector.read_volume_file(volume_uri, file_path)
            else:
                # For named volume
                volume_uri = f"volume:{self.volume_name}"
                file_path = f"{self.volume_path.strip('/')}/{model_name}"

                logger.info(f"[DEBUG] Reading model '{model_name}' (named volume):")
                logger.info(f"[DEBUG]   volume_uri: '{volume_uri}'")
                logger.info(f"[DEBUG]   file_path: '{file_path}'")

                return self.connector.read_volume_file(volume_uri, file_path)

        except Exception as exc:
            logger.error(f"Failed to read semantic model {model_name}: {exc}")
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": f"Failed to read semantic model {model_name}: {exc}",
                    "sql": f"GET volume file {model_name}"
                }
            ) from exc

    def exists(self, model_name: str) -> bool:
        """Check if semantic model file exists in volume.

        Args:
            model_name: Name of the model file

        Returns:
            True if file exists, False otherwise
        """
        try:
            models = self.list_models()
            return model_name in models
        except Exception as exc:
            logger.warning(f"Failed to check if model {model_name} exists: {exc}")
            return False

    def get_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for semantic model file.

        Args:
            model_name: Name of the model file

        Returns:
            Dictionary containing metadata
        """
        try:
            # For now, return basic metadata
            # ClickZetta LIST command may provide size and timestamp info
            return {
                "name": model_name,
                "size": None,  # Would need to parse from LIST command output
                "modified_time": None,  # Would need to parse from LIST command output
                "volume_type": self.volume_type,
                "volume_name": self.volume_name,
                "volume_path": self.volume_path
            }
        except Exception as exc:
            logger.warning(f"Failed to get metadata for {model_name}: {exc}")
            return {"name": model_name, "error": str(exc)}

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a semantic model file.

        Args:
            model_name: Name of the model file

        Returns:
            Dictionary containing model information including metadata and format
        """
        try:
            # Get basic metadata
            info = self.get_metadata(model_name)

            # Try to read and parse content to determine format and model name
            try:
                content = self.read_model(model_name)
                if content:
                    # Try to parse YAML to extract model information
                    import yaml
                    try:
                        # Try parsing as YAML
                        docs = list(yaml.safe_load_all(content))
                        for doc in docs:
                            if isinstance(doc, dict):
                                # Check for data_source (ClickZetta format)
                                if 'data_source' in doc:
                                    info['format'] = 'clickzetta'
                                    info['model_name'] = doc.get('data_source', {}).get('name', model_name.replace('.yaml', '').replace('.yml', ''))
                                    info['description'] = doc.get('data_source', {}).get('description', '')
                                    break
                                # Check for semantic_model (MetricFlow format)
                                elif 'semantic_model' in doc:
                                    info['format'] = 'metricflow'
                                    info['model_name'] = doc.get('semantic_model', {}).get('name', model_name.replace('.yaml', '').replace('.yml', ''))
                                    info['description'] = doc.get('semantic_model', {}).get('description', '')
                                    break

                        info['content_length'] = len(content)

                    except yaml.YAMLError as e:
                        info['format'] = 'unknown'
                        info['model_name'] = 'unknown'
                        info['content_error'] = f"YAML parsing error: {e}"
                else:
                    info['format'] = 'unknown'
                    info['model_name'] = 'unknown'
                    info['content_length'] = 0

            except Exception as e:
                info['format'] = 'unknown'
                info['model_name'] = 'unknown'
                info['content_error'] = str(e)
                info['content_length'] = 0

            return info

        except Exception as exc:
            logger.warning(f"Failed to get model info for {model_name}: {exc}")
            return {"name": model_name, "error": str(exc)}

    def _parse_list_result(self, result) -> List[str]:
        """Parse result from LIST command to extract filenames.

        Args:
            result: Result from ClickZetta LIST command (can be ClickZetta result object or pandas DataFrame)

        Returns:
            List of filenames
        """
        files = []
        try:
            # Handle different result formats from ClickZetta
            df = None
            if hasattr(result, 'to_pandas'):
                # ClickZetta result object
                logger.info(f"[DEBUG] Converting ClickZetta result to pandas")
                df = result.to_pandas()
            elif hasattr(result, 'columns'):
                # Already a pandas DataFrame
                logger.info(f"[DEBUG] Result is already pandas DataFrame")
                df = result

            if df is not None:
                logger.info(f"[DEBUG] DataFrame columns: {list(df.columns)}")
                logger.info(f"[DEBUG] DataFrame shape: {df.shape}")
                if len(df) > 0:
                    logger.info(f"[DEBUG] First few rows:")
                    for i, row in df.head().iterrows():
                        logger.info(f"[DEBUG]   Row {i}: {dict(row)}")

                if 'relative_path' in df.columns:
                    files = df['relative_path'].tolist()
                    logger.info(f"[DEBUG] Using 'relative_path' column: {files}")
                elif 'file_name' in df.columns:
                    files = df['file_name'].tolist()
                    logger.info(f"[DEBUG] Using 'file_name' column: {files}")
                elif 'name' in df.columns:
                    files = df['name'].tolist()
                    logger.info(f"[DEBUG] Using 'name' column: {files}")
                else:
                    # Fallback: try to extract from first column
                    if len(df.columns) > 0:
                        files = df.iloc[:, 0].tolist()
                        logger.info(f"[DEBUG] Using first column '{df.columns[0]}': {files}")

            # Filter out directories and non-files
            filtered_files = [f for f in files if f and isinstance(f, str) and '.' in f]
            logger.info(f"[DEBUG] After filtering non-files: {filtered_files}")

            # Extract just the filename from paths like "semantic_models/file.yaml"
            # We want to return just "file.yaml" for compatibility with the rest of the system
            filenames = []
            for f in filtered_files:
                if '/' in f:
                    # Extract filename from path
                    filename = f.split('/')[-1]
                    filenames.append(filename)
                    logger.info(f"[DEBUG] Extracted filename '{filename}' from path '{f}'")
                else:
                    filenames.append(f)
                    logger.info(f"[DEBUG] Using filename as-is: '{f}'")

            logger.info(f"[DEBUG] Final filenames: {filenames}")
            return filenames

        except Exception as exc:
            logger.error(f"Failed to parse LIST result: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return files