# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract storage adapter for semantic models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class SemanticModelStorageAdapter(ABC):
    """Abstract base class for semantic model storage adapters.

    This adapter provides a unified interface for accessing semantic models
    from various storage backends (volumes, stages, S3, local files, etc.).
    """

    def __init__(self, connector: Any, config: Dict[str, Any]):
        """Initialize storage adapter.

        Args:
            connector: Database connector instance
            config: Storage configuration dictionary
        """
        self.connector = connector
        self.config = config

    @abstractmethod
    def list_models(self) -> List[str]:
        """List all available semantic model files.

        Returns:
            List of semantic model filenames
        """
        pass

    @abstractmethod
    def read_model(self, model_name: str) -> str:
        """Read semantic model file content.

        Args:
            model_name: Name of the model file to read

        Returns:
            Raw content of the semantic model file

        Raises:
            DatusException: If model file cannot be read
        """
        pass

    @abstractmethod
    def exists(self, model_name: str) -> bool:
        """Check if semantic model file exists.

        Args:
            model_name: Name of the model file

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def get_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for semantic model file.

        Args:
            model_name: Name of the model file

        Returns:
            Dictionary containing metadata (size, modified_time, etc.)
        """
        pass

    def write_model(self, model_name: str, content: str) -> bool:
        """Write semantic model file content.

        This is an optional operation that may not be supported by all adapters.

        Args:
            model_name: Name of the model file to write
            content: Content to write

        Returns:
            True if successful, False otherwise

        Raises:
            NotImplementedError: If write operation is not supported
        """
        raise NotImplementedError("Write operation not supported by this adapter")

    def delete_model(self, model_name: str) -> bool:
        """Delete semantic model file.

        This is an optional operation that may not be supported by all adapters.

        Args:
            model_name: Name of the model file to delete

        Returns:
            True if successful, False otherwise

        Raises:
            NotImplementedError: If delete operation is not supported
        """
        raise NotImplementedError("Delete operation not supported by this adapter")

    def _filter_files_by_patterns(self, files: List[str], patterns: List[str]) -> List[str]:
        """Filter files by filename patterns.

        Args:
            files: List of filenames
            patterns: List of patterns (e.g., ['*.yml', '*.yaml'])

        Returns:
            Filtered list of filenames
        """
        import fnmatch

        filtered = []
        for file in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file, pattern):
                    filtered.append(file)
                    break
        return filtered

    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any of the patterns.

        Args:
            filename: Name of the file
            patterns: List of patterns to match against

        Returns:
            True if filename matches any pattern
        """
        import fnmatch

        return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)