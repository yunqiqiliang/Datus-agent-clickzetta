# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.node_models import SemanticModelPayload, SqlTask
from datus.schemas.semantic_model_node_models import SemanticModelInput, SemanticModelResult
from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.tools.semantic_models import SemanticModelRepository, SemanticModelRepositoryError
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SemanticModelNode(Node):
    """Node that loads semantic model specifications into workflow context."""

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[SemanticModelInput] = None,
        agent_config=None,
        tools=None,
    ):
        super().__init__(node_id, description, node_type, input_data, agent_config, tools)
        self._repository = SemanticModelRepository(agent_config) if agent_config else None

    def setup_input(self, workflow: Workflow) -> Dict:
        task = workflow.task or SqlTask(task="")
        defaults = self.agent_config.semantic_model_defaults()
        require_semantic_model = (task.context_strategy or defaults.default_strategy) == "semantic_model"
        self.input = SemanticModelInput(require_semantic_model=require_semantic_model)
        return {"success": True, "message": "Semantic model preferences captured"}

    def _initialize(self):  # type: ignore[override]
        """Semantic model node does not require LLM initialization."""
        return

    def execute(self):
        workflow = getattr(self, "workflow", None)
        if workflow is None or workflow.task is None:
            self.result = SemanticModelResult(
                success=False,
                error="Workflow task is not available for semantic model loading.",
                loaded=False,
            )
            return

        repository = self._repository or SemanticModelRepository(self.agent_config)

        try:
            connector = self._maybe_get_connector(workflow)
        except SemanticModelRepositoryError as exc:
            logger.warning("Semantic model load aborted: %s", exc)
            self.result = SemanticModelResult(success=False, error=str(exc), loaded=False)
            return
        try:
            payload = repository.load(workflow.task, connector)
        except SemanticModelRepositoryError as exc:
            logger.warning("Semantic model load failed: %s", exc)
            self.result = SemanticModelResult(success=False, error=str(exc), loaded=False)
            return

        if payload is None:
            self.result = SemanticModelResult(success=True, semantic_model=None, loaded=False)
        else:
            self.result = SemanticModelResult(success=True, semantic_model=payload, loaded=True)

    async def execute_stream(self, action_history_manager=None):  # type: ignore[override]
        self.execute()
        return

    def update_context(self, workflow: Workflow) -> Dict:
        if not isinstance(self.result, SemanticModelResult):
            return {"success": False, "message": "Unexpected result type for semantic model node"}

        payload: Optional[SemanticModelPayload] = self.result.semantic_model
        if not payload:
            return {"success": True, "message": "No semantic model loaded"}

        workflow.context.semantic_model = payload
        # Clear stale schema/value context so downstream nodes rely on semantic model data.
        workflow.context.table_schemas = []
        workflow.context.table_values = []
        workflow.context.metrics = []

        return {
            "success": True,
            "message": f"Semantic model '{payload.name or payload.source}' loaded successfully",
        }

    def _maybe_get_connector(self, workflow: Workflow):
        task = workflow.task
        if task.semantic_model_local_path.strip():
            return None

        # Only attempt connector retrieval when a ClickZetta volume is expected.
        try:
            connector = self._sql_connector(task.database_name or "")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Unable to obtain database connector for semantic model loading: %s", exc)
            return None

        if task.semantic_model_volume.strip():
            if isinstance(connector, ClickzettaConnector):
                return connector
            logger.debug(
                "Semantic model volume specified but active connector is %s; volume read is unsupported.",
                type(connector).__name__,
            )
            if task.context_strategy == "semantic_model":
                raise SemanticModelRepositoryError(
                    "Semantic model volume provided but connector does not support volume reads."
                )
        elif task.database_type.lower() == DBType.CLICKZETTA.value.lower():
            # Default to ClickZetta connector for auto mode even if volume not explicitly provided.
            if isinstance(connector, ClickzettaConnector):
                return connector

        return None
