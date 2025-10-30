# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SemanticModelPayload


class SemanticModelInput(BaseInput):
    """Input parameters controlling semantic model loading."""

    require_semantic_model: bool = Field(
        default=False,
        description="If True, raise an error when semantic model cannot be loaded.",
    )


class SemanticModelResult(BaseResult):
    """Result of semantic model loading."""

    semantic_model: Optional[SemanticModelPayload] = Field(
        default=None, description="Semantic model payload that was loaded (if any)"
    )
    loaded: bool = Field(default=False, description="Whether a semantic model was successfully loaded")
