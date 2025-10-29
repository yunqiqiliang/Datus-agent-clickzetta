import os
from typing import Any, Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DashscopeModel(OpenAICompatibleModel):
    """
    Dashscope (通义千问) model implementation using the OpenAI-compatible REST API.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._async_client = None

    def _get_api_key(self) -> str:
        """Prefer configuration value, fallback to DASHSCOPE_API_KEY environment variable."""
        api_key = self.model_config.api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Dashscope API key must be provided or set as DASHSCOPE_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """
        Dashscope exposes an OpenAI-compatible endpoint.
        Allow overriding via config while defaulting to the official gateway.
        """
        return self.model_config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        Override generate to pass through enable_thinking flag when explicitly requested.
        Dashscope supports an `enable_thinking` parameter through extra_body.
        """
        if enable_thinking:
            kwargs = kwargs.copy()
            extra_body = kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = True
            kwargs["extra_body"] = extra_body
        return super().generate(prompt, enable_thinking=enable_thinking, **kwargs)
