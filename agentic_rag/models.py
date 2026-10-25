"""Which model the agent runs on.

* ``ollama`` (default): ``LiteLlm(model="ollama_chat/<model>")`` on a loopback ollama.
* ``gemini`` (optional): ADK's ``Gemini`` with your own ``GOOGLE_API_KEY``. Never used
  by tests or CI.
* ``stub``: :class:`agentic_rag.offline.OfflineResearcher`, deterministic and offline.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from google.adk.models.base_llm import BaseLlm

from agentic_rag.config import ConfigError
from agentic_rag.config import Settings


def ollama_model_id(model: str) -> str:
    """``qwen3:4b`` -> ``ollama_chat/qwen3:4b`` (the LiteLLM provider that supports tools)."""
    return model if model.startswith(("ollama_chat/", "ollama/")) else f"ollama_chat/{model}"


def agent_model(settings: Settings, env: Mapping[str, str] | None = None) -> BaseLlm:
    env = os.environ if env is None else env
    if settings.provider == "stub":
        from agentic_rag.offline import OfflineResearcher

        return OfflineResearcher()
    if settings.provider == "gemini":
        if not (env.get("GOOGLE_API_KEY") or env.get("GOOGLE_GENAI_USE_VERTEXAI") == "1"):
            raise ConfigError(
                "RAG_PROVIDER=gemini needs GOOGLE_API_KEY (your key, your bill); "
                "the default provider is a local ollama"
            )
        from google.adk.models.google_llm import Gemini

        return Gemini(model=settings.model)
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=ollama_model_id(settings.model), api_base=settings.ollama_host)
