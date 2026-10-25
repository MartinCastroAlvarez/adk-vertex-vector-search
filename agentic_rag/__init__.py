"""agentic_rag: an agentic RAG agent in Google ADK, built for Vertex AI Vector Search.

``import agentic_rag`` builds nothing, opens no index and imports no Google Cloud SDK.
ADK's loaders import ``agentic_rag.agent`` (which builds ``root_agent`` from the
environment); the ``agent`` attribute is resolved lazily so they find it.
"""

from __future__ import annotations

import importlib
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.researcher import build_agent
from agentic_rag.retrieval.base import VectorStore
from agentic_rag.retrieval.factory import open_retriever
from agentic_rag.runner import Answer
from agentic_rag.runner import Researcher

__all__ = ["Answer", "Researcher", "Settings", "VectorStore", "build_agent", "open_retriever"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "agent":
        return importlib.import_module("agentic_rag.agent")
    raise AttributeError(f"module 'agentic_rag' has no attribute {name!r}")
