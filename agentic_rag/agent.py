"""ADK entry point: ``root_agent`` for ``adk web``, ``adk run`` and ``adk eval``.

The model comes from ``RAG_PROVIDER`` (``ollama`` by default, ``stub`` offline, ``gemini``
with your own key) and the index from ``RAG_STORE`` (``local`` by default). Building the
agent touches neither: the index is opened, and built from the bundled docs if missing,
on the first question.
"""

from __future__ import annotations

from agentic_rag.config import Settings
from agentic_rag.researcher import build_agent

root_agent = build_agent(Settings.from_env())
