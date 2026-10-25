"""Per-invocation retrieval state: hops used, passages retrieved, the answer's citations.

It lives in session state under ``rag:trace`` and is reset whenever a new invocation (a
new user message) starts, so the hop budget applies per question. Only ids and short
metadata are kept - never passage text - so the session stays small and nothing
retrieved is written back anywhere.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

TRACE_KEY = "rag:trace"


def trace(state: MutableMapping[str, Any] | Any, invocation_id: str) -> dict[str, Any]:
    """The trace of ``invocation_id``, created (and the previous one dropped) on first use."""
    current = state.get(TRACE_KEY)
    if not isinstance(current, dict) or current.get("invocation") != invocation_id:
        current = {
            "invocation": invocation_id,
            "hops": 0,
            "limit_hit": False,
            "searches": [],
            "retrieved": {},
            "answer": None,
        }
    return dict(current)


def save(state: MutableMapping[str, Any] | Any, value: dict[str, Any]) -> None:
    # Assign a fresh dict so ADK records the change in the event's state delta.
    state[TRACE_KEY] = dict(value)
