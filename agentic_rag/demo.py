"""The scripted demo: five questions that exercise each part of the agentic-RAG loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_rag.runner import Researcher


@dataclass(frozen=True)
class Step:
    question: str
    note: str


STEPS = (
    Step("What is the default port in Lumen 2.0?", "filters from the question"),
    Step(
        "What is the default port in Lumen 1.4 and in Lumen 3.0?",
        "decomposition: one filtered search per version",
    ),
    Step(
        "What is the default election timeout of the feature that replaced Mirror replication?",
        "multi-hop: find the replacement first, then search for it",
    ),
    Step(
        "How do I list the releases on each host in Ferry 2.0?",
        "self-check fails, the filter is relaxed and the agent searches again",
    ),
    Step("What is the capital of France?", "refusal: nothing in the docs supports an answer"),
)


def _filters(search: dict[str, Any]) -> str:
    filters = search.get("filters") or {}
    return ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"


async def run_demo(
    researcher: Researcher, echo: Callable[[str], None] = print
) -> list[dict[str, Any]]:
    """Ask every question in :data:`STEPS`; returns what a test (or CI) can check."""
    results = []
    for i, step in enumerate(STEPS, 1):
        echo(f"\n=== {i}. {step.note}")
        echo(f"> {step.question}")
        answer = await researcher.ask(step.question)
        for n, search in enumerate(answer.searches, 1):
            hits = len(search.get("hits") or [])
            query = search.get("query")
            echo(f"  hop {n}: search_docs({query!r}; {_filters(search)}) -> {hits} passages")
        echo(answer.reply)
        results.append(answer.as_dict())
    return results
