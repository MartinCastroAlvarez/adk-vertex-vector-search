"""``search_docs``: the agent's only tool, a filtered hybrid search with a hop budget.

Why a custom ``FunctionTool`` and not ADK's ``VertexAiRagRetrieval``: that tool targets
Vertex AI **RAG Engine** corpora, and on Gemini models it is a *built-in* retrieval the
model runs server-side - the agent never sees a tool call, can't pass per-call metadata
filters, numeric filters, crowding or an RRF weight, can't count hops, and only works
with Gemini on Google Cloud. Here every search is an ordinary function call, so the model
decides when and what to search, the code enforces the budget and records exactly which
passages came back (what citations are checked against), and the same tool runs on a
local index and ollama for free.

Arguments come from the model and are untrusted: the query is capped, filters are
checked against the catalog (an unknown product or version is an error the model can
correct), and passages are neutralised before they reach the context.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from google.adk.tools.tool_context import ToolContext

from agentic_rag import trace as tr
from agentic_rag.filters import FilterError
from agentic_rag.filters import validate_filters
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.sanitize import neutralise

MAX_QUERY_CHARS = 300
TOOL_NAME = "search_docs"


class RetrieverHandle:
    """Opens the retriever on first use, so importing the agent touches no index or model."""

    def __init__(self, opener: Callable[[], Retriever]) -> None:
        self._opener = opener
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    def get(self) -> Retriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = self._opener()
            return self._retriever

    @property
    def opened(self) -> bool:
        return self._retriever is not None

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()


def make_search_docs(
    handle: RetrieverHandle, max_hops: int = 3, top_k: int = 5
) -> Callable[..., dict[str, Any]]:
    """Build the ``search_docs`` function the agent calls (one per agent)."""

    def search_docs(
        query: str,
        tool_context: ToolContext,
        product: str | None = None,
        version: str | None = None,
        since_version: str | None = None,
        doc_type: str | None = None,
        released_after: str | None = None,
        released_before: str | None = None,
    ) -> dict[str, Any]:
        """Search the product documentation (hybrid keyword + semantic search).

        Args:
            query: a short keyword query, e.g. "default port" or "rollout abort threshold".
            product: only this product, e.g. "lumen" or "ferry".
            version: only this version: exact ("2.0") or a major version ("2").
            since_version: only versions >= this one, e.g. "2.0".
            doc_type: one of guide, reference, release-notes, troubleshooting.
            released_after: only docs released on or after this date (YYYY-MM-DD or YYYY).
            released_before: only docs released before this date (YYYY-MM-DD or YYYY).

        Returns:
            passages with an id to cite in square brackets, and how many searches are left.
        """
        state = tr.trace(tool_context.state, tool_context.invocation_id)
        if state["hops"] >= max_hops:
            state["limit_hit"] = True
            tr.save(tool_context.state, state)
            return {
                "status": "hop_limit",
                "hops_left": 0,
                "message": f"All {max_hops} searches are used. Answer from the passages you "
                "already have, citing them, or reply: I can't answer that from the "
                "documentation.",
            }
        state["hops"] += 1
        hops_left = max_hops - state["hops"]
        text = " ".join(str(query or "").split())[:MAX_QUERY_CHARS]
        retriever = handle.get()
        try:
            filters = validate_filters(
                retriever.catalog.products(),
                retriever.catalog.doc_types(),
                product=product,
                version=version,
                since_version=since_version,
                doc_type=doc_type,
                released_after=released_after,
                released_before=released_before,
            )
        except FilterError as exc:
            state["searches"].append({"query": text, "error": str(exc)})
            tr.save(tool_context.state, state)
            return {"status": "invalid_filter", "error": str(exc), "hops_left": hops_left}
        if not text:
            tr.save(tool_context.state, state)
            return {"status": "invalid_query", "error": "query is empty", "hops_left": hops_left}
        hits = retriever.search(text, filters, top_k=top_k)
        retrieved = dict(state["retrieved"])
        for hit in hits:
            retrieved[hit.chunk.chunk_id] = {
                "title": hit.chunk.title,
                "section": hit.chunk.section,
                "source": hit.chunk.source,
            }
        state["retrieved"] = retrieved
        state["searches"].append(
            {
                "query": text,
                "filters": filters.as_dict(),
                "hits": [h.chunk.chunk_id for h in hits],
            }
        )
        tr.save(tool_context.state, state)
        if not hits:
            return {
                "status": "no_results",
                "filters": filters.describe(),
                "hops_left": hops_left,
                "message": "Nothing matched. Rephrase the query or loosen the filters.",
            }
        return {
            "status": "ok",
            "filters": filters.describe(),
            "hops_left": hops_left,
            "passages": [
                {
                    "id": h.chunk.chunk_id,
                    "title": h.chunk.title,
                    "section": h.chunk.section,
                    "product": h.chunk.product,
                    "version": h.chunk.version,
                    "text": neutralise(h.chunk.text),
                }
                for h in hits
            ],
        }

    search_docs.__name__ = TOOL_NAME
    return search_docs
