"""The two callbacks that keep the agent honest.

* :class:`FilterHint` (``before_model_callback``) tells the model which products and
  versions exist and which filters the question seems to imply (rule-based, from
  :func:`~agentic_rag.filters.extract_filters`). The model is free to ignore them; small
  local models use them a lot.
* :class:`CitationGuard` (``after_model_callback``) checks every final answer:
  citations must name passages ``search_docs`` returned *in this invocation*; unknown ids
  are removed; an answer left without a valid citation becomes a refusal; and a model
  that keeps asking for searches after the hop budget is spent is stopped with a refusal.
"""

from __future__ import annotations

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agentic_rag import trace as tr
from agentic_rag.filters import extract_filters
from agentic_rag.filters import mentioned_versions
from agentic_rag.tools import TOOL_NAME
from agentic_rag.tools import RetrieverHandle

REFUSAL = "I can't answer that from the documentation."
CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._#:/-]{0,199})\]")
_REFUSAL = re.compile(
    r"\b(can(?:'|’|no)?t|cannot|unable to|not able to) (?:answer|find)\b"
    r".{0,40}\b(doc|documentation|passages|sources)\b"
    r"|\bnot (?:covered|mentioned|answered) (?:in|by) the (?:doc|documentation|passages)\b",
    re.I | re.S,
)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
HINT_PREFIX = "Filters detected in the question"
VERSIONS_PREFIX = "Versions mentioned"


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text))


def response_text(response: LlmResponse) -> str:
    parts = response.content.parts if response.content and response.content.parts else []
    return "".join(p.text or "" for p in parts if not p.thought)


def _text_response(text: str, original: LlmResponse) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        usage_metadata=original.usage_metadata,
    )


class FilterHint:
    """``before_model_callback``: the catalog, plus filters the question implies."""

    __name__ = "filter_hint"

    def __init__(self, handle: RetrieverHandle) -> None:
        self.handle = handle

    def __call__(self, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        catalog = self.handle.get().catalog
        products = catalog.products()
        lines = [
            "Known products and versions: "
            + "; ".join(f"{p} ({', '.join(vs)})" for p, vs in products.items())
            + ". Doc types: "
            + ", ".join(catalog.doc_types())
            + "."
        ]
        user = callback_context.user_content
        question = " ".join(p.text or "" for p in (user.parts or [])) if user else ""
        if question:
            found = extract_filters(question, products)
            if not found.is_empty():
                lines.append(f"{HINT_PREFIX}: {found.describe()}.")
            versions = mentioned_versions(question, products)
            if len(versions) > 1:
                lines.append(
                    f"{VERSIONS_PREFIX}: {', '.join(versions)}. Search each version separately."
                )
        llm_request.append_instructions(lines)


class CitationGuard:
    """``after_model_callback``: citations must exist, the hop budget must hold."""

    __name__ = "citation_guard"

    def __call__(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.partial or not llm_response.content:
            return None
        state = tr.trace(callback_context.state, callback_context.invocation_id)
        calls = [
            p.function_call for p in llm_response.content.parts or [] if p.function_call is not None
        ]
        if calls:
            searching = any(c.name == TOOL_NAME for c in calls)
            if searching and state["limit_hit"]:
                # The budget was spent and the model was told so; it asked again.
                state["answer"] = {"status": "refused", "reason": "hop_limit", "citations": []}
                tr.save(callback_context.state, state)
                return _text_response(REFUSAL, llm_response)
            return None
        text = _THINK.sub("", response_text(llm_response)).strip()
        if not text:
            return None
        retrieved: dict[str, dict[str, str]] = state["retrieved"]
        cited = list(dict.fromkeys(CITATION.findall(text)))
        valid = [c for c in cited if c in retrieved]
        dropped = [c for c in cited if c not in retrieved]
        if is_refusal(text) or not valid:
            reason = "model_refused" if is_refusal(text) else "no_valid_citation"
            state["answer"] = {
                "status": "refused",
                "reason": reason,
                "citations": [],
                "dropped": dropped,
            }
            tr.save(callback_context.state, state)
            return _text_response(REFUSAL, llm_response)
        for bad in dropped:
            text = text.replace(f"[{bad}]", "")
        text = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
        sources = "\n".join(
            f"- [{c}] {retrieved[c]['title']} - {retrieved[c]['section']}" for c in valid
        )
        state["answer"] = {"status": "answered", "citations": valid, "dropped": dropped}
        tr.save(callback_context.state, state)
        return _text_response(f"{text}\n\nSources:\n{sources}", llm_response)
